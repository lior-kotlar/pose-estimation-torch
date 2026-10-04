"""The cluster side of local_reanalysis.py, run over ssh with the system python3.

local_reanalysis.py (on a PC) talks to the cluster only by running this script over ssh and
piping data through it, so the whole exchange is one connection per step and needs nothing on
the cluster beyond this file and a python3. Standard library only, on purpose: it runs with
whatever python3 the node has, not the project's .env. It never runs on the login gateway --
the PC hands every call to srun, so this script's work happens on a compute node.

    python3 local_reanalysis_server.py declarations
        stdin : a JSON list of perturbation.json paths the PC's movies may need
        stdout: a tar of the ones that exist, each stored under its absolute path minus the
                leading '/'

    python3 local_reanalysis_server.py receive --dest <collected_h5 dir>
        stdin : a tar whose first member is MANIFEST.json ({"files": {relpath: sha256}}),
                followed by exactly those files
        stdout: one JSON line, {"ok": true, "stamp": ..., "results": {relpath: status}}

receive unpacks into a hidden staging folder inside <dest>, checks every file's sha256 against
the manifest, and only then installs them: an identical file already in place is 'unchanged',
a different one is moved into superseded_<stamp>/ next to it ('replaced'), a missing one is
'new'. Nothing is installed unless every file arrived intact.

The remaining verbs run one round of code/realign_ensemble.py for a PC, over the ensemble
members it uploaded (with receive --dest <realign_jobs/JOB>/inputs). They all take --job, a
plain name that can only ever be a single folder under realign_jobs/:

    round-submit --job JOB --kind KIND [--throttle N]  submit the array job; one JSON line
    round-status --job JOB --kind KIND                per-movie state, and what slurm says
    round-fetch  --job JOB --kind KIND                stdout: a gzipped tar of what finished
    round-clean  --job JOB                            delete the job folder once it is home

KIND is 'realign' (re-run a movie's ensemble from its members) or 'video' (reproject and render
the overlay mp4 from an analysis h5 and a box). Each decides what runs on the node, what it asks
slurm for, and what a finished movie hands back.

submit and clean touch only a job folder the caller owns; every path they build is checked to
stay inside realign_jobs/JOB. Nothing here re-analyses anything: the PC does that, with the
declarations and path maps it already has.

The predict-* verbs run a PC's prep + predict round (code/local_predict.py) and live in
code/local_predict_server.py, which this script hands them to.
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

MANIFEST = "MANIFEST.json"
DECLARATION_NAME = "perturbation.json"
DIR_MODE = 0o2755   # setgid, so new folders keep the project's group like the rest of the tree
FILE_MODE = 0o644

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_ROOT = os.path.join(PROJECT, "realign_jobs")
INPUTS = "inputs"                  # where the PC uploads each movie's ensemble members
MOVIE_MANIFEST = "movies.txt"      # one movie dir per line, the array job's task list
JOB_RECORD = "job.json"
POINTS_ALL = "points_3D_all.npy"   # one per ensemble member: the only real input
ANALYSIS_SUFFIX = "_analysis_smoothed.h5"
MP4 = "movie 2D and 3D.mp4"
REPROJECTED = "points_ensemble_smoothed_reprojected.npy"
VIDEO_STAMP = "video.json"
REALIGN_MARKER = ".realigned_ensemble.json"
BLOCKED = os.path.join(".realign_staging", "BLOCKED.json")
ARRAY_SCRIPT = os.path.join("sbatch_files", "round_array.sh")
# What each kind of round runs on a node, what it asks slurm for, and what a finished movie
# yields. The entry point is an argument to the array script, so one script serves both.
KINDS = {
    "realign": {
        "script": os.path.join("code", "realign_ensemble.py"),
        "args": ["--no-reanalyse"],
        # measured: a realignment peaks around 2 GB and takes well under an hour, so the old
        # 32g/8h ask was keeping tasks behind higher-priority work for nothing
        "sbatch": ["--cpus-per-task=16", "--mem=16g", "--time=05:00:00", "--partition=glacier"],
    },
    "video": {
        "script": os.path.join("code", "reanalyse_movies.py"),
        "args": ["--only-video", "--force-mp4", "--perturbation-source", "h5"],
        # rendering is one long single-threaded encode, not a parallel search
        "sbatch": ["--cpus-per-task=2", "--mem=8g", "--time=04:00:00", "--partition=glacier"],
    },
}
# slurm is not always on a shell's PATH, nor its config in the environment; moriah keeps both here
SLURM_BIN = "/vol/slurm/moriah/bindir/bin"
SLURM_CONF = "/vol/slurm/moriah/slurm.conf"
JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relpath(name):
    """A tar member name that stays inside the folder it is unpacked into."""
    parts = name.replace("\\", "/").split("/")
    return (bool(name) and not name.startswith("/") and ".." not in parts
            and all(parts) and ":" not in parts[0])


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        try:
            os.chmod(path, DIR_MODE)
        except OSError:
            pass


def declarations():
    wanted = json.load(sys.stdin)
    out = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
    sent = set()
    for path in wanted:
        path = os.path.normpath(str(path))
        if (os.path.basename(path) != DECLARATION_NAME or not os.path.isabs(path)
                or path in sent or not os.path.isfile(path)):
            continue
        out.add(path, arcname=path.lstrip("/"), recursive=False)
        sent.add(path)
    out.close()
    sys.stdout.buffer.flush()
    return 0


def receive(dest):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    makedirs(dest)
    staging = os.path.join(dest, ".incoming_%s_%d" % (stamp, os.getpid()))
    os.makedirs(staging)
    try:
        manifest = None
        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tar:
            for member in tar:
                if member.name == MANIFEST:
                    manifest = json.loads(tar.extractfile(member).read().decode("utf-8"))["files"]
                    continue
                if manifest is None:
                    raise ValueError("the upload did not start with %s" % MANIFEST)
                if not member.isfile() or not safe_relpath(member.name) or member.name not in manifest:
                    raise ValueError("unexpected entry in the upload: %r" % member.name)
                target = os.path.join(staging, *member.name.split("/"))
                if not os.path.isdir(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                source = tar.extractfile(member)
                with open(target, "wb") as f:
                    shutil.copyfileobj(source, f)
        if manifest is None:
            raise ValueError("empty upload")

        for rel, digest in manifest.items():
            staged = os.path.join(staging, *rel.split("/"))
            if not os.path.isfile(staged):
                raise ValueError("%s is in the manifest but was not received" % rel)
            if sha256(staged) != digest:
                raise ValueError("%s arrived damaged (checksum mismatch)" % rel)

        results = {}
        for rel in sorted(manifest):
            staged = os.path.join(staging, *rel.split("/"))
            final = os.path.join(dest, *rel.split("/"))
            if os.path.isfile(final):
                if sha256(final) == manifest[rel]:
                    results[rel] = "unchanged"
                    continue
                archive = os.path.join(os.path.dirname(final), "superseded_%s" % stamp)
                makedirs(archive)
                os.replace(final, os.path.join(archive, os.path.basename(final)))
                results[rel] = "replaced"
            else:
                results[rel] = "new"
            parent = os.path.dirname(final)
            missing = []
            while parent and not os.path.isdir(parent):
                missing.append(parent)
                parent = os.path.dirname(parent)
            for folder in reversed(missing):
                makedirs(folder)
            os.replace(staged, final)
            os.chmod(final, FILE_MODE)
        print(json.dumps({"ok": True, "stamp": stamp, "dest": dest, "results": results}))
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}))
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# -- the realignment round ------------------------------------------------------------------------

def job_dir(job):
    """The one folder a --job name may name, refusing anything that could point elsewhere."""
    if not JOB_NAME.match(job or ""):
        raise ValueError("a job name may hold letters, digits, dot, dash and underscore only")
    path = os.path.join(JOBS_ROOT, job)
    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(JOBS_ROOT):
        raise ValueError("a job name may not name a path")
    return path


def inside(root, path):
    """True when path stays inside root, so a walked name can never lead out of the job."""
    root = os.path.abspath(root)
    path = os.path.abspath(path)
    return path == root or path.startswith(root + os.sep)


def owned_job(job):
    """The job folder, which must exist and belong to whoever is asking."""
    path = job_dir(job)
    if not os.path.isdir(path):
        raise ValueError("no such job: %s (upload its movies first)" % job)
    if os.stat(path).st_uid != os.getuid():
        raise ValueError("job %s belongs to another account; only its owner may run or delete it"
                         % job)
    return path


def is_realign_movie(dirpath, dirnames, filenames):
    """A movie to realign: two or more members with 3D candidates."""
    members = [d for d in dirnames if os.path.isfile(os.path.join(dirpath, d, POINTS_ALL))]
    return len(members) >= 2


def is_video_movie(dirpath, dirnames, filenames):
    """A movie to render: an analysis h5 to render from."""
    return any(f.endswith(ANALYSIS_SUFFIX) for f in filenames)


def job_movies(inputs, kind="realign"):
    """Every uploaded movie folder of this round."""
    looks_like = is_video_movie if kind == "video" else is_realign_movie
    movies = []
    for dirpath, dirnames, filenames in os.walk(inputs):
        dirnames[:] = sorted(d for d in dirnames
                             if not (d.startswith("superseded_") or d.startswith(".")))
        if looks_like(dirpath, dirnames, filenames):
            movies.append(dirpath)
            dirnames[:] = []       # a movie holds no movies
    return sorted(movies)


def slurm(*command, env=None):
    """A slurm command, from the PATH or from the place moriah keeps it. `env` replaces the
    environment it runs in (sbatch hands its environment on to the job)."""
    name = command[0]
    program = shutil.which(name) or os.path.join(SLURM_BIN, name)
    if not os.path.isfile(program):
        raise ValueError("%s was not found on the server, so the job could not be handled here "
                         "(looked on the PATH and in %s)" % (name, SLURM_BIN))
    environment = dict(os.environ if env is None else env)
    if not environment.get("SLURM_CONF") and os.path.isfile(SLURM_CONF):
        environment["SLURM_CONF"] = SLURM_CONF     # without it slurm cannot find the cluster
    done = subprocess.run([program] + list(command[1:]), cwd=PROJECT, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    if done.returncode != 0:
        raise ValueError("%s failed: %s" % (name, done.stdout.strip()))
    return done.stdout


def round_submit(job, kind, throttle):
    settings = KINDS[kind]
    path = owned_job(job)
    inputs = os.path.join(path, INPUTS)
    movies = job_movies(inputs, kind)
    if not movies:
        raise ValueError("job %s holds no movie a %s round can work on" % (job, kind))
    record = {}
    record_path = os.path.join(path, JOB_RECORD)
    if os.path.isfile(record_path):
        with open(record_path) as f:
            record = json.load(f)
    if record.get("job_id"):
        # a resubmit would run the same movies twice, over each other's output
        return {"ok": True, "job_id": record["job_id"], "movies": len(movies),
                "already_submitted": True}
    manifest = os.path.join(path, MOVIE_MANIFEST)
    with open(manifest, "w") as f:
        f.write("".join(m + "\n" for m in movies))
    os.chmod(manifest, FILE_MODE)
    array = "1-%d" % len(movies)
    if throttle:
        array += "%%%d" % throttle
    out = slurm("sbatch", "--parsable", "--array", array, "-J", "%s_%s" % (kind, job),
                *settings["sbatch"],
                os.path.join(PROJECT, ARRAY_SCRIPT), manifest,
                os.path.join(PROJECT, settings["script"]), *settings["args"])
    job_id = out.strip().splitlines()[-1].split(";")[0]
    record = {"job_id": job_id, "kind": kind, "movies": len(movies), "array": array,
              "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(record_path, "w") as f:
        json.dump(record, f, indent=1)
    os.chmod(record_path, FILE_MODE)
    return {"ok": True, "job_id": job_id, "movies": len(movies)}


def movie_state(movie, kind="realign"):
    """What the round made of one movie: done, refused, or not finished yet."""
    if kind == "video":
        # the stamp is written last, after the mp4 is in place, so it is the finished mark
        return "rendered" if os.path.isfile(os.path.join(movie, VIDEO_STAMP)) else "pending"
    if os.path.isfile(os.path.join(movie, REALIGN_MARKER)):
        return "realigned"
    if os.path.isfile(os.path.join(movie, BLOCKED)):
        return "blocked"
    return "pending"


# slurm states that mean a task is still to come or under way
SLURM_BUSY = ("PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING",
              "COMPLETING")


def array_states(job_id):
    """How many of the array's tasks are in each slurm state, and whether any is still to finish.

    sacct first: squeue forgets a job minutes after it ends, and a round whose tasks all died
    would otherwise be waited on for ever. Returns (counts, still working) or (None, None) when
    neither command can be run."""
    counts, asked = {}, False
    # sacct can be a moment behind sbatch, and squeue drops a job as it ends, so a job neither
    # of them knows of is one that really has finished
    for command in (("sacct", "-j", str(job_id), "-n", "-X", "-P", "-o", "State"),
                    ("squeue", "-j", str(job_id), "-h", "-o", "%T")):
        try:
            lines = slurm(*command).split("\n")
        except ValueError:
            continue
        asked = True
        for line in lines:
            state = line.strip().split()[0] if line.strip() else ""
            if state:
                counts[state] = counts.get(state, 0) + 1
        if counts:
            break
    if not asked:
        return None, None
    return counts, any(state in SLURM_BUSY for state in counts)


def round_status(job, kind):
    path = job_dir(job)
    if not os.path.isdir(path):
        raise ValueError("no such job: %s" % job)
    inputs = os.path.join(path, INPUTS)
    movies = job_movies(inputs, kind)
    states = {os.path.relpath(m, inputs).replace(os.sep, "/"): movie_state(m, kind)
              for m in movies}
    record = {}
    record_path = os.path.join(path, JOB_RECORD)
    if os.path.isfile(record_path):
        with open(record_path) as f:
            record = json.load(f)
    counts, working = array_states(record["job_id"]) if record.get("job_id") else (None, None)
    return {"ok": True, "job": job, "job_id": record.get("job_id"), "states": states,
            "slurm": counts, "working": working}


def result_entries(movie, kind="realign"):
    """What the round made here and the PC should be given back."""
    if kind == "video":
        made = [name for name in (MP4, REPROJECTED, VIDEO_STAMP)
                if os.path.isfile(os.path.join(movie, name))]
        # the stamp is what says the render finished; without it there is nothing to send
        return made if VIDEO_STAMP in made else []
    marker = os.path.join(movie, REALIGN_MARKER)
    if os.path.isfile(marker):
        with open(marker) as f:
            names = json.load(f).get("replaced") or []
        return [REALIGN_MARKER] + [n for n in names if n not in ("", ".", "..") and os.sep not in n
                                   and "/" not in n]
    if os.path.isfile(os.path.join(movie, BLOCKED)):
        return [BLOCKED.replace(os.sep, "/")]
    return []


def round_fetch(job, kind):
    """A gzipped tar of every finished movie's new files, MANIFEST.json first."""
    path = job_dir(job)
    if not os.path.isdir(path):
        raise ValueError("no such job: %s" % job)
    inputs = os.path.join(path, INPUTS)
    files = {}
    for movie in job_movies(inputs, kind):
        key = os.path.relpath(movie, inputs).replace(os.sep, "/")
        for entry in result_entries(movie, kind):
            source = os.path.join(movie, *entry.split("/"))
            if not inside(movie, source):
                continue
            if os.path.isdir(source):
                for dirpath, dirnames, filenames in os.walk(source):
                    dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                    for name in sorted(filenames):
                        full = os.path.join(dirpath, name)
                        files[key + "/" + os.path.relpath(full, movie).replace(os.sep, "/")] = full
            elif os.path.isfile(source):
                files[key + "/" + entry] = source
    manifest = json.dumps({"files": {rel: sha256(full) for rel, full in sorted(files.items())}},
                          indent=1).encode("utf-8")
    out = tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz")
    info = tarfile.TarInfo(MANIFEST)
    info.size = len(manifest)
    info.mtime = int(time.time())
    out.addfile(info, io.BytesIO(manifest))
    for rel, full in sorted(files.items()):
        out.add(full, arcname=rel, recursive=False)
    out.close()
    sys.stdout.buffer.flush()
    return 0


def round_clean(job):
    path = owned_job(job)
    shutil.rmtree(path)
    return {"ok": True, "removed": path}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("declarations")
    rec = sub.add_parser("receive")
    rec.add_argument("--dest", required=True)
    submit = sub.add_parser("round-submit")
    submit.add_argument("--job", required=True)
    submit.add_argument("--kind", choices=sorted(KINDS), default="realign")
    submit.add_argument("--throttle", type=int, default=0,
                        help="at most this many array tasks at once (0: no limit)")
    for name in ("round-status", "round-fetch", "round-clean"):
        verb = sub.add_parser(name)
        verb.add_argument("--job", required=True)
        verb.add_argument("--kind", choices=sorted(KINDS), default="realign")
    import local_predict_server as predict
    predict.add_verbs(sub)
    args = parser.parse_args()
    if args.command == "declarations":
        return declarations()
    if args.command == "receive":
        return receive(args.dest)
    if args.command == "predict-keep":
        # a long-running job of its own, not a question: its log is the output
        return predict.predict_keep(args.job, args.interval)
    if args.command in ("round-fetch", "predict-fetch"):
        try:
            if args.command == "predict-fetch":
                return predict.predict_fetch(args.job, predict.split(args.movies),
                                             predict.split(args.units))
            return round_fetch(args.job, args.kind)
        except Exception as e:
            # stdout is the tar itself, so the reason goes to stderr and the PC sees an empty
            # stream rather than a tar with a traceback in it
            sys.stderr.write("%s: %s\n" % (type(e).__name__, e))
            return 1
    handlers = {"round-submit": lambda: round_submit(args.job, args.kind, args.throttle),
                "round-status": lambda: round_status(args.job, args.kind),
                "round-clean": lambda: round_clean(args.job)}
    handlers.update(predict.handlers(args))
    if args.command in handlers:
        try:
            print(json.dumps(handlers[args.command]()))
            return 0
        except Exception as e:
            print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}))
            return 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
