"""The cluster side of a PC's predict round (code/local_predict.py). Standard library only.

local_reanalysis_server.py hands its predict-* verbs to this module; see that file for how a PC
reaches it (ssh to the gateway, srun to a node, the system python3). A predict round is one job
folder, realign_jobs/predict_<PC>_<stamp>/, which the PC fills with `receive --dest <job>`:

    units/<U>.json                     what to prep: the unit's input folder, easyWand, mirror
                                       camera, run name, extra prep arguments, its movies
    inference_datasets/<source>/<experiment>[/<batch>]/      the unit's input folder, laid out
        <easyWand>.mat, perturbation.json,                   as prep wants it, under a path that
        mov<N>/*_sparse.mat (blanked) + trim.json            keeps provenance (source.json) right

and to which this side adds, as each unit runs:

    units/<U>.state.json       the prep job's id, and every predict array submitted for the unit
    predict_config.json        the deployed predict config, writing into <job>/predict_output
    manifests/<U>.txt          prep's verify-passed movies = the array's tasks (line k = task k-1)
    status/<U>.json            prep's per-movie outcome (process_experiment.py --status-json)
    array_ids/<U>.txt          the predict array pipeline.sh submitted
    render_boxes/<U>/          <stem>_render.h5 per predicted movie (predict_array.sh RENDER_BOX_DIR)
    predict_output/<run>/<stem>/   the predictions
    failed_attempts/           a failed task's partial output, moved aside before it is retried

A unit's prep runs as `pipeline.sh` under the job name pose_prep with --dependency=singleton, so
the preps of every round this account starts run one at a time (MATLAB builds stall each other);
pipeline.sh then submits the unit's predict array itself, as it always has.

Verbs (one JSON line each, except predict-fetch, whose stdout is a tar):

    predict-submit  --job J --unit U [--throttle N] [--again]
                                                       start one unit's prep + predict (--again:
                                                       once more, after a prep that never ran)
    predict-status  --job J                            every movie's state, and slurm's
    predict-retry   --job J --movies U/movN,...        resubmit failed predict tasks
    predict-reset   --job J --unit U                   forget a unit whose prep crashed
    predict-fetch   --job J [--movies ...] [--units ...]   gzip tar of results, MANIFEST.json first
    predict-release --job J --movies ...               delete movies the PC has installed
    predict-space   --job J                            free space here, and the job's size
    predict-clean   --job J                            cancel what still runs, delete the job
    predict-keeper  --job J [--no-mail]                start the round's keeper (below)
    predict-keep    --job J                            BE the keeper: run by the keeper's job

The keeper is what lets the PC be switched off once a round is uploaded. It is a small CPU job
that does, every few minutes, what the PC's own watching does on the cluster's side: a GPU task
that failed is retried (once), a prep that never started -- a node without the lab's disk -- is
submitted again. It ends when nothing more can happen on the cluster and then has slurm email the
owner, through a mail-only job named predictions_<run>_<n>_of_<total>_ready (or ..._needs_the_PC, when a prep
crashed or the round is stuck, which only the PC can mend). While a keeper runs the PC leaves
those actions to it and only brings movies home; a lock and idempotent verbs keep the two from
ever acting twice on one failure.

A keeper asks for up to KEEPER_DAYS but lets slurm shorten that to fit before a maintenance
reservation (--time-min), and shortly before its time runs out it hands over to a successor --
which, during maintenance, simply waits for the cluster to come back. Only the keeper that sees
the round end sends the email.
"""
import gzip
import io
import json
import os
import re
import shlex
import shutil
import sys
import tarfile
import time

import local_reanalysis_server as base

PROJECT = base.PROJECT
CONFIG_DIR = os.path.join(PROJECT, "predict_configurations")
DEFAULT_CONFIG = "config1.json"
PIPELINE = os.path.join(PROJECT, "sbatch_files", "pipeline.sh")
PREDICT_ARRAY = os.path.join(PROJECT, "sbatch_files", "predict_array.sh")
# one name for every PC-launched prep, so --dependency=singleton runs them one at a time
PREP_JOB_NAME = "pose_prep"
# prep peaks under 2 GB; the time grows with the movies (raw movie ~6.5 min, build ~3 min each)
PREP_SBATCH = ["--partition=glacier", "--mem=16g", "--cpus-per-task=4"]
PREP_MINUTES_BASE, PREP_MINUTES_PER_MOVIE = 60, 12
# predict_array.sh's own default (256g, 32 cpus, an L40S on salmon) sits behind salmon's queue;
# this is what schedules (see PIPELINE.md: ~8.4 MB a frame + 3 GB, so 96g covers any movie)
PREDICT_SBATCH = ["-p", "catfish,salmon", "--gres=gpu:1", "--mem=96g", "--cpus-per-task=12"]
DEFAULT_THROTTLE = 32

UNIT = re.compile(r"^u\d{1,4}$")
MOVIE_DIR = re.compile(r"^mov\d+$")
CAM = re.compile(r"^(cam\d+|none)$")
PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
VALUE = re.compile(r"^[A-Za-z0-9.+-]{1,32}$")
# the prep options a PC may pass, and how many values each takes
PREP_FLAGS = {"--num-cams": 1, "--bottom-cam": 1, "--skip-raw-movies": 0,
              "--prescan-min-intersection": 1, "--prescan-pixel-threshold": 1,
              "--prescan-blob-ratio": 1, "--prescan-blob-distance": 1,
              "--prescan-min-edge-margin": 1, "--prescan-min-cams-in-frame": 1,
              "--verify-threshold": 1, "--verify-threshold-2cam": 1}
TRIM_FILE = "trim.json"
ANALYSIS_SUFFIX = "_analysis_smoothed.h5"
RENDER_SUFFIX = "_render.h5"
# what comes home beside a movie's mats, and beside a unit's movies
MOVIE_PREP_FILES = ("prescan_cam_validity.npz", "raw_movie.log")
RAW_MOVIE = re.compile(r"_raw_fr\d+_skip\d+\.mp4$")
UNIT_FILES = ("calibration.h5", "process_report.txt", "pipeline_timings.csv",
              "build_calibration.log")
FAILED = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL",
          "DEADLINE", "PREEMPTED")
# a task that dies this fast with no log never ran: the automount was gone on its node
INSTANT_SECONDS = 10
# how often a failed GPU task is tried again, and a prep that never started submitted again
MAX_RETRIES = 1
MAX_RESUBMITS = 3
# the keeper: a CPU job that looks after a round while the PC is off
KEEPER_FILE = "keeper.json"
CLEARING = ".clearing"
KEEP_INTERVAL = 300
# submitted with --wrap rather than through sbatch_configurable.sh, whose #SBATCH mail lines win
# over a command-line --mail-type=NONE: the keeper itself must never email -- only notify() does
KEEPER_SBATCH = ["--partition=glacier", "--gres=gpu:0", "--mem=1g", "--cpus-per-task=1"]
KEEPER_MAX_MINUTES = 6 * 24 * 60
# hand over to a successor this long before the keeper's own time runs out
HANDOVER_SECONDS = 900
LOCK = ".lock"
# touched whenever a PC asks how the round is going: a PC that is watching mends a crashed prep
# itself, so the keeper does not email it that the round needs it
PC_SEEN = ".pc_seen"
PC_PRESENT_SECONDS = 900
SETTLED = ("predicted", "rejected", "prep_failed", "prep_stopped", "released", "missing")


class job_lock:
    """One actor at a time on a round's jobs: the PC and the round's keeper may both decide to act
    on the same failure. mkdir is atomic on NFS, where file locks are not to be trusted."""

    def __init__(self, path, wait=120, stale=900):
        self.lock, self.wait, self.stale = os.path.join(path, LOCK), wait, stale

    def __enter__(self):
        start = time.time()
        while True:
            try:
                os.mkdir(self.lock)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.stat(self.lock).st_mtime > self.stale:
                        os.rmdir(self.lock)      # left by an action that died holding it
                        continue
                except OSError:
                    continue
                if time.time() - start > self.wait:
                    raise ValueError("the round is busy with another action; try again")
                time.sleep(2)

    def __exit__(self, *exc):
        try:
            os.rmdir(self.lock)
        except OSError:
            pass


# -- the job folder -------------------------------------------------------------------------------

def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, value):
    base.makedirs(os.path.dirname(path))
    staged = path + ".partial"
    with open(staged, "w") as f:
        json.dump(value, f, indent=1)
    os.chmod(staged, base.FILE_MODE)
    os.replace(staged, path)


def within(job_path, rel):
    """A job-relative path the PC named, refusing anything that leads out of the job."""
    parts = str(rel).split("/")
    if not parts or not all(PART.match(p) for p in parts):
        raise ValueError("not a plain relative path: %r" % rel)
    full = os.path.join(job_path, *parts)
    if not base.inside(job_path, full):
        raise ValueError("path leaves the job: %r" % rel)
    return full


def unit_names(job_path):
    folder = os.path.join(job_path, "units")
    if not os.path.isdir(folder):
        return []
    return sorted((n[:-5] for n in os.listdir(folder)
                   if n.endswith(".json") and UNIT.match(n[:-5])),
                  key=lambda u: int(u[1:]))


def load_unit(job_path, unit):
    """The unit's spec, checked: every path inside the job, every value of a known shape."""
    if not UNIT.match(unit or ""):
        raise ValueError("not a unit name: %r" % unit)
    spec = load_json(os.path.join(job_path, "units", unit + ".json"))
    if not isinstance(spec, dict):
        raise ValueError("unit %s has not been uploaded" % unit)
    parts = str(spec.get("input", "")).split("/")
    if len(parts) < 2 or parts[0] != "inference_datasets":
        raise ValueError("unit %s: its input must lie under inference_datasets/" % unit)
    spec["input_dir"] = within(job_path, spec["input"])
    spec["easywand_path"] = within(job_path, spec.get("easywand", ""))
    if not spec["easywand_path"].endswith(".mat"):
        raise ValueError("unit %s: the easyWand must be a .mat" % unit)
    if not CAM.match(str(spec.get("cam", ""))):
        raise ValueError("unit %s: the mirror camera must be camN or none" % unit)
    if not base.JOB_NAME.match(str(spec.get("run_name", ""))):
        raise ValueError("unit %s: a run name may hold letters, digits, . - _ only" % unit)
    args = list(spec.get("prep_args") or [])
    i = 0
    while i < len(args):
        if args[i] not in PREP_FLAGS:
            raise ValueError("unit %s: prep option %r is not allowed from a PC" % (unit, args[i]))
        values = args[i + 1:i + 1 + PREP_FLAGS[args[i]]]
        if len(values) != PREP_FLAGS[args[i]] or not all(VALUE.match(str(v)) for v in values):
            raise ValueError("unit %s: bad value for %s" % (unit, args[i]))
        i += 1 + PREP_FLAGS[args[i]]
    spec["prep_args"] = [str(a) for a in args]
    movies = spec.get("movies")
    if not isinstance(movies, dict) or not movies:
        raise ValueError("unit %s lists no movies" % unit)
    for name in movies:
        if not MOVIE_DIR.match(name):
            raise ValueError("unit %s: %r is not a movie folder name" % (unit, name))
    config = str(spec.get("predict_config") or DEFAULT_CONFIG)
    if not PART.match(config) or not os.path.isfile(os.path.join(CONFIG_DIR, config)):
        raise ValueError("unit %s: no predict config %r on the server" % (unit, config))
    spec["predict_config"] = config
    return spec


def unit_paths(job_path, unit):
    return {
        "state": os.path.join(job_path, "units", unit + ".state.json"),
        "manifest": os.path.join(job_path, "manifests", unit + ".txt"),
        "status": os.path.join(job_path, "status", unit + ".json"),
        "array_id": os.path.join(job_path, "array_ids", unit + ".txt"),
        "render": os.path.join(job_path, "render_boxes", unit),
        "output": os.path.join(job_path, "predict_output"),
    }


def slurm_environment():
    """This environment, with slurm reachable: the PATH a PC's ssh session arrives with on the
    gateway lacks slurm's commands, and a job inherits it -- pipeline.sh, which submits the predict
    array itself, would find no sbatch."""
    env = dict(os.environ)
    if base.SLURM_BIN not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = os.pathsep.join(p for p in (env.get("PATH"), base.SLURM_BIN) if p)
    if not env.get("SLURM_CONF") and os.path.isfile(base.SLURM_CONF):
        env["SLURM_CONF"] = base.SLURM_CONF
    return env


# The predict array's time limit grows with the unit's longest movie. Measured on catfish's L4s
# with 12 CPUs: 754 frames took 1.1 h, 3879 frames 6.3 h, and 5891 frames ran out of 8 h (its
# eight members took ~4 h, then the ensemble step was still running). A movie whose task timed
# out anyway is retried with twice the time.
PREDICT_MINUTES_BASE, PREDICT_MINUTES_PER_1000_FRAMES = 60, 120
PREDICT_MAX_MINUTES = 6 * 24 * 60


def predict_minutes(spec, longer=False):
    frames = max([int((m or {}).get("frames") or 0) for m in spec["movies"].values()] or [0])
    minutes = PREDICT_MINUTES_BASE + PREDICT_MINUTES_PER_1000_FRAMES * frames / 1000.0
    return int(min(PREDICT_MAX_MINUTES, minutes * (2 if longer else 1)))


def task_entries(attempts, index, slurm):
    """(attempt, task id, slurm's entry) for every array that ran this manifest line, oldest
    first."""
    found = []
    for attempt in attempts:
        if attempt["indices"] is None or index in attempt["indices"]:
            task_id = "%s_%d" % (attempt["array_id"], index)
            found.append((attempt, task_id, (slurm or {}).get(task_id)))
    return found


def died_instantly(run_name, task_id, entry):
    """A task that ended within seconds without a log never ran: its node did not have the lab's
    disk mounted."""
    return (bool(entry) and not busy(entry) and entry.get("state") != "COMPLETED"
            and (entry.get("elapsed") or 0) <= INSTANT_SECONDS
            and not os.path.isfile(os.path.join(PROJECT, "logs",
                                                "%s_%s.out" % (run_name, task_id))))


def retry_budget(attempts, index, slurm, run_name):
    """(retries that ran, retries that never started, timed out before) for one movie. Only the
    first use up MAX_RETRIES; a retry lost to a bad node counts against MAX_RESUBMITS instead."""
    ran = lost = 0
    timed_out = False
    for attempt, task_id, entry in task_entries(attempts, index, slurm):
        if entry and entry.get("state") == "TIMEOUT":
            timed_out = True
        if attempt["indices"] is None or attempt.get("initial"):
            continue                       # the movie's first array is not a retry
        if died_instantly(run_name, task_id, entry):
            lost += 1
        elif entry is not None and not busy(entry):
            ran += 1
    return ran, lost, timed_out


def may_retry(run_name, task_id, entry, ran, lost, max_retries=MAX_RETRIES):
    if died_instantly(run_name, task_id, entry):
        return lost < MAX_RESUBMITS
    return ran < max_retries


def predict_environment(job_path, unit, spec):
    """What a unit's prep and predict jobs need in their environment. sbatch passes it on, and
    pipeline.sh hands it to the predict array it submits."""
    paths = unit_paths(job_path, unit)
    return dict(slurm_environment(),
                POSE_PROJECT=PROJECT,
                PIPELINE_RUN_NAME=spec["run_name"],
                PIPELINE_MANIFEST=paths["manifest"],
                PIPELINE_ARRAY_ID_FILE=paths["array_id"],
                PREDICT_SBATCH_ARGS=" ".join(PREDICT_SBATCH
                                             + ["--time=%d" % predict_minutes(spec)]),
                DROP_BOX_CACHE="1",
                RENDER_BOX_DIR=paths["render"])


def job_config(job_path, spec):
    """The job's predict config: the deployed one, writing into the job's own predict_output."""
    path = os.path.join(job_path, "predict_config.json")
    config = load_json(os.path.join(CONFIG_DIR, spec["predict_config"]))
    config["output directory"] = os.path.join(job_path, "predict_output")
    save_json(path, config)
    return path


# -- slurm ----------------------------------------------------------------------------------------

def _seconds(text):
    """slurm's [D-]HH:MM:SS / MM:SS / a plain number, in seconds."""
    text = (text or "").strip()
    if text.isdigit():
        return int(text)
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d) if d.isdigit() else 0
    try:
        parts = [int(p) for p in text.split(":")]
    except ValueError:
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return days * 86400 + seconds


def _expand(job_field):
    """'123' -> ['123']; '123_4' -> ['123_4']; '123_[3-5,9%2]' -> ['123_3', '123_4', ...]."""
    m = re.match(r"^(\d+)_\[([^\]]+)\]$", job_field)
    if not m:
        return [job_field]
    out = []
    for piece in m.group(2).split("%")[0].split(","):
        if "-" in piece:
            a, b = piece.split("-", 1)
            out += ["%s_%d" % (m.group(1), i) for i in range(int(a), int(b) + 1)]
        elif piece.isdigit():
            out.append("%s_%s" % (m.group(1), piece))
    return out


def slurm_states(job_ids):
    """{'<id>' or '<id>_<task>': {'state', 'elapsed', 'reason'}} for these jobs, from squeue for
    what is still queued or running and sacct for what has ended. None when slurm cannot be
    asked at all."""
    ids = sorted({str(j) for j in job_ids if j})
    if not ids:
        return {}
    found, asked = {}, False
    try:
        out = base.slurm("sacct", "-j", ",".join(ids), "-n", "-X", "-P",
                         "-o", "JobID,State,ElapsedRaw")
        asked = True
        for line in out.splitlines():
            fields = line.strip().split("|")
            if len(fields) < 3 or not fields[0]:
                continue
            for key in _expand(fields[0]):
                found[key] = {"state": fields[1].split()[0] if fields[1] else "",
                              "elapsed": _seconds(fields[2]), "reason": ""}
    except ValueError:
        pass
    try:
        out = base.slurm("squeue", "-h", "-r", "-j", ",".join(ids), "-o", "%i|%T|%r|%M")
        asked = True
        for line in out.splitlines():
            fields = line.strip().split("|")
            if len(fields) < 4 or not fields[0]:
                continue
            for key in _expand(fields[0]):
                found[key] = {"state": fields[1], "elapsed": _seconds(fields[3]),
                              "reason": fields[2]}
    except ValueError:
        # squeue refuses ids it has already forgotten; sacct has them
        pass
    return found if asked else None


def busy(entry):
    return bool(entry) and entry.get("state") in base.SLURM_BUSY


# -- verbs ----------------------------------------------------------------------------------------

def prep_log(prep_id):
    return os.path.join(PROJECT, "logs", "%s_%s.out" % (PREP_JOB_NAME, prep_id))


def prep_never_ran(paths, prep_id, entry):
    """True when a unit's prep job died before doing anything: within seconds, with no log and
    no status -- a node without the lab's filesystem. Its mats are untouched, so it may simply be
    submitted again."""
    return (bool(entry) and not busy(entry)
            and (entry.get("elapsed") or 0) <= INSTANT_SECONDS
            and not os.path.isfile(prep_log(prep_id))
            and not os.path.isfile(paths["status"]))


def predict_submit(job, unit, throttle, again=False):
    path = base.owned_job(job)
    with job_lock(path):
        return _submit(path, unit, throttle, again)


def _submit(path, unit, throttle, again):
    spec = load_unit(path, unit)
    paths = unit_paths(path, unit)
    state = load_json(paths["state"], {}) or {}
    if state.get("prep_job_id"):
        prep_id = state["prep_job_id"]
        if not again:
            # a second prep would flip the mirror camera back
            return {"ok": True, "prep_job_id": prep_id, "already_submitted": True}
        if not prep_never_ran(paths, prep_id, (slurm_states([prep_id]) or {}).get(prep_id)):
            raise ValueError("unit %s's prep ran (or is running); submitting it again would "
                             "prep the same mats twice" % unit)
    missing = [m for m in spec["movies"]
               if not os.path.isfile(os.path.join(spec["input_dir"], m, TRIM_FILE))]
    if missing:
        raise ValueError("unit %s is not fully uploaded: %s" % (unit, ", ".join(missing[:5])))
    if not os.path.isfile(spec["easywand_path"]):
        raise ValueError("unit %s: its easyWand was not uploaded" % unit)
    config = job_config(path, spec)
    for folder in (os.path.dirname(paths["manifest"]), os.path.dirname(paths["status"]),
                   os.path.dirname(paths["array_id"]), paths["render"]):
        base.makedirs(folder)
    minutes = PREP_MINUTES_BASE + PREP_MINUTES_PER_MOVIE * len(spec["movies"])
    out = base.slurm(
        "sbatch", "--parsable", "-J", PREP_JOB_NAME, "--dependency=singleton",
        *PREP_SBATCH, "--time=%d" % minutes,
        PIPELINE, spec["input_dir"], spec["easywand_path"], spec["cam"], config,
        str(throttle or DEFAULT_THROTTLE),
        "--status-json", paths["status"], *spec["prep_args"],
        env=predict_environment(path, unit, spec))
    job_id = out.strip().splitlines()[-1].split(";")[0]
    state = {"prep_job_id": job_id, "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "throttle": throttle or DEFAULT_THROTTLE, "attempts": [],
             "lost_preps": state.get("lost_preps", []) + ([state["prep_job_id"]]
                                                          if state.get("prep_job_id") else [])}
    save_json(paths["state"], state)
    return {"ok": True, "prep_job_id": job_id, "movies": len(spec["movies"])}


def manifest_lines(path):
    try:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []


def movie_stem(status_entry):
    box = (status_entry or {}).get("box") or ""
    return box[:-3] if box.endswith(".h5") else ""


def tree_bytes(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def unit_report(path, unit, slurm):
    """One unit's prep, and the state of each of its movies."""
    spec = load_unit(path, unit)
    paths = unit_paths(path, unit)
    state = load_json(paths["state"], {}) or {}
    status = load_json(paths["status"], {}) or {}
    lines = manifest_lines(paths["manifest"])
    prep_id = state.get("prep_job_id")
    prep = (slurm or {}).get(str(prep_id)) if prep_id else None
    submitted = state.get("submitted_at")
    age = (time.time() - time.mktime(time.strptime(submitted, "%Y-%m-%d %H:%M:%S"))
           if submitted else 0)
    if not prep_id:
        prep_phase = "not submitted"
    elif busy(prep):
        prep_phase = ("waiting" if prep["state"] == "PENDING" and "Dependency" in prep["reason"]
                      else "queued" if prep["state"] == "PENDING" else "prepping")
    elif prep is None and (slurm is None or age < 600):
        prep_phase = "queued"      # slurm has not caught up with a fresh submission yet
    else:
        prep_phase = "done"
    array_ids = []
    first = manifest_lines(paths["array_id"])
    if first:
        array_ids.append({"array_id": first[0], "indices": None})
    array_ids += state.get("attempts") or []
    movies = {}
    for name in sorted(spec["movies"], key=lambda m: int(m[3:])):
        key = "%s/%s" % (unit, name)
        movie_dir = os.path.join(spec["input_dir"], name)
        entry = (status.get("movies") or {}).get(name) or {}
        stem = movie_stem(entry) or spec["movies"][name].get("stem", "")
        row = {"state": "", "reason": "", "stem": stem}
        movies[key] = row
        if not os.path.isdir(movie_dir):
            row["state"] = "released" if state.get("released", {}).get(name) else "missing"
            continue
        if prep_phase != "done":
            row["state"] = prep_phase if prep_phase != "not submitted" else "uploaded"
            continue
        if movie_dir in lines:
            index = lines.index(movie_dir)
            ran, lost, timed_out = retry_budget(array_ids, index, slurm, spec["run_name"])
            row["retries"] = ran
            task_id = None
            for attempt in array_ids:
                if attempt["indices"] is None or index in attempt["indices"]:
                    task_id = "%s_%d" % (attempt["array_id"], index)
            task = (slurm or {}).get(task_id) if task_id else None
            output = os.path.join(paths["output"], spec["run_name"], stem)
            analysed = os.path.isfile(os.path.join(output, stem + ANALYSIS_SUFFIX))
            if task_id is None:
                row.update(state="failed", no_array=True, retryable=True,
                           reason="prep passed it, but no predict array was submitted")
            elif busy(task):
                row["state"] = "predicting" if task["state"] == "RUNNING" else "queued"
            elif task is not None and task["state"] == "COMPLETED" and analysed:
                row["state"] = "predicted"
                row["bytes"] = (tree_bytes(output) + tree_bytes(movie_dir)
                                + tree_bytes(os.path.join(paths["render"],
                                                          stem + RENDER_SUFFIX)))
            elif task is None and slurm is not None:
                row["state"] = "queued"        # not in slurm's books yet
            elif task is None:
                row["state"] = "predicted" if analysed else "queued"
            else:
                log = os.path.join(PROJECT, "logs", "%s_%s.out" % (spec["run_name"], task_id))
                instant = ((task.get("elapsed") or 0) <= INSTANT_SECONDS
                           and not os.path.isfile(log))
                row.update(state="failed", task=task_id, instant=instant,
                           timed_out=timed_out,
                           retryable=may_retry(spec["run_name"], task_id, task, ran, lost),
                           reason="predict task %s %s%s" % (
                               task_id, task["state"],
                               " within %ss with no log (an unmounted node)" % task.get("elapsed")
                               if instant else "; log: logs/%s_%s.out" % (spec["run_name"],
                                                                         task_id)))
            continue
        # not in the manifest: prep turned it away, or prep never got that far
        prescan, verify = entry.get("prescan"), entry.get("verify")
        if prescan and prescan != "OK":
            row.update(state="rejected", reason=entry.get("reason") or "prescan " + prescan)
        elif verify and verify != "PASS":
            medians = entry.get("medians")
            row.update(state="prep_failed", reason="verify %s%s" % (
                verify, " %s" % medians if medians else ""))
        elif status.get("stopped"):
            row.update(state="prep_stopped", reason=status["stopped"])
        elif not status and prep_never_ran(paths, prep_id, prep):
            row.update(state="prep_crashed", instant=True,
                       reason="prep job %s died within %ss with no log (an unmounted node)"
                              % (prep_id, (prep or {}).get("elapsed")))
        elif not status:
            row.update(state="prep_crashed",
                       reason="prep ended (%s) without saying how each movie went; see "
                              "logs/%s_%s.out" % ((prep or {}).get("state", "?"),
                                                  PREP_JOB_NAME, prep_id))
        else:
            row.update(state="prep_failed", reason="prep did not pass it on to prediction")
        row["bytes"] = tree_bytes(movie_dir)
    return {"prep": prep_phase, "prep_job_id": prep_id,
            "prep_resubmits": len(state.get("lost_preps") or []),
            "prep_slurm": (prep or {}).get("state"), "run_name": spec["run_name"],
            "stopped": status.get("stopped"), "mirror": status.get("mirror")}, movies


def note_pc(job):
    """Remember that a PC just looked at the round (predict-status asked over ssh)."""
    path = base.job_dir(job)
    if os.path.isdir(path):
        try:
            with open(os.path.join(path, PC_SEEN), "w") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        except OSError:
            pass


def pc_present(path):
    try:
        return time.time() - os.path.getmtime(os.path.join(path, PC_SEEN)) < PC_PRESENT_SECONDS
    except OSError:
        return False


def predict_status(job):
    path = base.job_dir(job)
    if not os.path.isdir(path):
        raise ValueError("no such job: %s" % job)
    ids = []
    for unit in unit_names(path):
        paths = unit_paths(path, unit)
        state = load_json(paths["state"], {}) or {}
        ids.append(state.get("prep_job_id"))
        ids += manifest_lines(paths["array_id"])[:1]
        ids += [a["array_id"] for a in state.get("attempts") or []]
    slurm = slurm_states(ids)
    units, movies = {}, {}
    for unit in unit_names(path):
        try:
            units[unit], rows = unit_report(path, unit, slurm)
        except ValueError as e:
            units[unit] = {"prep": "invalid", "error": str(e)}
            continue
        movies.update(rows)
    working = None if slurm is None else any(busy(v) for v in slurm.values())
    return {"ok": True, "job": job, "units": units, "movies": movies, "working": working,
            "keeper": keeper_state(path)}


def predict_retry(job, keys, max_retries=MAX_RETRIES):
    """Submit failed predict tasks again. Idempotent: a movie whose latest task is queued,
    running or done, or that has had its retries, is left alone and reported as skipped -- so
    the PC and the keeper may both ask without a movie ever running twice."""
    path = base.owned_job(job)
    with job_lock(path):
        return _retry(path, keys, max_retries)


def _retry(path, keys, max_retries):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    by_unit = {}
    for key in keys:
        unit, _, name = key.partition("/")
        by_unit.setdefault(unit, []).append(name)
    submitted, skipped = {}, {}
    for unit, names in sorted(by_unit.items()):
        spec = load_unit(path, unit)
        paths = unit_paths(path, unit)
        lines = manifest_lines(paths["manifest"])
        status = load_json(paths["status"], {}) or {}
        state = load_json(paths["state"], {}) or {}
        attempts = ([{"array_id": i, "indices": None} for i in manifest_lines(paths["array_id"])[:1]]
                    + (state.get("attempts") or []))
        slurm = slurm_states([a["array_id"] for a in attempts]) or {}
        indices, initial, longer = [], True, False
        for name in names:
            key = "%s/%s" % (unit, name)
            movie_dir = os.path.join(spec["input_dir"], name)
            if movie_dir not in lines:
                raise ValueError("%s was never passed on to prediction" % key)
            index = lines.index(movie_dir)
            ran, lost, timed_out = retry_budget(attempts, index, slurm, spec["run_name"])
            latest = None
            for attempt in attempts:
                if attempt["indices"] is None or index in attempt["indices"]:
                    latest = "%s_%d" % (attempt["array_id"], index)
            task = slurm.get(latest) if latest else None
            stem = movie_stem((status.get("movies") or {}).get(name))
            partial = os.path.join(paths["output"], spec["run_name"], stem)
            analysed = bool(stem) and os.path.isfile(os.path.join(partial,
                                                                  stem + ANALYSIS_SUFFIX))
            if latest is not None and task is not None and not busy(task) and not \
                    may_retry(spec["run_name"], latest, task, ran, lost, max_retries):
                skipped[key] = "retried already (%d ran, %d lost to a bad node)" % (ran, lost)
                continue
            # no task at all is fine when no array was ever submitted for it (pipeline.sh could
            # not); an array that slurm does not know yet, or a task still going, is not failed
            if (latest is not None and task is None) or busy(task) or \
                    (task is not None and task["state"] == "COMPLETED" and analysed):
                skipped[key] = "its task is not a failed one"
                continue
            indices.append(index)
            initial = initial and latest is None
            longer = longer or timed_out
            # a re-run into a folder that holds a failed attempt's members would pool them
            if stem and os.path.isdir(partial):
                aside = os.path.join(path, "failed_attempts", "%s_%s" % (stem, stamp))
                base.makedirs(os.path.dirname(aside))
                os.replace(partial, aside)
        if not indices:
            continue
        out = base.slurm(
            "sbatch", "--parsable", "-J", spec["run_name"],
            "--array=%s%%%d" % (",".join(str(i) for i in sorted(indices)),
                                state.get("throttle") or DEFAULT_THROTTLE),
            *PREDICT_SBATCH, "--time=%d" % predict_minutes(spec, longer),
            PREDICT_ARRAY, paths["manifest"],
            os.path.join(path, "predict_config.json"),
            os.path.join(spec["input_dir"], "pipeline_timings.csv"),
            env=predict_environment(path, unit, spec))
        array_id = out.strip().splitlines()[-1].split(";")[0]
        state.setdefault("attempts", []).append(
            {"array_id": array_id, "indices": sorted(indices), "initial": initial,
             "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        save_json(paths["state"], state)
        submitted[unit] = array_id
    return {"ok": True, "submitted": submitted, "skipped": skipped}


def predict_reset(job, unit):
    """Forget a unit whose prep died part-way, so the PC can upload it afresh. Its mats may be
    half flipped, so they are never prepped again in place."""
    path = base.owned_job(job)
    with job_lock(path):
        return _reset(path, unit)


def _reset(path, unit):
    spec = load_unit(path, unit)
    paths = unit_paths(path, unit)
    state = load_json(paths["state"], {}) or {}
    slurm = slurm_states([state.get("prep_job_id")] + manifest_lines(paths["array_id"])[:1])
    if any(busy(v) for v in (slurm or {}).values()):
        raise ValueError("unit %s is still running on the cluster" % unit)
    for target in (spec["input_dir"], paths["render"]):
        shutil.rmtree(target, ignore_errors=True)
    for target in (paths["state"], paths["manifest"], paths["status"], paths["array_id"],
                   os.path.join(path, "units", unit + ".json")):
        if os.path.isfile(target):
            os.remove(target)
    return {"ok": True, "reset": unit}


def movie_files(path, key):
    """{name in the tar: file here} for one movie: its prep products and its predictions."""
    unit, _, name = key.partition("/")
    spec = load_unit(path, unit)
    if not MOVIE_DIR.match(name) or name not in spec["movies"]:
        raise ValueError("%s is not a movie of this round" % key)
    paths = unit_paths(path, unit)
    status = load_json(paths["status"], {}) or {}
    movie_dir = os.path.join(spec["input_dir"], name)
    files = {}
    if os.path.isdir(movie_dir):
        for entry in sorted(os.listdir(movie_dir)):
            full = os.path.join(movie_dir, entry)
            if os.path.isfile(full) and (entry in MOVIE_PREP_FILES or RAW_MOVIE.search(entry)
                                         or (entry.startswith("build_") and
                                             entry.endswith(".log"))):
                files["%s/prep/%s" % (key, entry)] = full
    stem = movie_stem((status.get("movies") or {}).get(name))
    if stem:
        render = os.path.join(paths["render"], stem + RENDER_SUFFIX)
        if os.path.isfile(render):
            files["%s/prep/%s" % (key, stem + RENDER_SUFFIX)] = render
        output = os.path.join(paths["output"], spec["run_name"], stem)
        for dirpath, dirnames, filenames in os.walk(output):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for entry in sorted(filenames):
                full = os.path.join(dirpath, entry)
                rel = os.path.relpath(full, output).replace(os.sep, "/")
                files["%s/predicted/%s" % (key, rel)] = full
    return files


def unit_files(path, unit):
    spec = load_unit(path, unit)
    paths = unit_paths(path, unit)
    files = {}
    for name in UNIT_FILES:
        full = os.path.join(spec["input_dir"], name)
        if os.path.isfile(full):
            files["%s/unit/%s" % (unit, name)] = full
    if os.path.isfile(paths["status"]):
        files["%s/unit/prep_status.json" % unit] = paths["status"]
    return files


def predict_fetch(job, keys, units):
    """A tar of the asked-for movies' and units' files, gzipped lightly (most of it is mp4 and
    compressed h5; the scores json is what shrinks), with MANIFEST.json and its sha256 first."""
    path = base.job_dir(job)
    if not os.path.isdir(path):
        raise ValueError("no such job: %s" % job)
    files = {}
    for key in keys:
        files.update(movie_files(path, key))
    for unit in units:
        files.update(unit_files(path, unit))
    manifest = json.dumps({"files": {rel: base.sha256(full)
                                     for rel, full in sorted(files.items())}},
                          indent=1).encode("utf-8")
    stream = gzip.GzipFile(fileobj=sys.stdout.buffer, mode="wb", compresslevel=1)
    out = tarfile.open(fileobj=stream, mode="w|")
    info = tarfile.TarInfo(base.MANIFEST)
    info.size = len(manifest)
    info.mtime = int(time.time())
    out.addfile(info, io.BytesIO(manifest))
    for rel, full in sorted(files.items()):
        out.add(full, arcname=rel, recursive=False)
    out.close()
    stream.close()
    sys.stdout.buffer.flush()
    return 0


def predict_release(job, keys):
    """Delete what the PC has safely installed, so a long round does not hold the lab's disk."""
    path = base.owned_job(job)
    freed = 0
    for key in keys:
        unit, _, name = key.partition("/")
        spec = load_unit(path, unit)
        if name not in spec["movies"]:
            raise ValueError("%s is not a movie of this round" % key)
        paths = unit_paths(path, unit)
        status = load_json(paths["status"], {}) or {}
        stem = movie_stem((status.get("movies") or {}).get(name))
        targets = [os.path.join(spec["input_dir"], name)]
        if stem:
            targets += [os.path.join(paths["output"], spec["run_name"], stem),
                        os.path.join(paths["render"], stem + RENDER_SUFFIX)]
        for target in targets:
            if not base.inside(path, target) or not os.path.exists(target):
                continue
            freed += tree_bytes(target)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            else:
                os.remove(target)
        state = load_json(paths["state"], {}) or {}
        state.setdefault("released", {})[name] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_json(paths["state"], state)
    return {"ok": True, "released": len(keys), "freed_bytes": freed}


def predict_space(job):
    stats = os.statvfs(PROJECT)
    path = base.job_dir(job)
    return {"ok": True, "free_bytes": stats.f_bavail * stats.f_frsize,
            "job_bytes": tree_bytes(path) if os.path.isdir(path) else 0}


def predict_clean(job):
    path = base.owned_job(job)
    keeper = keeper_state(path)
    if keeper.get("active"):
        # asked to stop rather than cancelled, so it ends COMPLETED and its email says so
        with open(os.path.join(path, CLEARING), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        deadline = time.time() + 90
        while time.time() < deadline and keeper_state(path).get("active"):
            time.sleep(5)
    ids = []
    for unit in unit_names(path):
        paths = unit_paths(path, unit)
        state = load_json(paths["state"], {}) or {}
        ids += [state.get("prep_job_id")] + manifest_lines(paths["array_id"])[:1]
        ids += [a["array_id"] for a in state.get("attempts") or []]
    keeper = keeper_state(path)
    if keeper.get("active"):
        ids.append(keeper["job_id"])
    ids = [str(i) for i in ids if i]
    if ids:
        try:
            base.slurm("scancel", *ids)
        except ValueError:
            pass
    shutil.rmtree(path)
    return {"ok": True, "removed": path, "cancelled": ids}


# -- the keeper -----------------------------------------------------------------------------------

def keeper_state(path):
    """Whether the round's keeper job is running, and what it last wrote down."""
    record = load_json(os.path.join(path, KEEPER_FILE), {}) or {}
    job_id = record.get("job_id")
    entry = (slurm_states([job_id]) or {}).get(str(job_id)) if job_id else None
    submitted = record.get("submitted_at")
    fresh = bool(submitted) and time.time() - time.mktime(
        time.strptime(submitted, "%Y-%m-%d %H:%M:%S")) < 600
    exists = bool(job_id) and (busy(entry) or (entry is None and fresh
                                               and not record.get("finished_at")))
    # a keeper that is queued -- e.g. behind a maintenance window -- looks after nothing yet, so a
    # PC that is on keeps acting itself
    active = exists and ((entry or {}).get("state") == "RUNNING" or fresh)
    return {"job_id": job_id, "slurm": (entry or {}).get("state"), "exists": exists,
            "active": active, "summary": record.get("summary"),
            "finished_at": record.get("finished_at")}


def write_keeper(path, **fields):
    """Update keeper.json -- never recreating a round the PC has just deleted."""
    if not os.path.isdir(path):
        return
    record = load_json(os.path.join(path, KEEPER_FILE), {}) or {}
    record.update(fields)
    try:
        save_json(os.path.join(path, KEEPER_FILE), record)
    except OSError:
        pass


def minutes_before_maintenance():
    """Minutes until the next maintenance reservation starts (less a margin), or None when none is
    announced. A job whose time limit reaches into one is not started until it is over."""
    try:
        out = base.slurm("scontrol", "show", "reservation", "-o")
    except ValueError:
        return None
    now, soonest = time.time(), None
    for line in out.splitlines():
        fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
        if "MAINT" not in fields.get("Flags", ""):
            continue
        try:
            start = time.mktime(time.strptime(fields["StartTime"], "%Y-%m-%dT%H:%M:%S"))
            end = time.mktime(time.strptime(fields["EndTime"], "%Y-%m-%dT%H:%M:%S"))
        except (KeyError, ValueError):
            continue
        if end > now and start > now:
            soonest = start if soonest is None else min(soonest, start)
    return None if soonest is None else int((soonest - now) / 60) - 10


def keeper_time():
    """The keeper's time request: up to 6 days, ending before the next maintenance window; when
    that is less than an hour away (or slurm cannot be asked), slurm is let shorten it."""
    room = minutes_before_maintenance()
    if room is not None and room >= 60:
        return ["--time=%d" % min(KEEPER_MAX_MINUTES, room)]
    return ["--time=%d" % KEEPER_MAX_MINUTES, "--time-min=60"]


def round_name(path, job):
    runs = sorted({load_unit(path, u)["run_name"] for u in unit_names(path)})
    return (runs[0] if runs else job) + ("_and_more" if len(runs) > 1 else "")


def predict_keeper(job, mail=True, successor=False):
    """Start the round's keeper, unless one is queued or running already (a successor is what a
    keeper starts for itself when its time is nearly up)."""
    path = base.owned_job(job)
    with job_lock(path):
        current = keeper_state(path)
        if current.get("exists") and not successor:
            return {"ok": True, "keeper_job_id": current["job_id"], "already_running": True}
        if not successor:
            # a round already over needs no keeper -- and one started now would email about it
            status = predict_status(job)
            if status["working"] is False and all(
                    settled(r, status["units"].get(k.split("/")[0], {}))
                    for k, r in status["movies"].items()):
                return {"ok": True, "keeper_job_id": None, "not_needed": True}
        name = ("keep_predictions_" + round_name(path, job))[:60]
        command = "cd %s && exec .env/bin/python -u code/local_reanalysis_server.py predict-keep " \
                  "--job %s" % (shlex.quote(PROJECT), shlex.quote(job))
        out = base.slurm("sbatch", "--parsable", "-J", name, *KEEPER_SBATCH, *keeper_time(),
                         "--output", os.path.join(PROJECT, "logs", "%x_%j.out"),
                         "--error", os.path.join(PROJECT, "logs", "%x_%j.err"),
                         "--wrap", command, env=slurm_environment())
        job_id = out.strip().splitlines()[-1].split(";")[0]
        try:
            os.remove(os.path.join(path, CLEARING))
        except OSError:
            pass
        save_json(os.path.join(path, KEEPER_FILE),
                  {"job_id": job_id, "name": name, "mail": bool(mail),
                   "successor": bool(successor),
                   "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    return {"ok": True, "keeper_job_id": job_id, "name": name}


def notify(path, job, needs_pc):
    """Have slurm email the round's owner that it is over, by a job that does nothing but end --
    slurm's own mail, since compute nodes cannot send mail. Its name is the message. No
    --mail-user: since the 2026-10 upgrade slurm mails only the submitting account, and refuses
    a job that names an address."""
    record = load_json(os.path.join(path, KEEPER_FILE), {}) or {}
    if not record.get("mail", True):
        return
    rows = predict_status(job)["movies"] if os.path.isdir(path) else {}
    done = sum(1 for r in rows.values() if r["state"] in ("predicted", "released"))
    name = ("predictions_%s_%d_of_%d_%s" % (round_name(path, job), done, len(rows),
                                            "needs_the_PC" if needs_pc else "ready"))[:90]
    try:
        base.slurm("sbatch", "-J", name, "--partition=glacier", "--gres=gpu:0", "--mem=100m",
                   "--cpus-per-task=1", "--time=5", "--output=/dev/null",
                   "--mail-type=END", "--wrap", "true")
        print("  emailing the owner: %s" % name, flush=True)
    except ValueError as e:
        print("  could not send the email: %s" % e, flush=True)


def own_end_time():
    """When slurm will stop this keeper (epoch seconds), or None."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        out = base.slurm("squeue", "-h", "-j", job_id, "-o", "%e").strip()
        return time.mktime(time.strptime(out.splitlines()[0], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, IndexError):
        return None


def settled(row, unit):
    """Nothing more the cluster can do for this movie on its own."""
    state = row.get("state")
    if state in SETTLED:
        return True
    if state == "failed":
        return not row.get("retryable")
    if state == "prep_crashed":
        return not row.get("instant") or unit.get("prep_resubmits", 0) >= MAX_RESUBMITS
    return False


def keep_once(job, status):
    """One look: retry what failed, resubmit what never started. Returns what it did."""
    rows, units = status["movies"], status["units"]
    done = []
    retry = sorted(k for k, r in rows.items() if r["state"] == "failed" and r.get("retryable"))
    if retry:
        answer = predict_retry(job, retry)
        if answer["submitted"]:
            done.append("retried %s" % ", ".join(k for k in retry if k not in answer["skipped"]))
    lost = sorted({k.split("/")[0] for k, r in rows.items()
                   if r["state"] == "prep_crashed" and r.get("instant")
                   and units.get(k.split("/")[0], {}).get("prep_resubmits", 0) < MAX_RESUBMITS})
    for unit in lost:
        path = base.job_dir(job)
        state = load_json(unit_paths(path, unit)["state"], {}) or {}
        predict_submit(job, unit, state.get("throttle") or DEFAULT_THROTTLE, again=True)
        done.append("submitted %s's prep again (it never started)" % unit)
    return done


def tally(rows):
    counts = {}
    for row in rows.values():
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return ", ".join("%s %d" % (k, v) for k, v in sorted(counts.items()))


def predict_keep(job, interval=KEEP_INTERVAL):
    """Look after a round until nothing more can happen on the cluster, then email the owner
    whether it is ready to fetch or needs the PC (a prep crashed, or the round is stuck)."""
    path = base.job_dir(job)
    ends = own_end_time()
    print("%s  looking after round %s every %d min%s" % (
        time.strftime("%H:%M"), job, interval // 60,
        "; this job may run until %s" % time.strftime("%Y-%m-%d %H:%M", time.localtime(ends))
        if ends else ""), flush=True)
    last, idle = None, 0
    # only a keeper that watched something happen emails: one that finds the round already over
    # was started by a PC that is on, and its owner is looking
    # ...but a successor carries on its predecessor's watch, which saw the round under way
    saw_work = bool((load_json(os.path.join(path, KEEPER_FILE), {}) or {}).get("successor"))
    while True:
        if ends and time.time() > ends - HANDOVER_SECONDS and os.path.isdir(path):
            # out of time (a maintenance window, or the 6 days): a successor carries on, and
            # will wait in the queue for as long as the cluster is down
            record = load_json(os.path.join(path, KEEPER_FILE), {}) or {}
            answer = predict_keeper(job, mail=record.get("mail", True), successor=True)
            print("%s  this keeper's time is nearly up; keeper %s carries on" % (
                time.strftime("%H:%M"), answer["keeper_job_id"]), flush=True)
            return 0
        if not os.path.isdir(path) or os.path.isfile(os.path.join(path, CLEARING)):
            print("%s  the round was brought home and cleared from the PC; done"
                  % time.strftime("%H:%M"), flush=True)
            return 0
        try:
            status = predict_status(job)
            actions = keep_once(job, status)
        except ValueError as e:
            if not os.path.isdir(path):
                continue
            print("%s  could not look this time: %s" % (time.strftime("%H:%M"), e), flush=True)
            actions, status = [], None
        if status is not None:
            rows = status["movies"]
            now = tally(rows)
            for action in actions:
                print("%s  %s" % (time.strftime("%H:%M"), action), flush=True)
            if now != last:
                print("%s  %s" % (time.strftime("%H:%M"), now), flush=True)
                last = now
            write_keeper(path, summary=now, looked_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            all_settled = all(settled(r, status["units"].get(k.split("/")[0], {}))
                              for k, r in rows.items())
            saw_work = saw_work or bool(actions) or not all_settled or bool(status["working"])
            if not actions and status["working"] is not None and not status["working"]:
                if all_settled:
                    needs_pc = sorted(k for k, r in rows.items() if r["state"] == "prep_crashed")
                    print("%s  nothing more can happen on the cluster: %s" % (
                        time.strftime("%H:%M"), now), flush=True)
                    if needs_pc:
                        print("  a prep crashed for %s; run predict.bat again to send it anew"
                              % ", ".join(needs_pc), flush=True)
                    write_keeper(path, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                 needs_pc=needs_pc)
                    # a watching PC mends a crashed prep itself; only an absent one is told
                    if saw_work and not (needs_pc and pc_present(path)):
                        notify(path, job, bool(needs_pc))
                    return 0
                idle += 1
                if idle >= 3:
                    print("%s  nothing is running for the movies still open (%s); the PC "
                          "will deal with them" % (time.strftime("%H:%M"), now), flush=True)
                    write_keeper(path, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                 stuck=True)
                    if saw_work and not pc_present(path):
                        notify(path, job, True)
                    return 0
            else:
                idle = 0
        # sleep in short steps, so a PC clearing the round is never kept waiting
        for _ in range(max(1, interval // 10)):
            if not os.path.isdir(path) or os.path.isfile(os.path.join(path, CLEARING)):
                break
            time.sleep(10)


def add_verbs(sub):
    submit = sub.add_parser("predict-submit")
    submit.add_argument("--job", required=True)
    submit.add_argument("--unit", required=True)
    submit.add_argument("--throttle", type=int, default=DEFAULT_THROTTLE)
    submit.add_argument("--again", action="store_true")
    for name in ("predict-status", "predict-space", "predict-clean"):
        sub.add_parser(name).add_argument("--job", required=True)
    keeper = sub.add_parser("predict-keeper")
    keeper.add_argument("--job", required=True)
    keeper.add_argument("--no-mail", action="store_true")
    keep = sub.add_parser("predict-keep")
    keep.add_argument("--job", required=True)
    keep.add_argument("--interval", type=int, default=KEEP_INTERVAL)
    for name in ("predict-retry", "predict-release"):
        verb = sub.add_parser(name)
        verb.add_argument("--job", required=True)
        verb.add_argument("--movies", required=True, help="comma-separated U/movN keys")
    reset = sub.add_parser("predict-reset")
    reset.add_argument("--job", required=True)
    reset.add_argument("--unit", required=True)
    fetch = sub.add_parser("predict-fetch")
    fetch.add_argument("--job", required=True)
    fetch.add_argument("--movies", default="")
    fetch.add_argument("--units", default="")


def split(value):
    return [v for v in (value or "").split(",") if v]


def handlers(args):
    return {
        "predict-submit": lambda: predict_submit(args.job, args.unit, args.throttle,
                                                 args.again),
        "predict-status": lambda: (note_pc(args.job), predict_status(args.job))[1],
        "predict-retry": lambda: predict_retry(args.job, split(args.movies)),
        "predict-reset": lambda: predict_reset(args.job, args.unit),
        "predict-release": lambda: predict_release(args.job, split(args.movies)),
        "predict-space": lambda: predict_space(args.job),
        "predict-clean": lambda: predict_clean(args.job),
        "predict-keeper": lambda: predict_keeper(args.job, not args.no_mail),
    }
