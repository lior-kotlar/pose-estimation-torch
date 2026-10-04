"""Bring predicted movies kept on a PC up to date, doing only what each one still needs.

The plain-language guide is LOCAL_REANALYSIS.md; this is the program behind the double-click
files in local_reanalysis/:

    python code/local_reanalysis.py setup             once per PC: username, datasets folder, key
    python code/local_reanalysis.py run [FOLDER ...]  do what each movie needs, collect, upload
    python code/local_reanalysis.py update            download the latest committed code
    python code/local_reanalysis.py predict [FOLDER]  prep and predict raw movies kept on this PC,
                                                      on the cluster (code/local_predict.py,
                                                      LOCAL_PREDICT.md)

A run takes one or more folders -- a single experiment, or a folder holding many -- surveys every
movie in them (code/movie_survey.py), and does only the stages that have work in them:

  realign    one of the pose models labelled the two wings the other way round, so the ensemble
             mixed them into one physical wing. Only re-running the ensemble fixes it, which is
             half an hour of computing, so it happens on the cluster.
  reanalyse  the analysis h5, CSV, plots, viewer and pages are older than the analysis code, the
             declaration or the 3D points. Seconds a movie, here.
  render     the overlay mp4 was made from points the movie no longer has. The camera images it
             needs are the one thing a movie folder does not hold, so a shrunk copy goes to the
             cluster and the finished video comes back.

The dependency between them is not a rule this file enforces; it falls out of the fingerprints.
A new ensemble changes the points fingerprint, which makes the analysis stale, which changes the
analysed points, which makes the video stale. So a run surveys again after every stage rather
than deciding everything up front -- and a movie the cluster refused to realign quietly drops out
of the later stages instead of being redone for nothing.

The two cluster stages need an account that may write there, so only the pipeline's owner runs
them; anyone else gets the survey, which reads nothing but their own disk.

Everything it talks to the cluster about goes through ssh to code/local_reanalysis_server.py in
the cluster's copy of the project, one connection per step. None of it runs on the login gateway:
every command is handed to slurm with srun, which runs it on a compute node (see on_node and the
srun_flags setting), and only the login itself, and the one line that adds this PC's ssh key
during setup, happen on the gateway.

Movies of experiments the cluster does not know are fine: they keep the declaration their old h5
recorded, and are collected under the experiment their own records name, or local_only/<folder>
when they name none.
"""
import argparse
import datetime as dt
import glob
import io
import json
import os
import posixpath
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (CODE_DIR, os.path.join(CODE_DIR, 'prediction_code_lior')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cluster_round as rounds
import movie_survey
from cluster_link import (Problem, SERVER_HELPER, count, helper_command, on_node, remote,
                          sha256, stage)

# The folder the code was unpacked into on the PC; everything the tool keeps lives next to it.
HOME = os.path.dirname(CODE_DIR)
SETTINGS_PATH = os.path.join(HOME, 'local_reanalysis_settings.json')
DECLARATIONS_CACHE = os.path.join(HOME, 'declarations_cache')
REPORTS_DIR = os.path.join(HOME, 'reports')
BUNDLE_COMMIT = os.path.join(HOME, 'BUNDLE_COMMIT')
# What `update` downloads: the committed versions of these paths in the cluster's project.
BUNDLE_PATHS = ('code', 'local_reanalysis', 'requirements-analysis.txt', 'LOCAL_REANALYSIS.md',
                'LOCAL_PREDICT.md')

DEFAULT_SETTINGS = {
    'server_user': '',
    'server_host': 'moriah-gw-01.cs.huji.ac.il',
    'server_project': '/cs/labs/tsevi/lior.kotlar/pose-estimation-torch',
    # empty: <server_project>/collected_h5
    'upload_to': '',
    # whether this PC may publish to the cluster. setup turns it off for anyone who cannot
    # write to the destination: only the pipeline's owner uploads, everyone else keeps their
    # re-analysed movies on their own machine.
    'upload': True,
    # empty: collected_h5 next to the code on this PC
    'collected_h5': '',
    # 0: half the processor threads, one movie each
    'jobs': 0,
    # where this PC keeps the source datasets -- the box h5 a movie was predicted from, its
    # calibration and its perturbation.json. The paths a prediction recorded are the cluster's,
    # and the datasets are far too large to keep there, so this is how a movie's own data is
    # found once it has been moved off. Empty: only the recorded paths are tried.
    'dataset_root': '',
    # Nothing this PC asks for runs on the login gateway: every command is handed to slurm,
    # which places it on a compute node. Emptying this runs them on the login host instead.
    'srun_flags': '--ntasks=1 --cpus-per-task=1 --mem=4g --time=2:00:00 --gres=gpu:0 '
                  '--chdir=/tmp --job-name=pose_pc',
    # predict.bat (code/local_predict.py): where predictions of raw movies sent from this PC go
    # (empty: predict_output next to the code), the server's predict config, and how much goes
    # over one connection at a time -- each is one srun step, which slurm caps at 2 hours
    'predict_output': '',
    'predict_config': 'config1.json',
    'upload_chunk_mb': 2000,
    'fetch_chunk_mb': 2000,
    # the cluster's disk is shared by the lab: a round never fills it past this much free space
    'server_reserve_gb': 30,
    # a predict round's keeper on the cluster has slurm email the owner when the round is done
    'email_when_done': True,
}
UPLOAD_LEDGER = '.uploaded.json'
# where the rounds this PC has going are written down, so one can be picked up again
REALIGN_JOBS_DIR = os.path.join(HOME, rounds.JOBS_DIRNAME)
REALIGN_MARKER = '.realigned_ensemble.json'
BLOCKED_REPORT = '.realign_staging/BLOCKED.json'
# left in a movie the cluster refused to change, so later rounds do not re-run it for nothing
BLOCKED_MARKER = '.realign_blocked.json'
POINTS_ALL = 'points_3D_all.npy'
POINTS_SMOOTHED = 'points_3D_smoothed_ensemble_best_method.npy'
OLD_POINTS = (POINTS_SMOOTHED, 'points_3D_ensemble_best_method.npy')
# what a rendered movie carries back, and the sidecar that says which cameras saw a whole fly
MP4_NAME = 'movie 2D and 3D.mp4'
VIDEO_STAMP = 'video.json'
CAM_VALIDITY = 'prescan_cam_validity.npz'
# Set for the re-run that follows an automatic update, so it cannot update again in a loop.
UPDATED_FLAG = 'POSE_REANALYSIS_JUST_UPDATED'




class Tee:
    """Write everything printed to the console to a log file as well.

    A run's per-movie messages are the only record of a movie that failed, or of a figure that
    was skipped, and the console window is usually closed long before anyone asks."""

    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle

    def write(self, text):
        self.stream.write(text)
        self.handle.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        return self.stream.isatty()


# -- settings and the ssh connection -------------------------------------------------------------

def load_settings(required=True):
    settings = dict(DEFAULT_SETTINGS)
    if os.path.isfile(SETTINGS_PATH):
        with open(SETTINGS_PATH, encoding='utf-8') as f:
            settings.update(json.load(f))
    elif required:
        raise Problem(f"no settings yet ({SETTINGS_PATH}); run local_reanalysis\\setup.bat first")
    return settings


def save_settings(settings):
    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)


def upload_destination(settings):
    return settings['upload_to'] or posixpath.join(settings['server_project'], 'collected_h5')


def may_upload(settings):
    """True when this PC is allowed to publish to the cluster."""
    return bool(settings.get('upload', True))


def destination_writable(settings):
    """Whether this account can write the upload destination on the cluster.

    Tests the nearest folder of it that exists, since the destination itself is created on the
    first upload."""
    dest = shlex.quote(upload_destination(settings))
    command = (f'd={dest}; while [ ! -e "$d" ] && [ "$d" != "/" ]; do d=$(dirname "$d"); done; '
               'if [ -w "$d" ]; then echo WRITABLE; else echo READONLY; fi')
    try:
        proc = remote(settings, on_node(settings, command, attempts=3),
                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = proc.communicate(timeout=600)
    except (OSError, subprocess.SubprocessError, Problem):
        return None
    if proc.returncode != 0:
        return None
    return out.decode('utf-8', errors='replace').strip().endswith('WRITABLE')


def collected_root(settings):
    return settings['collected_h5'] or os.path.join(HOME, 'collected_h5')












# -- setup ---------------------------------------------------------------------------------------

def ask(question, default=''):
    shown = f" [{default}]" if default else ''
    answer = input(f"{question}{shown}: ").strip().strip('"')
    return answer or default


def setup(args):
    settings = load_settings(required=False)
    print("Pose-estimation re-analysis: one-time setup. Press Enter to keep a value in [brackets].\n")
    if not shutil.which('ssh'):
        raise Problem("the 'ssh' command was not found. On Windows, turn on 'OpenSSH Client' in "
                      "Settings > System > Optional features, then run setup.bat again")
    while True:
        settings['server_user'] = ask("Your username on the lab server (e.g. lior.kotlar)",
                                      settings['server_user'])
        if settings['server_user']:
            break
    settings['server_host'] = ask("Server address", settings['server_host'])
    print("\nWhere do you keep the source datasets -- the folder that holds each experiment's")
    print("movies as they came off the rig (the box h5 files and calibration.h5)? Needed only to")
    print("rebuild a movie's video; leave empty if you do not have them here.")
    root = ask("Datasets folder (e.g. E:\\Lior\\inference_datasets)", settings['dataset_root'])
    if root and not os.path.isdir(root):
        print(f"  note: {root} is not a folder on this PC. Saving it anyway -- correct it in "
              f"{os.path.basename(SETTINGS_PATH)} if it is wrong.")
    settings['dataset_root'] = root
    save_settings(settings)
    print(f"\nsaved {SETTINGS_PATH}")

    key = os.path.join(os.path.expanduser('~'), '.ssh', 'id_ed25519')
    if ask("\nLog in to the server without a password from now on? (recommended) y/n", 'y').lower().startswith('y'):
        if not os.path.isfile(key + '.pub'):
            os.makedirs(os.path.dirname(key), exist_ok=True)
            subprocess.run(['ssh-keygen', '-t', 'ed25519', '-N', '', '-q', '-f', key], check=True)
            print(f"created an ssh key: {key}")
        with open(key + '.pub', encoding='utf-8') as f:
            public = f.read().strip()
        print("adding it to your server account -- type your SERVER password if asked "
              "(nothing shows while you type)")
        # the one thing that does not go to a compute node: it sets up the login itself, and it
        # runs before there is a key to log in with. Two shell builtins on the account's own
        # ~/.ssh, no work
        command = ("umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; "
                   f"grep -qxF {shlex.quote(public)} ~/.ssh/authorized_keys || "
                   f"echo {shlex.quote(public)} >> ~/.ssh/authorized_keys")
        if remote(settings, command).wait() != 0:
            raise Problem("could not add the key; check the username and password and run setup.bat again")

    print("\nchecking the connection and the server's copy of the project ...", flush=True)
    print("(the work runs on a cluster node, so this waits for slurm to give it one)", flush=True)
    check = remote(settings, on_node(
        settings, f"test -f {shlex.quote(posixpath.join(settings['server_project'], SERVER_HELPER))}"
                  " && python3 -c \"import socket; print('server ok on', socket.gethostname())\""),
        stdout=subprocess.PIPE)
    answer, _ = check.communicate()
    answer = answer.decode('utf-8', errors='replace').strip()
    if check.returncode != 0 or 'server ok' not in answer:
        raise Problem(f"connected, but the check did not come back. Either {SERVER_HELPER} is not "
                      f"under {settings['server_project']}, or the server has no python3, or "
                      f"slurm could not be reached to run it on a node ({answer or 'no answer'})")
    print(answer)
    writable = destination_writable(settings)
    settings['upload'] = writable is not False
    save_settings(settings)
    if writable is False:
        print(f"\nThis account cannot write to {upload_destination(settings)},\n"
              f"so this PC will re-analyse and collect movies for itself only -- nothing is "
              f"uploaded.\nThe results stay in {collected_root(settings)} and in each movie's "
              f"own folder.")
    elif writable is None:
        print("\ncould not tell whether the upload folder is writable; uploads are left on")

    print(f"\nSetup finished. To re-analyse, run local_reanalysis\\reanalyse.bat (see LOCAL_REANALYSIS.md).")
    return 0


# -- update --------------------------------------------------------------------------------------

def safe_member(name):
    parts = name.replace('\\', '/').split('/')
    return bool(name) and not name.startswith('/') and '..' not in parts and ':' not in parts[0]


def update(args):
    settings = load_settings()
    project = settings['server_project']
    print(f"downloading the latest committed code from {settings['server_host']}:{project} ...")
    command = (f"cd {shlex.quote(project)} && git -c 'safe.directory=*' archive --format=tar HEAD "
               + ' '.join(shlex.quote(p) for p in BUNDLE_PATHS))
    staging = os.path.join(HOME, '.update_new')
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    proc = remote(settings, on_node(settings, command), stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
        for member in tar:
            if not safe_member(member.name):
                raise Problem(f"unexpected path in the download: {member.name}")
            if member.isdir():
                os.makedirs(os.path.join(staging, *member.name.split('/')), exist_ok=True)
            elif member.isfile():
                target = os.path.join(staging, *member.name.split('/'))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with tar.extractfile(member) as src, open(target, 'wb') as out:
                    shutil.copyfileobj(src, out)
        commit = (tar.pax_headers or {}).get('comment', '')[:12]
    if proc.wait() != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise Problem("the download failed; nothing was changed")

    old_requirements = os.path.join(HOME, 'requirements-analysis.txt')
    before = open(old_requirements, 'rb').read() if os.path.isfile(old_requirements) else b''
    retired = os.path.join(HOME, '.update_old')
    shutil.rmtree(retired, ignore_errors=True)
    os.makedirs(retired)
    for name in BUNDLE_PATHS:
        new, current = os.path.join(staging, name), os.path.join(HOME, name)
        if not os.path.exists(new):
            continue
        if name == 'code':
            # swapped whole, so a module deleted on the cluster does not linger here
            if os.path.exists(current):
                os.replace(current, os.path.join(retired, name))
            os.replace(new, current)
        elif os.path.isdir(new):
            # overwritten file by file: update.bat is running from this folder, and Windows
            # will not move a folder while a file inside it is open
            for dirpath, _dirnames, filenames in os.walk(new):
                target_dir = os.path.join(current, os.path.relpath(dirpath, new))
                os.makedirs(target_dir, exist_ok=True)
                for filename in filenames:
                    shutil.copyfile(os.path.join(dirpath, filename), os.path.join(target_dir, filename))
        else:
            shutil.copyfile(new, current)
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(retired, ignore_errors=True)
    if commit:
        with open(BUNDLE_COMMIT, 'w', encoding='utf-8') as f:
            f.write(commit[:7] + '\n')
    print(f"code updated to commit {commit[:7] or '(unknown)'}")

    if open(old_requirements, 'rb').read() != before:
        print("the package list changed; installing ...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', old_requirements], check=True)
    return 0


# -- run -----------------------------------------------------------------------------------------

def bundle_commit():
    """The cluster commit this copy of the code was downloaded at, or '' when unknown."""
    try:
        with open(BUNDLE_COMMIT, encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return ''


def server_commit(settings):
    """The commit the cluster's project is at, or '' when it cannot be asked."""
    command = (f"cd {shlex.quote(settings['server_project'])} && "
               "git -c 'safe.directory=*' rev-parse --short HEAD")
    try:
        proc = remote(settings, on_node(settings, command, attempts=3),
                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = proc.communicate(timeout=600)
    except (OSError, subprocess.SubprocessError, Problem):
        return ''
    return out.decode('utf-8', errors='replace').strip() if proc.returncode == 0 else ''


def update_if_outdated(args, settings):
    """Update to the cluster's committed code, then redo the run with it.

    The analysis code decides what the products contain, so re-analysing with a copy older than
    the cluster's only earns the movies another re-analysis later. Returns the exit code of the
    re-run, or None when this copy is already current."""
    if args.no_update or os.environ.get(UPDATED_FLAG):
        return None
    here, there = bundle_commit(), server_commit(settings)
    if not there or there == here:
        return None
    print(f"the server has newer code ({here or 'unknown'} -> {there}); updating before the run",
          flush=True)
    update(args)
    print("\nstarting the run with the updated code", flush=True)
    return subprocess.run([sys.executable, '-X', 'utf8'] + sys.argv,
                          env={**os.environ, UPDATED_FLAG: '1'}).returncode


def declaration_candidates(settings, box_paths):
    """The perturbation.json paths load_perturbation would try for these source movies."""
    wanted = set()
    for box in box_paths:
        box = box.replace('\\', '/')
        if not box.startswith('/'):
            # old member configs record the movie relative to the project
            box = posixpath.join(settings['server_project'], box)
        movie_dir = posixpath.dirname(box)
        wanted.add(posixpath.join(movie_dir, 'perturbation.json'))
        wanted.add(posixpath.join(posixpath.dirname(movie_dir), 'perturbation.json'))
    return sorted(wanted)


def fetch_declarations(settings, box_paths):
    """Mirror the needed perturbation.json files into DECLARATIONS_CACHE; returns path maps."""
    wanted = declaration_candidates(settings, box_paths)
    # a declaration deleted on the cluster must not survive in the cache
    for path in wanted:
        cached = os.path.join(DECLARATIONS_CACHE, *path.lstrip('/').split('/'))
        if os.path.isfile(cached):
            os.remove(cached)
    proc = remote(settings, helper_command(settings, 'declarations'),
                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc.stdin.write(json.dumps(wanted).encode('utf-8'))
    proc.stdin.close()
    allowed = {p.lstrip('/') for p in wanted}
    received = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
            for member in tar:
                if not member.isfile() or member.name not in allowed:
                    continue
                target = os.path.join(DECLARATIONS_CACHE, *member.name.split('/'))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with tar.extractfile(member) as src, open(target, 'wb') as out:
                    shutil.copyfileobj(src, out)
                received += 1
    except tarfile.TarError as e:
        proc.wait()
        raise Problem(f"could not download the declarations from the server ({e}); nothing was "
                      "re-analysed. Check the connection and run again")
    if proc.wait() != 0:
        raise Problem("could not download the declarations from the server; nothing was "
                      "re-analysed. Check the connection and run again")

    # every recorded cluster path is looked up inside the cache; a path recorded relative to
    # the project (old member configs) is looked up under the project's mirror
    maps = set()
    project_mirror = os.path.join(DECLARATIONS_CACHE, *settings['server_project'].lstrip('/').split('/'))
    for box in box_paths:
        box = box.replace('\\', '/')
        first = box.lstrip('/').split('/')[0]
        if box.startswith('/'):
            maps.add(('/' + first, os.path.join(DECLARATIONS_CACHE, first)))
        else:
            maps.add((first, os.path.join(project_mirror, first)))
    return received, tuple(sorted(maps))


def auto_jobs(settings):
    return int(settings.get('jobs') or 0) or max(1, (os.cpu_count() or 2) // 2)


def is_bad(movie_dir, root, excluded):
    rel = os.path.relpath(movie_dir, root).split(os.sep)
    return any(part in excluded for part in rel)


def collect_movies(settings, done_dirs, movie_root, groups, collector):
    """Copy the analysis h5 of done_dirs into the PC's collected_h5 tree."""
    sources, keys = [], {}
    for movie_dir in done_dirs:
        for name in sorted(os.listdir(movie_dir)):
            if name.endswith('_' + collector.SUFFIX):
                path = os.path.join(movie_dir, name)
                sources.append(path)
                keys[path] = groups[movie_dir]
    return collector.collect(sources, collected_root(settings), keys=keys)


def upload(settings):
    """Send every collected h5 the cluster has not confirmed yet; returns (sent, results)."""
    root = collected_root(settings)
    destination = upload_destination(settings)
    ledger_path = os.path.join(root, UPLOAD_LEDGER)
    ledger = {}
    if os.path.isfile(ledger_path):
        with open(ledger_path, encoding='utf-8') as f:
            ledger = json.load(f)
    confirmed = ledger.setdefault('confirmed', {}).setdefault(destination, {})
    hashes = ledger.setdefault('hashes', {})

    pending = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not (d.startswith('superseded_') or d.startswith('.')))
        for name in sorted(filenames):
            if not name.endswith('.h5'):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace(os.sep, '/')
            stat = os.stat(path)
            cached = hashes.get(rel)
            if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
                digest = cached[2]
            else:
                digest = sha256(path)
                hashes[rel] = [stat.st_size, stat.st_mtime_ns, digest]
            if confirmed.get(rel) != digest:
                pending[rel] = digest
    if not pending:
        print("nothing new to upload: the server already has every collected file")
        save_ledger(ledger_path, ledger)
        return 0, {}

    size = sum(os.path.getsize(os.path.join(root, *rel.split('/'))) for rel in pending)
    print(f"uploading {len(pending)} file(s), {size / 1e6:.1f} MB, to "
          f"{settings['server_host']}:{destination}", flush=True)
    proc = remote(settings, helper_command(settings, 'receive', '--dest', destination),
                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin, mode='w|') as tar:
            manifest = json.dumps({'files': pending}).encode('utf-8')
            info = tarfile.TarInfo('MANIFEST.json')
            info.size = len(manifest)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(manifest))
            for i, rel in enumerate(sorted(pending), 1):
                tar.add(os.path.join(root, *rel.split('/')), arcname=rel, recursive=False)
                if i % 25 == 0 or i == len(pending):
                    print(f"  sent {i}/{len(pending)}", flush=True)
        proc.stdin.close()
    except (BrokenPipeError, OSError) as e:
        output = proc.stdout.read().decode('utf-8', errors='replace')
        proc.wait()
        raise Problem(f"the upload was cut off ({e}). {output.strip()} -- the analysis and the "
                      "local collection are kept; run again to retry the upload")
    output = proc.stdout.read().decode('utf-8', errors='replace').strip()
    returncode = proc.wait()
    try:
        answer = json.loads(output.splitlines()[-1])
    except (IndexError, ValueError):
        answer = {'ok': False, 'error': output or f'no answer from the server (exit {returncode})'}
    if not answer.get('ok'):
        raise Problem(f"the server did not accept the upload: {answer.get('error')}. The analysis "
                      "and the local collection are kept; run again to retry the upload. If it "
                      "says the folder cannot be written, this account may not be allowed to "
                      "publish to the cluster -- run setup.bat again, which turns uploading off "
                      "for such an account")
    for rel in answer['results']:
        confirmed[rel] = pending[rel]
    save_ledger(ledger_path, ledger)
    return len(pending), answer['results']


def save_ledger(path, ledger):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    staged = path + '.partial'
    with open(staged, 'w', encoding='utf-8') as f:
        json.dump(ledger, f, indent=1)
    os.replace(staged, path)



def run(args):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    log_path = os.path.join(REPORTS_DIR, f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_path, 'w', encoding='utf-8') as handle:
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(out, handle), Tee(err, handle)
        try:
            return run_steps(args, log_path)
        except Problem as e:
            # inside the tee, so the log says why the run stopped
            print(f"\nSTOPPED: {e}", flush=True)
            print(f"log of this run  : {log_path}", flush=True)
            return 1
        finally:
            sys.stdout, sys.stderr = out, err


def wanted_folders(args, what='re-analyse'):
    folders = list(args.folders)
    if not folders:
        answer = ask(f"Folder to {what} (one experiment, or a folder of experiments)")
        folders = [answer] if answer else []
    roots = []
    for folder in folders:
        folder = os.path.abspath(folder.strip().strip('"'))
        if not os.path.isdir(folder):
            raise Problem(f"not a folder: {folder}")
        roots.append(folder)
    if not roots:
        raise Problem("no folder given")
    return roots


def dataset_roots(settings):
    root = (settings.get('dataset_root') or '').strip()
    return (root,) if root else ()


class Work:
    """One run's picture of a folder of movies, refreshed between stages.

    Surveying is cheap and the stages change what the answers are, so a run asks again after every
    one rather than deciding everything up front. That is what keeps the dependency between the
    stages honest: a movie the cluster refused to realign simply stops appearing in the later
    stages, because its fingerprints never moved."""

    def __init__(self, settings, roots, args):
        import collect_analysis_h5 as collector
        import reanalyse_movies as rm
        self.settings, self.roots, self.args = settings, roots, args
        self.collector, self.rm = collector, rm
        self.roots_for_data = dataset_roots(settings)

        self.movie_root = {}
        for root in roots:
            for movie_dir in rm.find_movie_dirs(root):
                self.movie_root.setdefault(movie_dir, root)
        if not self.movie_root:
            raise Problem(f"no predicted movies (folders with {rm.POINTS_NAME}) under: "
                          f"{', '.join(roots)}")
        self.movie_dirs = sorted(self.movie_root)
        self.groups = {d: collector.experiment_key(d, self.movie_root[d]) for d in self.movie_dirs}
        self.path_maps = ()
        self.code_fp, self.commit = rm.code_fingerprint(), rm.git_commit()
        self.rows = []

    def declarations(self):
        """Mirror the experiments' perturbation.json from the cluster, for the ones it has."""
        boxes = sorted({b for b in (self.rm.recorded_source_box(d) for d in self.movie_dirs) if b})
        received, self.path_maps = fetch_declarations(self.settings, boxes)
        return received

    def look(self, show=False):
        self.rows = movie_survey.survey(self.movie_dirs, self.code_fp, self.path_maps,
                                        self.roots_for_data, group_of=self.groups.get,
                                        retry_blocked=self.args.retry_blocked,
                                        render_unknown=self.args.render_unknown, show=show)
        return self.rows

    def needing(self, stage):
        return [row for row in self.rows if stage in row['needs']]


# -- what each kind of round sends, and what it does with what comes back -------------------------

MEMBER_CONFIGS = ('specific_configuration.json', 'configuration.json')
README_GLOB = ('README_mov*.txt',)
# where a staged movie's source data and its synthesised member config are put on the cluster
STAGED_SOURCE = 'source'
STAGED_MEMBER = 'staged_member'


def realign_inputs(movie_dir):
    """What the cluster needs to re-run this movie's ensemble: (name under the movie, path here).

    Each model's 3D candidates and the config that names it, plus the movie's current ensemble
    points, which are the 'before' side of the comparison the cluster makes. Nothing else: not
    the source movie, not the analysis h5, not the video."""
    names = []
    for member in movie_survey.ensemble_members(movie_dir):
        base = os.path.basename(member)
        names.append(f'{base}/{POINTS_ALL}')
        for config in MEMBER_CONFIGS:
            if os.path.isfile(os.path.join(member, config)):
                names.append(f'{base}/{config}')
    names += [name for name in OLD_POINTS if os.path.isfile(os.path.join(movie_dir, name))]
    for pattern in README_GLOB:
        for path in sorted(glob.glob(os.path.join(glob.escape(movie_dir), pattern))):
            names.append(os.path.basename(path))
    return [(name, os.path.join(movie_dir, *name.split('/'))) for name in names]


def prepare_video(settings, state, jobs_dir, rows):
    """Shrink each movie's box and lay out what the cluster needs to render it.

    The camera images are the only bulky thing a render needs, and the renderer reads one
    time-channel per camera out of nine. Shrinking to those before sending turns 185 MB a movie
    into about 60 MB. The member config is written here rather than taken from the movie, so the
    cluster opens the copies that were just uploaded and nothing depends on where this movie's
    data lived when it was predicted."""
    import dataset_paths

    staged = {}
    for number, row in enumerate(rows, 1):
        movie_dir, key = row['movie_dir'], row['key']
        outgoing = os.path.join(jobs_dir, state['job'], 'outgoing', *key.split('/'))
        source_dir = os.path.join(outgoing, STAGED_SOURCE)
        os.makedirs(source_dir, exist_ok=True)
        box = row['source']
        reduced = os.path.join(source_dir, os.path.basename(box))
        if dataset_paths.is_render_box(box):
            # a movie predicted from this PC came home with only these channels already
            # (local_predict.py); it goes as it is
            reduced = box
        elif not os.path.isfile(reduced):
            print(f"  [{number}/{len(rows)}] shrinking {os.path.basename(movie_dir)}'s images",
                  flush=True)
            _, before, after = dataset_paths.reduce_box(box, reduced)
            print(f"      {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB", flush=True)

        files = [(f'{STAGED_SOURCE}/{os.path.basename(reduced)}', reduced)]
        beside = os.path.dirname(box)
        calibration = row.get('calibration') or os.path.join(os.path.dirname(beside),
                                                             'calibration.h5')
        for extra, name in ((calibration, 'calibration.h5'),
                            (os.path.join(beside, CAM_VALIDITY), CAM_VALIDITY)):
            if os.path.isfile(extra):
                files.append((f'{STAGED_SOURCE}/{name}', extra))

        h5_path = self_analysis_h5(movie_dir)
        files.append((os.path.basename(h5_path), h5_path))

        # the config regenerate_video reads, pointing at the copies as the cluster will see them
        on_cluster = posixpath.join(rounds.job_inputs(settings, state['job']), key, STAGED_SOURCE)
        config_path = os.path.join(outgoing, STAGED_MEMBER, 'configuration.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({'movie path': posixpath.join(on_cluster, os.path.basename(reduced)),
                       'calibration path': posixpath.join(on_cluster, 'calibration.h5'),
                       'IMAGE HEIGHT': row.get('image_height', 800),
                       'IMAGE WIDTH': row.get('image_width', 1280)}, f, indent=1)
        files.append((f'{STAGED_MEMBER}/configuration.json', config_path))
        staged[movie_dir] = files
    return staged


def self_analysis_h5(movie_dir):
    import reanalyse_movies as rm
    path = rm.analysis_h5(movie_dir)
    if not path:
        raise Problem(f"{os.path.basename(movie_dir)} has no analysis h5 to render from")
    return path


def install_realign(work, state, staging, relpaths):
    """Put each movie's new ensemble in place, keeping the one it replaces beside it."""
    import numpy as np

    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    installed = dict(state.get('installed') or {})
    for key, names in sorted(rounds.by_movie(state, relpaths).items()):
        if installed.get(key):
            print(f"  {key}: already done earlier in this round")
            continue
        movie_dir = state['movies'][key]
        source = os.path.join(staging, *key.split('/'))
        if REALIGN_MARKER not in names:
            report = os.path.join(source, *BLOCKED_REPORT.split('/'))
            record, reason = {}, 'blocked'
            if os.path.isfile(report):
                with open(report, encoding='utf-8') as f:
                    record = json.load(f)
                reason = record.get('status', 'blocked')
            print(f"  {key}: LEFT ALONE, {reason}")
            # remembered here, so a later round does not spend another half hour being refused
            record.update(movie=movie_dir, blocked_in=state['job'],
                          blocked_at=dt.datetime.now().isoformat(timespec='seconds'))
            with open(os.path.join(movie_dir, BLOCKED_MARKER), 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=1)
            rounds.record(REALIGN_JOBS_DIR, state, key, 'blocked')
            installed[key] = 'blocked'
            continue

        new_points = os.path.join(source, POINTS_SMOOTHED)
        old_points = os.path.join(movie_dir, POINTS_SMOOTHED)
        if os.path.isfile(new_points) and os.path.isfile(old_points):
            if np.load(new_points).shape != np.load(old_points).shape:
                print(f"  {key}: LEFT ALONE, the new points have a different shape from this "
                      f"movie's own; nothing was replaced")
                rounds.record(REALIGN_JOBS_DIR, state, key, 'mismatch')
                installed[key] = 'mismatch'
                continue

        archive = os.path.join(movie_dir, f'superseded_ensemble_{stamp}')
        moved = move_in(source, movie_dir, [n for n in names if n != REALIGN_MARKER], archive)
        marker = json.load(open(os.path.join(source, REALIGN_MARKER), encoding='utf-8'))
        marker.update(movie=movie_dir, installed_from=state['job'],
                      installed_at=dt.datetime.now().isoformat(timespec='seconds'))
        # the marker last, so an interrupted install leaves a movie that is redone, not one that
        # looks finished
        with open(os.path.join(movie_dir, REALIGN_MARKER), 'w', encoding='utf-8') as f:
            json.dump(marker, f, indent=1)
        if os.path.isfile(os.path.join(movie_dir, BLOCKED_MARKER)):
            os.remove(os.path.join(movie_dir, BLOCKED_MARKER))
        collapse = marker.get('collapse_pct') or []
        change = (f", frames with both wings on one wing {collapse[0]:.1f}% -> {collapse[1]:.1f}%"
                  if len(collapse) == 2 else '')
        print(f"  {key}: new ensemble in place{change}; what it replaced is in "
              f"{os.path.basename(archive)}/ ({len(moved)} file(s))")
        rounds.record(REALIGN_JOBS_DIR, state, key, 'realigned')
        installed[key] = 'realigned'
    return installed


def install_video(work, state, staging, relpaths):
    """Put each movie's new overlay video in place, keeping the one it replaces beside it."""
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    installed = dict(state.get('installed') or {})
    for key, names in sorted(rounds.by_movie(state, relpaths).items()):
        if installed.get(key):
            print(f"  {key}: already done earlier in this round")
            continue
        movie_dir = state['movies'][key]
        source = os.path.join(staging, *key.split('/'))
        if VIDEO_STAMP not in names:
            print(f"  {key}: no video came back; left alone")
            rounds.record(REALIGN_JOBS_DIR, state, key, 'no video')
            installed[key] = 'no video'
            continue
        archive = os.path.join(movie_dir, f'superseded_{stamp}')
        # the stamp last: it is what says the video in this folder is the one the h5 describes
        moved = move_in(source, movie_dir, [n for n in names if n != VIDEO_STAMP], archive)
        move_in(source, movie_dir, [VIDEO_STAMP], archive)
        mp4 = os.path.join(movie_dir, MP4_NAME)
        size = f" ({os.path.getsize(mp4) / 1e6:.0f} MB)" if os.path.isfile(mp4) else ''
        print(f"  {key}: new video in place{size}; what it replaced is in "
              f"{os.path.basename(archive)}/ ({len(moved)} file(s))")
        rounds.record(REALIGN_JOBS_DIR, state, key, 'rendered')
        installed[key] = 'rendered'
    return installed


def move_in(source, movie_dir, names, archive):
    """Move files from the staging folder into the movie, keeping what they replace."""
    moved = []
    for name in sorted(names):
        target = os.path.join(movie_dir, *name.split('/'))
        top = name.split('/')[0]
        existing = os.path.join(movie_dir, top)
        if os.path.exists(existing) and top not in moved:
            os.makedirs(archive, exist_ok=True)
            shutil.move(existing, os.path.join(archive, top))
            moved.append(top)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(os.path.join(source, *name.split('/')), target)
    return moved


ROUNDS = {
    'realign': {'what': 'realigning', 'inputs': None, 'install': install_realign,
                'note': 'Each movie takes about half an hour, longer for a long one.'},
    'video': {'what': 'rendering', 'inputs': None, 'install': install_video,
              'note': 'A video takes about 20 minutes per thousand frames.'},
}


def do_round(work, kind, rows, number, total):
    """One round of work on the cluster, resumable: send, run, watch, fetch, install, clear."""
    settings, args = work.settings, work.args
    spec = ROUNDS[kind]
    state = None if args.restart else rounds.unfinished_job(REALIGN_JOBS_DIR, work.roots, kind)
    if state:
        print(f"\ncontinuing the {kind} round started {state['created']} "
              f"({len(state['movies'])} movie(s), job {state['job']})")
    else:
        state = rounds.new_job_state(work.roots, rounds.movie_keys(rows), kind)
        rounds.save_job_state(REALIGN_JOBS_DIR, state)
    # the round's own names for the movies, whether it was just made or taken up again
    by_dir = {movie_dir: key for key, movie_dir in state['movies'].items()}
    rows = [row for row in rows if row['movie_dir'] in by_dir]
    for row in rows:
        row['key'] = by_dir[row['movie_dir']]

    if not state['uploaded']:
        stage(number, total, f"sending {len(state['movies'])} movie(s) to the cluster for "
                             f"{spec['what']}")
        if kind == 'video':
            staged = prepare_video(settings, state, REALIGN_JOBS_DIR, rows)
            # a movie of the round that is no longer in the survey has nothing to send
            files_for = lambda movie_dir: staged.get(movie_dir, [])
        else:
            files_for = realign_inputs
        rounds.upload_round(settings, state, files_for)
        state['uploaded'] = True
        rounds.save_job_state(REALIGN_JOBS_DIR, state)
    else:
        stage(number, total, f"the cluster already has these movies ({spec['what']})")

    if not state['job_id']:
        job_id, movies = rounds.submit(settings, state, args.at_once)
        state['job_id'] = job_id
        rounds.save_job_state(REALIGN_JOBS_DIR, state)
        print(f"slurm job {job_id}, {movies} task(s), at most {args.at_once} at a time")
    else:
        print(f"the cluster is already running this round (slurm job {state['job_id']})")

    print(f"\n{spec['note']} You can close this window and run the same command later to pick "
          f"it up again.", flush=True)
    rounds.wait_for_job(settings, state, args.poll_seconds)

    staging, relpaths = rounds.fetch_results(settings, REALIGN_JOBS_DIR, state)
    installed = spec['install'](work, state, staging, relpaths)
    state['installed'] = installed
    shutil.rmtree(staging, ignore_errors=True)
    if args.keep_on_server:
        print(f"kept, as asked: {settings['server_project']}/realign_jobs/{state['job']}")
    else:
        rounds.clean(settings, state)
        print("the round is gone from the cluster; this PC keeps both the new files and the ones "
              "they replaced")
    state['finished'] = True
    rounds.save_job_state(REALIGN_JOBS_DIR, state)
    shutil.rmtree(os.path.join(REALIGN_JOBS_DIR, state['job']), ignore_errors=True)
    return installed


def run_steps(args, log_path):
    started = time.time()
    settings = load_settings()
    roots = wanted_folders(args)
    upload_step = may_upload(settings) and not args.no_upload

    # before anything is imported or re-analysed, so the run uses the cluster's current code
    updated = update_if_outdated(args, settings)
    if updated is not None:
        return updated

    # heavy imports only now, so setup/update and a wrong folder answer quickly
    print("\n=== working out what each movie needs ===", flush=True)
    work = Work(settings, roots, args)
    print(f"{len(work.movie_dirs)} movie(s) in {len(set(work.groups.values()))} experiment(s)")
    received = work.declarations()
    print(f"{received} declaration file(s) found on the server; experiments without one keep "
          f"what their movies' previous analysis recorded")
    print(f"analysis code {work.code_fp} (commit {work.commit})\n")
    work.look()
    for line in movie_survey.summarise(work.rows):
        print(line)

    allowed = set(args.only or movie_survey.STAGES)
    todo = [stage_name for stage_name in movie_survey.STAGES
            if stage_name in allowed and work.needing(stage_name)]
    if args.check:
        print("\nThis was the check only; nothing was changed.")
        return 0
    if not todo:
        print("\nNothing to do: every movie is up to date.")
        return 0
    if not may_upload(settings):
        blocked = [s for s in todo if s in ('realign', 'render')]
        if blocked:
            print(f"\n{', '.join(blocked)} runs on the lab cluster, and this account may not "
                  f"write there, so only the pipeline's owner can do it. The list above is what "
                  f"to send on.")
            todo = [s for s in todo if s not in blocked]
            if not todo:
                return 0

    total = len(todo) + 1 + (1 if upload_step else 0)
    number = 0
    rows = []

    if 'realign' in todo:
        number += 1
        do_round(work, 'realign', work.needing('realign'), number, total)
        work.look()          # the cascade, asked rather than assumed

    if 'reanalyse' in todo:
        number += 1
        stale = [row['movie_dir'] for row in work.needing('reanalyse')]
        jobs = auto_jobs(settings)
        stage(number, total, f"re-analysing {len(stale)} movie(s), {jobs} at a time")
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        opts = dict(archive=True, with_video=False, force_video=False, pert_source='auto',
                    path_maps=work.path_maps, allow_no_trigger=False, only_stale=True,
                    code_fp=work.code_fp, commit=work.commit, dataset_roots=work.roots_for_data)
        rows = work.rm.run_movies(stale, stamp, opts, jobs) if stale else []
        for row in rows:
            row['experiment'] = work.groups.get(row['movie_dir'], row.get('experiment', ''))
        report = work.rm.write_report(rows, os.path.join(REPORTS_DIR,
                                                         f'reanalyse_report_{stamp}.csv'))
        print(f"\nreport: {report}")
        work.rm.print_run_summary(rows)
        work.look()

    if 'render' in todo:
        number += 1
        do_round(work, 'video', work.needing('render'), number, total)
        work.look()

    number += 1
    stage(number, total, "collecting the analysis h5 files")
    collector = work.collector
    excluded = () if args.include_bad else collector.DEFAULT_EXCLUDE_DIRS
    ready = [row['movie_dir'] for row in work.rows if 'reanalyse' not in row['needs']]
    bad = [d for d in ready if excluded and is_bad(d, work.movie_root[d], excluded)]
    ready = [d for d in ready if d not in bad]
    collected = collect_movies(settings, ready, work.movie_root, work.groups, collector)
    print(f"{len(collected)} file(s) into {collected_root(settings)}: "
          f"{count(r['status'] for r in collected) or 'none'}")
    if bad:
        print(f"not collected, being in {'/'.join(sorted(excluded))} folders: {len(bad)} movie(s)")
    not_ready = [row for row in work.rows if 'reanalyse' in row['needs']]
    if not_ready:
        print(f"not collected, because they are still not up to date: {len(not_ready)} movie(s)")

    if upload_step:
        number += 1
        stage(number, total, "uploading to the server")
        _, results = upload(settings)
        if results:
            print(f"server: {count(results.values())}")

    minutes = (time.time() - started) / 60
    print(f"\n=== finished in {minutes:.1f} min ===")
    print(f"did              : {', '.join(todo)}")
    if rows:
        print(f"re-analysed      : {count(r.get('status') for r in rows)}")
    print(f"log of this run  : {log_path}")
    print(f"collected on PC  : {collected_root(settings)}")
    if upload_step:
        print(f"on the server    : {settings['server_host']}:{upload_destination(settings)}")
    elif not may_upload(settings):
        print("uploads are off for this PC; the collected files stay here")
    if not_ready:
        print("\nSome movies are still not up to date -- see the messages above.")
    return 1 if not_ready else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('setup', help='one-time setup of this PC')
    run_parser = sub.add_parser(
        'run', help='work out what each movie needs -- realigning, re-analysing, rendering -- '
                    'and do only that, then collect and upload')
    run_parser.add_argument('folders', nargs='*',
                            help='one experiment folder, or a folder of experiments (asked if omitted)')
    run_parser.add_argument('--check', action='store_true',
                            help='say what each movie needs and stop; nothing is changed')
    run_parser.add_argument('--only', action='append', choices=movie_survey.STAGES,
                            help='do just this stage, even if others are needed (repeatable)')
    run_parser.add_argument('--render-unknown', action='store_true',
                            help='also render the videos of movies made before videos were '
                                 'stamped, which nothing on disk can judge either way')
    run_parser.add_argument('--retry-blocked', action='store_true',
                            help='offer again the movies a realignment round was refused')
    run_parser.add_argument('--at-once', type=int, default=20,
                            help='how many movies the cluster works on at a time (default 20)')
    run_parser.add_argument('--poll-seconds', type=int, default=300,
                            help='how often to ask the cluster how a round is going (default 300; '
                                 'each question is itself a small job on a node)')
    run_parser.add_argument('--restart', action='store_true',
                            help='start a new round instead of continuing the last unfinished one')
    run_parser.add_argument('--keep-on-server', action='store_true',
                            help='leave a round\'s uploaded files and results on the cluster')
    run_parser.add_argument('--no-upload', action='store_true',
                            help='re-analyse and collect on this PC only')
    run_parser.add_argument('--no-update', action='store_true',
                            help='run with this copy of the code even if the server has newer')
    run_parser.add_argument('--include-bad', action='store_true',
                            help='also collect and upload movies from bad_signal/bad_wings '
                                 'folders (they are always re-analysed; by default they are '
                                 'not collected). Each goes under its experiment\'s <bad '
                                 'folder>/ subfolder, never among its usable movies')
    sub.add_parser('update', help='download the latest committed code from the server')
    import local_predict
    local_predict.add_parser(sub)
    args = parser.parse_args()
    handlers = {'setup': setup, 'run': run, 'update': update, 'predict': local_predict.predict}
    if args.command not in handlers:
        parser.print_help()
        return 2
    try:
        return handlers[args.command](args)
    except Problem as e:
        print(f"\nSTOPPED: {e}", flush=True)
        return 1
    except KeyboardInterrupt:
        print("\nSTOPPED by the user. Running the same command again continues where it stopped.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
