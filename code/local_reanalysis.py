"""Re-analyse predicted movies on a PC, then collect and upload their analysis h5 files.

The plain-language guide is LOCAL_REANALYSIS.md; this is the program behind the three
double-click files in local_reanalysis/:

    python code/local_reanalysis.py setup                 once per PC: server username, ssh key
    python code/local_reanalysis.py run [FOLDER ...]      re-analyse, collect, upload
    python code/local_reanalysis.py realign [FOLDER ...]  re-run wrong-handed ensembles, then run
    python code/local_reanalysis.py update                download the latest committed code

A run takes one or more folders -- a single experiment, or a folder holding many -- and:

  1. finds every predicted movie in them (a folder with points_3D_smoothed_ensemble_best_method.npy)
  2. downloads, from the cluster, the live perturbation.json of every experiment they belong to
  3. checks every movie: where its trigger and declaration come from, and whether it is stale
  4. re-analyses the stale ones in parallel with reanalyse_movies.py; a re-run resumes
  5. collects the analysis h5 of every up-to-date movie into collected_h5/<experiment>/
  6. uploads whatever the cluster does not have yet; the cluster checks every file's checksum
     before installing it, and keeps the version it replaces in superseded_<time>/

realign is for the defect underneath the analysis, which re-analysing cannot fix: in some movies
an ensemble member labelled the two wings the other way round, and the ensemble mixed them into
one physical wing. It asks each movie whether that happened, sends only the ensemble members of
the ones where it did (10-20 MB a movie), has the cluster re-run their ensembles and keep the
result only where nothing got worse, brings the new 3D points back, and re-analyses them here.
It needs an account that may write to the cluster, so only the pipeline's owner can run it.

Everything it talks to the cluster about goes through ssh to code/local_reanalysis_server.py in
the cluster's copy of the project, one connection per step. Movies of experiments the cluster
does not know are fine: they keep the declaration their old h5 recorded, and are collected under
the experiment their own records name, or local_only/<folder> when they name none.
"""
import argparse
import datetime as dt
import glob
import hashlib
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

# The folder the code was unpacked into on the PC; everything the tool keeps lives next to it.
HOME = os.path.dirname(CODE_DIR)
SETTINGS_PATH = os.path.join(HOME, 'local_reanalysis_settings.json')
DECLARATIONS_CACHE = os.path.join(HOME, 'declarations_cache')
REPORTS_DIR = os.path.join(HOME, 'reports')
BUNDLE_COMMIT = os.path.join(HOME, 'BUNDLE_COMMIT')
# What `update` downloads: the committed versions of these paths in the cluster's project.
BUNDLE_PATHS = ('code', 'local_reanalysis', 'requirements-analysis.txt', 'LOCAL_REANALYSIS.md')

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
}
SERVER_HELPER = 'code/local_reanalysis_server.py'
UPLOAD_LEDGER = '.uploaded.json'
# realignment: what a movie's ensemble is made of, and what re-running it leaves behind
REALIGN_JOBS_DIR = os.path.join(HOME, 'realign_jobs')
REALIGN_MARKER = '.realigned_ensemble.json'
BLOCKED_REPORT = '.realign_staging/BLOCKED.json'
# left in a movie the cluster refused to change, so later rounds do not re-run it for nothing
BLOCKED_MARKER = '.realign_blocked.json'
POINTS_ALL = 'points_3D_all.npy'
POINTS_SMOOTHED = 'points_3D_smoothed_ensemble_best_method.npy'
OLD_POINTS = (POINTS_SMOOTHED, 'points_3D_ensemble_best_method.npy')
MEMBER_CONFIGS = ('specific_configuration.json', 'configuration.json')
README_GLOB = ('README_mov*.txt',)
# Set for the re-run that follows an automatic update, so it cannot update again in a loop.
UPDATED_FLAG = 'POSE_REANALYSIS_JUST_UPDATED'


class Problem(Exception):
    """Something the user has to fix; printed without a traceback."""


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
        proc = remote(settings, command, stdout=subprocess.PIPE)
        out, _ = proc.communicate(timeout=120)
    except (OSError, subprocess.SubprocessError, Problem):
        return None
    if proc.returncode != 0:
        return None
    return out.decode('utf-8', errors='replace').strip().endswith('WRITABLE')


def collected_root(settings):
    return settings['collected_h5'] or os.path.join(HOME, 'collected_h5')


def remote(settings, command, batch=False, **popen_kwargs):
    """Start `command` in a shell on the cluster, over ssh."""
    if not settings.get('server_user'):
        raise Problem("no server username in the settings; run local_reanalysis\\setup.bat")
    argv = ['ssh', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ServerAliveInterval=30']
    if batch:
        argv += ['-o', 'BatchMode=yes']
    argv += [f"{settings['server_user']}@{settings['server_host']}", command]
    try:
        return subprocess.Popen(argv, **popen_kwargs)
    except FileNotFoundError:
        raise Problem("the 'ssh' command was not found. On Windows, turn on 'OpenSSH Client' in "
                      "Settings > System > Optional features, then try again")


def helper_command(settings, *args):
    helper = posixpath.join(settings['server_project'], SERVER_HELPER)
    return ' '.join(['python3', shlex.quote(helper)] + [shlex.quote(a) for a in args])


def stage(number, total, text):
    print(f"\n=== {number}/{total}  {text} ===", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


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
        command = ("umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; "
                   f"grep -qxF {shlex.quote(public)} ~/.ssh/authorized_keys || "
                   f"echo {shlex.quote(public)} >> ~/.ssh/authorized_keys")
        if remote(settings, command).wait() != 0:
            raise Problem("could not add the key; check the username and password and run setup.bat again")

    print("\nchecking the connection and the server's copy of the project ...", flush=True)
    check = remote(settings, f"test -f {shlex.quote(posixpath.join(settings['server_project'], SERVER_HELPER))}"
                             " && python3 -c 'print(\"server ok\")'")
    if check.wait() != 0:
        raise Problem(f"connected, but {SERVER_HELPER} was not found under {settings['server_project']} "
                      "or the server has no python3")
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
    proc = remote(settings, command, stdout=subprocess.PIPE)
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
        proc = remote(settings, command, stdout=subprocess.PIPE)
        out, _ = proc.communicate(timeout=120)
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


def count(items):
    tally = {}
    for item in items:
        tally[item] = tally.get(item, 0) + 1
    return ', '.join(f'{k} {v}' for k, v in sorted(tally.items()))


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


def run_steps(args, log_path):
    started = time.time()
    settings = load_settings()
    folders = list(args.folders)
    if not folders:
        answer = ask("Folder to re-analyse (one experiment, or a folder of experiments)")
        folders = [answer] if answer else []
    roots = []
    for folder in folders:
        folder = os.path.abspath(folder.strip().strip('"'))
        if not os.path.isdir(folder):
            raise Problem(f"not a folder: {folder}")
        roots.append(folder)
    if not roots:
        raise Problem("no folder given")
    upload_step = may_upload(settings) and not args.no_upload
    total = 6 if upload_step else 5

    # before anything is imported or re-analysed, so the run uses the cluster's current code
    updated = update_if_outdated(args, settings)
    if updated is not None:
        return updated

    # heavy imports only now, so setup/update and a wrong folder answer quickly
    stage(1, total, "finding predicted movies")
    import collect_analysis_h5 as collector
    import reanalyse_movies as rm

    movie_root = {}
    for root in roots:
        for movie_dir in rm.find_movie_dirs(root):
            movie_root.setdefault(movie_dir, root)
    movie_dirs = sorted(movie_root)
    if not movie_dirs:
        raise Problem(f"no predicted movies (folders with {rm.POINTS_NAME}) under: {', '.join(roots)}")
    groups = {d: collector.experiment_key(d, movie_root[d]) for d in movie_dirs}
    print(f"{len(movie_dirs)} movie(s) in {len(set(groups.values()))} experiment(s):")
    for group in sorted(set(groups.values())):
        print(f"  {group}: {sum(g == group for g in groups.values())}")

    stage(2, total, "downloading the experiments' declarations (perturbation.json) from the server")
    box_paths = sorted({b for b in (rm.recorded_source_box(d) for d in movie_dirs) if b})
    received, path_maps = fetch_declarations(settings, box_paths)
    print(f"{received} declaration file(s) found on the server; experiments without one keep "
          f"what their movies' previous analysis recorded")

    stage(3, total, "checking every movie")
    code_fp, commit = rm.code_fingerprint(), rm.git_commit()
    print(f"analysis code {code_fp} (commit {commit})")
    checks = rm.preflight_rows(movie_dirs, 'auto', path_maps, code_fp, group_of=groups.get)
    rm.print_preflight_summary(checks)

    stale = [c['movie_dir'] for c in checks if c.get('state') != 'current']
    current = [c['movie_dir'] for c in checks if c.get('state') == 'current']
    jobs = auto_jobs(settings)
    stage(4, total, f"re-analysing {len(stale)} movie(s), {jobs} at a time "
                    f"({len(current)} already up to date)")
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    opts = dict(archive=True, with_video=False, force_video=False, pert_source='auto',
                path_maps=path_maps, allow_no_trigger=False, only_stale=True,
                code_fp=code_fp, commit=commit)
    rows = rm.run_movies(stale, stamp, opts, jobs) if stale else []
    rows += [{'experiment': os.path.basename(os.path.dirname(d)), 'movie': os.path.basename(d),
              'movie_dir': d, 'status': 'current'} for d in current]
    for row in rows:
        row['experiment'] = groups[row['movie_dir']]
    report = rm.write_report(rows, os.path.join(REPORTS_DIR, f'reanalyse_report_{stamp}.csv'))
    failed_any = rm.print_run_summary(rows)

    stage(5, total, "collecting the analysis h5 files")
    excluded = () if args.include_bad else collector.DEFAULT_EXCLUDE_DIRS
    ready = [r['movie_dir'] for r in rows if r.get('status') in ('done', 'current')]
    bad = [d for d in ready if excluded and is_bad(d, movie_root[d], excluded)]
    ready = [d for d in ready if d not in bad]
    collected = collect_movies(settings, ready, movie_root, groups, collector)
    print(f"{len(collected)} file(s) into {collected_root(settings)}: "
          f"{count(r['status'] for r in collected) or 'none'}")
    if bad:
        print(f"not collected, being in {'/'.join(sorted(excluded))} folders: {len(bad)} movie(s)")
    if args.include_bad:
        from_bad = sum(1 for r in collected if collector.bad_folder(os.path.dirname(r['src'])))
        print(f"included {from_bad} movie(s) from bad_signal/bad_wings folders, each under its "
              f"experiment's own subfolder")
    not_ready = [r for r in rows if r.get('status') not in ('done', 'current')]
    if not_ready:
        print(f"not collected, because they failed: {len(not_ready)} movie(s) (listed above)")

    upload_results = {}
    if upload_step:
        stage(6, total, "uploading to the server")
        _, upload_results = upload(settings)
        if upload_results:
            print(f"server: {count(upload_results.values())}")

    minutes = (time.time() - started) / 60
    print(f"\n=== finished in {minutes:.1f} min ===")
    print(f"re-analysed      : {count(r.get('status') for r in rows)}")
    print(f"report           : {report}")
    print(f"log of this run  : {log_path}")
    print(f"collected on PC  : {collected_root(settings)}")
    if upload_step:
        print(f"on the server    : {settings['server_host']}:{upload_destination(settings)}")
    elif not may_upload(settings):
        print("uploads are off for this PC; the collected files stay here")
    if not_ready:
        print("\nSome movies FAILED -- see the messages above and the report.")
    return 1 if failed_any or not_ready else 0


# -- realignment: one round of the cluster's ensemble re-run, driven from here -------------------

def ensemble_members(movie_dir):
    """The member folders of a movie that hold 3D candidates, in the ensemble's own order.

    Archived and hidden folders are passed over, so an earlier realignment's superseded_ensemble_*
    or .realign_staging can never be taken for a member."""
    return sorted(os.path.join(movie_dir, name) for name in os.listdir(movie_dir)
                  if not (name.startswith('superseded_') or name.startswith('.'))
                  and os.path.isfile(os.path.join(movie_dir, name, POINTS_ALL)))


def already_realigned(movie_dir):
    return os.path.isfile(os.path.join(movie_dir, REALIGN_MARKER))


def screen_movies(movie_dirs, retry_blocked=False):
    """Which movies' ensembles would change if they were re-run, and by how much.

    Reads each movie's members and asks wing_labels.harmonize_wing_labels whether any candidate
    has its wings the other way round. Numpy alone, so it needs no pose estimator on this PC.

    A movie already realigned is passed over, and so is one a round has already tried and been
    refused: re-running it would cost the same half hour and be refused again."""
    from wing_labels import harmonize_wing_labels
    import numpy as np

    flagged, skipped, unreadable, blocked = [], 0, [], 0
    for number, movie_dir in enumerate(movie_dirs, 1):
        if number % 25 == 0 or number == len(movie_dirs):
            print(f"  checked {number}/{len(movie_dirs)}", flush=True)
        if already_realigned(movie_dir):
            skipped += 1
            continue
        if not retry_blocked and os.path.isfile(os.path.join(movie_dir, BLOCKED_MARKER)):
            blocked += 1
            continue
        members = ensemble_members(movie_dir)
        if len(members) < 2:
            skipped += 1
            continue
        try:
            points = [np.load(os.path.join(d, POINTS_ALL)) for d in members]
            _, exchanged = harmonize_wing_labels(points)
        except (OSError, ValueError) as e:
            unreadable.append((movie_dir, f'{type(e).__name__}: {e}'))
            continue
        if exchanged:
            flagged.append({'movie_dir': movie_dir, 'exchanged_pairs': int(exchanged),
                            'members': len(members)})
    return flagged, skipped, unreadable, blocked


def job_state_path(job):
    return os.path.join(REALIGN_JOBS_DIR, f'{job}.json')


def save_job_state(state):
    os.makedirs(REALIGN_JOBS_DIR, exist_ok=True)
    path = job_state_path(state['job'])
    staged = path + '.partial'
    with open(staged, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=1)
    os.replace(staged, path)


def unfinished_job(folders):
    """The newest round over these same folders that has not been finished off yet."""
    if not os.path.isdir(REALIGN_JOBS_DIR):
        return None
    for name in sorted(os.listdir(REALIGN_JOBS_DIR), reverse=True):
        if not name.endswith('.json'):
            continue
        try:
            with open(os.path.join(REALIGN_JOBS_DIR, name), encoding='utf-8') as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not state.get('finished') and state.get('folders') == folders:
            return state
    return None


def new_job_state(folders, movies):
    # the round is named after this PC, so several PCs' rounds never share a folder on the cluster
    pc = os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME') or 'pc'
    name = ''.join(c if c.isalnum() else '_' for c in pc)[:24]
    job = f"{name or 'pc'}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return {'job': job, 'created': dt.datetime.now().isoformat(timespec='seconds'),
            'folders': folders, 'movies': movies, 'uploaded': False, 'job_id': '',
            'installed': {}, 'finished': False}


def movie_keys(flagged, groups):
    """A short name per movie for the job tree: <experiment>/<movie>, kept unique."""
    keys, used = {}, set()
    for entry in sorted(flagged, key=lambda e: e['movie_dir']):
        movie_dir = entry['movie_dir']
        base = f"{groups[movie_dir]}/{os.path.basename(movie_dir)}"
        key, number = base, 1
        while key in used:
            number += 1
            key = f"{base}_{number}"
        used.add(key)
        keys[key] = movie_dir
    return keys


def realign_inputs(movie_dir):
    """The files the cluster needs to re-run this movie's ensemble, relative to the movie.

    Each model's 3D candidates and the config that names it, plus the movie's current ensemble
    points, which are the 'before' side of the comparison the cluster makes. Nothing else: not
    the source movie, not the analysis h5, not the video."""
    names = []
    for member in ensemble_members(movie_dir):
        base = os.path.basename(member)
        names.append(f'{base}/{POINTS_ALL}')
        for config in MEMBER_CONFIGS:
            if os.path.isfile(os.path.join(member, config)):
                names.append(f'{base}/{config}')
    names += [name for name in OLD_POINTS if os.path.isfile(os.path.join(movie_dir, name))]
    for pattern in README_GLOB:
        for path in sorted(glob.glob(os.path.join(glob.escape(movie_dir), pattern))):
            names.append(os.path.basename(path))
    return names


def upload_members(settings, state):
    """Send every flagged movie's ensemble members to its folder of the job on the cluster."""
    destination = posixpath.join(settings['server_project'], 'realign_jobs', state['job'], 'inputs')
    pending = {}
    for key, movie_dir in sorted(state['movies'].items()):
        for name in realign_inputs(movie_dir):
            source = os.path.join(movie_dir, *name.split('/'))
            pending[f'{key}/{name}'] = (source, sha256(source))
    size = sum(os.path.getsize(source) for source, _ in pending.values())
    print(f"sending {len(pending)} file(s), {size / 1e6:.0f} MB, to "
          f"{settings['server_host']}:{destination}", flush=True)
    proc = remote(settings, helper_command(settings, 'receive', '--dest', destination),
                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin, mode='w|') as tar:
            manifest = json.dumps({'files': {rel: digest
                                             for rel, (_, digest) in pending.items()}}).encode('utf-8')
            info = tarfile.TarInfo('MANIFEST.json')
            info.size = len(manifest)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(manifest))
            for number, rel in enumerate(sorted(pending), 1):
                tar.add(pending[rel][0], arcname=rel, recursive=False)
                if number % 50 == 0 or number == len(pending):
                    print(f"  sent {number}/{len(pending)}", flush=True)
        proc.stdin.close()
    except (BrokenPipeError, OSError) as e:
        output = proc.stdout.read().decode('utf-8', errors='replace')
        proc.wait()
        raise Problem(f"the upload was cut off ({e}). {output.strip()} -- nothing on this PC was "
                      "touched; run the same command again to start over")
    answer = server_answer(proc, 'the server did not accept the ensemble members')
    print(f"the server has every file: {count(answer['results'].values())}")


def server_answer(proc, what):
    """The one JSON line a helper verb prints, or a Problem naming what went wrong."""
    output = proc.stdout.read().decode('utf-8', errors='replace').strip()
    returncode = proc.wait()
    try:
        answer = json.loads(output.splitlines()[-1])
    except (IndexError, ValueError):
        answer = {'ok': False, 'error': output or f'no answer from the server (exit {returncode})'}
    if not answer.get('ok'):
        raise Problem(f"{what}: {answer.get('error')}")
    return answer


def helper_json(settings, *args, what='the server could not do that'):
    proc = remote(settings, helper_command(settings, *args), stdout=subprocess.PIPE)
    return server_answer(proc, what)


def wait_for_job(settings, state, poll_seconds):
    """Watch the array job until every movie has been realigned, blocked or given up on."""
    job = state['job']
    started, last, shown, warned, settled = time.time(), None, 0, False, False
    while True:
        answer = helper_json(settings, 'realign-status', '--job', job,
                             what='the server could not say how the realignment is going')
        states = answer.get('states') or {}
        tally = count(states.values()) or 'none'
        pending = [k for k, v in states.items() if v == 'pending']
        queued = answer.get('slurm')
        minutes = (time.time() - started) / 60
        queue = (', slurm: ' + ', '.join(f'{state} {number}'
                                         for state, number in sorted(queued.items()))
                 if queued else '')
        # a line whenever anything moves -- a movie finishing, or the queue letting one start --
        # and a heartbeat now and then, so a long wait never looks like a hung window
        if (tally, queue) != last or minutes - shown >= 10:
            print(f"  [{minutes:5.1f} min] {tally}{queue}", flush=True)
            last, shown = (tally, queue), minutes
        if not pending:
            return states
        if answer.get('working') is False:
            # slurm drops a task from its books the moment it ends, which can be before its last
            # file is there to be seen; give the results one more poll before giving up on them
            if not settled:
                settled = True
                time.sleep(min(poll_seconds, 30))
                continue
            print(f"\n{len(pending)} movie(s) finished without a result. The cluster keeps each "
                  f"task's messages in logs/realign_{job}_*.out", flush=True)
            for key in pending[:10]:
                print(f"  no result: {key}")
            return states
        settled = False
        if answer.get('working') is None and not warned:
            print("  (the cluster's queue could not be asked; watching for the results "
                  "themselves instead. Close the window if this never moves)", flush=True)
            warned = True
        time.sleep(poll_seconds)


def safe_key(relpath, movies):
    """True when a downloaded name belongs to a movie of this round and stays inside it."""
    parts = relpath.split('/')
    if not relpath or relpath.startswith('/') or '..' in parts or not all(parts):
        return False
    return any(relpath.startswith(key + '/') for key in movies)


def fetch_results(settings, state):
    """Download the new ensembles into this PC's staging folder; returns the files' relative paths."""
    staging = os.path.join(REALIGN_JOBS_DIR, state['job'], 'incoming')
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    proc = remote(settings, helper_command(settings, 'realign-fetch', '--job', state['job']),
                  stdout=subprocess.PIPE)
    manifest, received = None, []
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|gz') as tar:
            for member in tar:
                if member.name == 'MANIFEST.json':
                    manifest = json.loads(tar.extractfile(member).read().decode('utf-8'))['files']
                    continue
                if not member.isfile() or manifest is None or member.name not in manifest:
                    continue
                if not safe_key(member.name, state['movies']):
                    raise Problem(f"the server offered a file this PC did not ask for "
                                  f"({member.name!r}); nothing was touched")
                target = os.path.join(staging, *member.name.split('/'))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with tar.extractfile(member) as source, open(target, 'wb') as out:
                    shutil.copyfileobj(source, out)
                received.append(member.name)
    except tarfile.TarError as e:
        proc.wait()
        raise Problem(f"the download was cut off ({e}); nothing on this PC was touched. Run the "
                      "same command again to retry it")
    if proc.wait() != 0 or manifest is None:
        raise Problem("the server did not send the realigned files; nothing on this PC was "
                      "touched. Run the same command again to retry it")
    for rel, digest in manifest.items():
        path = os.path.join(staging, *rel.split('/'))
        if not os.path.isfile(path):
            raise Problem(f"{rel} is missing from the download; nothing on this PC was touched")
        if sha256(path) != digest:
            raise Problem(f"{rel} arrived damaged; nothing on this PC was touched. Run the same "
                          "command again to download it afresh")
    print(f"{len(received)} file(s) downloaded and checked")
    return staging, sorted(manifest)


def install_results(state, staging, relpaths):
    """Put each movie's new ensemble in place, keeping the one it replaces beside it.

    Each movie is recorded as soon as it is done, so a round taken up again installs only what
    it had not installed yet, instead of archiving a movie's files a second time."""
    import numpy as np

    by_movie = {}
    for rel in relpaths:
        for key in state['movies']:
            if rel.startswith(key + '/'):
                by_movie.setdefault(key, []).append(rel[len(key) + 1:])
                break
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    installed = dict(state.get('installed') or {})
    for key in sorted(by_movie):
        if installed.get(key):
            print(f"  {key}: already done earlier in this round")
            continue
        movie_dir = state['movies'][key]
        names = by_movie[key]
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
            installed[key] = 'blocked'
            state['installed'] = installed
            save_job_state(state)
            continue

        new_points = os.path.join(source, POINTS_SMOOTHED)
        old_points = os.path.join(movie_dir, POINTS_SMOOTHED)
        if os.path.isfile(new_points) and os.path.isfile(old_points):
            if np.load(new_points).shape != np.load(old_points).shape:
                print(f"  {key}: LEFT ALONE, the new points have a different shape from this "
                      f"movie's own; nothing was replaced")
                installed[key] = 'mismatch'
                state['installed'] = installed
                save_job_state(state)
                continue

        archive = os.path.join(movie_dir, f'superseded_ensemble_{stamp}')
        os.makedirs(archive, exist_ok=True)
        moved = []
        # the marker last, so an interrupted install leaves a movie that is redone, not one that
        # looks finished
        for name in sorted(n for n in names if n != REALIGN_MARKER):
            target = os.path.join(movie_dir, *name.split('/'))
            top = name.split('/')[0]
            existing = os.path.join(movie_dir, top)
            if os.path.exists(existing) and top not in moved:
                shutil.move(existing, os.path.join(archive, top))
                moved.append(top)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.move(os.path.join(source, *name.split('/')), target)
        marker = json.load(open(os.path.join(source, REALIGN_MARKER), encoding='utf-8'))
        marker.update(movie=movie_dir, installed_from=state['job'],
                      installed_at=dt.datetime.now().isoformat(timespec='seconds'))
        with open(os.path.join(movie_dir, REALIGN_MARKER), 'w', encoding='utf-8') as f:
            json.dump(marker, f, indent=1)
        if os.path.isfile(os.path.join(movie_dir, BLOCKED_MARKER)):
            os.remove(os.path.join(movie_dir, BLOCKED_MARKER))
        collapse = marker.get('collapse_pct') or []
        change = (f", frames with both wings on one wing {collapse[0]:.1f}% -> {collapse[1]:.1f}%"
                  if len(collapse) == 2 else '')
        print(f"  {key}: new ensemble in place{change}; what it replaced is in "
              f"{os.path.basename(archive)}/")
        installed[key] = 'realigned'
        state['installed'] = installed
        save_job_state(state)
    return installed


def realign(args):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    log_path = os.path.join(REPORTS_DIR, f"realign_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_path, 'w', encoding='utf-8') as handle:
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(out, handle), Tee(err, handle)
        try:
            return realign_steps(args, log_path)
        except Problem as e:
            print(f"\nSTOPPED: {e}", flush=True)
            print(f"log of this round: {log_path}", flush=True)
            return 1
        finally:
            sys.stdout, sys.stderr = out, err


def realign_steps(args, log_path):
    started = time.time()
    settings = load_settings()
    # the check itself reads nothing but this PC's own disk, so anyone may run it; repairing a
    # movie computes on the cluster and writes there, which only the pipeline's owner may do
    check_only = args.check_only or not may_upload(settings)
    folders = list(args.folders)
    if not folders:
        answer = ask("Folder to realign (one experiment, or a folder of experiments)")
        folders = [answer] if answer else []
    roots = []
    for folder in folders:
        folder = os.path.abspath(folder.strip().strip('"'))
        if not os.path.isdir(folder):
            raise Problem(f"not a folder: {folder}")
        roots.append(folder)
    if not roots:
        raise Problem("no folder given")

    updated = update_if_outdated(args, settings)
    if updated is not None:
        return updated

    total = 1 if check_only else (6 if args.no_reanalyse else 7)
    state = None if (args.restart or check_only) else unfinished_job(roots)
    if state:
        print(f"\ncontinuing the round started {state['created']} ({len(state['movies'])} movie(s), "
              f"job {state['job']})")

    if state is None:
        stage(1, total, "checking which movies' ensembles would change")
        import collect_analysis_h5 as collector
        import reanalyse_movies as rm
        movie_root = {}
        for root in roots:
            for movie_dir in rm.find_movie_dirs(root):
                movie_root.setdefault(movie_dir, root)
        movie_dirs = sorted(movie_root)
        if not movie_dirs:
            raise Problem(f"no predicted movies (folders with {rm.POINTS_NAME}) under: "
                          f"{', '.join(roots)}")
        print(f"{len(movie_dirs)} movie(s) to check; this reads each one's ensemble members")
        flagged, skipped, unreadable, blocked = screen_movies(movie_dirs, args.retry_blocked)
        for movie_dir, problem in unreadable:
            print(f"  could not read {os.path.basename(movie_dir)}: {problem}")
        unchanged = len(movie_dirs) - len(flagged) - len(unreadable) - blocked
        print(f"\n{len(flagged)} movie(s) would change, {unchanged} would come out exactly as "
              f"they are ({skipped} of them already realigned or without members to compare)")
        if blocked:
            print(f"{blocked} more were tried in an earlier round and refused, so they are left "
                  f"out; --retry-blocked offers them again")
        for entry in sorted(flagged, key=lambda e: -e['exchanged_pairs'])[:10]:
            print(f"  {os.path.basename(entry['movie_dir']):<34} "
                  f"{entry['exchanged_pairs']} (frame, candidate) pair(s) the other way round")
        if len(flagged) > 10:
            print(f"  ... and {len(flagged) - 10} more")
        if not flagged:
            print("\nNothing to realign: " + ("the only movies that would change were tried "
                                               "before and refused" if blocked else
                                               "every movie's ensemble already has its wings "
                                               "the same way round") + ".")
            return 0
        if check_only:
            if args.check_only:
                print("\nThis was the check only. Run it without --check-only to repair them.")
            else:
                print("\nRepairing these movies runs on the lab cluster, and this account may "
                      "not write there, so only the pipeline's owner can do it. Send the list "
                      "above to Lior. Nothing has left this PC, and nothing here was changed.")
            return 0
        groups = {e['movie_dir']: collector.experiment_key(e['movie_dir'], movie_root[e['movie_dir']])
                  for e in flagged}
        state = new_job_state(roots, movie_keys(flagged, groups))
        save_job_state(state)

    if not state['uploaded']:
        stage(2, total, f"sending {len(state['movies'])} movie(s) to the cluster")
        upload_members(settings, state)
        state['uploaded'] = True
        save_job_state(state)
    else:
        stage(2, total, "the cluster already has these movies")

    if not state['job_id']:
        stage(3, total, "asking the cluster to re-run their ensembles")
        answer = helper_json(settings, 'realign-submit', '--job', state['job'],
                             '--throttle', str(args.at_once),
                             what='the cluster would not start the realignment')
        state['job_id'] = str(answer['job_id'])
        save_job_state(state)
        print(f"slurm job {state['job_id']}, {answer['movies']} task(s), "
              f"at most {args.at_once} at a time")
    else:
        stage(3, total, f"the cluster is already running this round (slurm job {state['job_id']})")

    stage(4, total, "waiting for the cluster")
    print("Each movie takes about half an hour, longer for a long one. You can close this "
          "window and run the same command later to pick it up again.")
    wait_for_job(settings, state, args.poll_seconds)

    stage(5, total, "downloading the new ensembles and putting them in place")
    staging, relpaths = fetch_results(settings, state)
    installed = install_results(state, staging, relpaths)
    state['installed'] = installed
    save_job_state(state)
    shutil.rmtree(staging, ignore_errors=True)
    realigned = [k for k, v in installed.items() if v == 'realigned']
    print(f"\n{len(realigned)} movie(s) realigned, {len(installed) - len(realigned)} left alone")

    stage(6, total, "clearing the round off the cluster")
    if args.keep_on_server:
        print(f"kept, as asked: {settings['server_project']}/realign_jobs/{state['job']}")
    else:
        helper_json(settings, 'realign-clean', '--job', state['job'],
                    what='the cluster could not delete this round')
        print("the uploaded members and their results are gone from the cluster; this PC keeps "
              "both the new files and the ones they replaced")
    state['finished'] = True
    save_job_state(state)

    minutes = (time.time() - started) / 60
    print(f"\n=== realignment finished in {minutes:.1f} min ===")
    print(f"log of this round: {log_path}")
    if not realigned:
        print("No movie changed, so there is nothing to re-analyse.")
        return 0
    if args.no_reanalyse:
        print("\nThe realigned movies now need re-analysing: run reanalyse.bat over the same "
              "folder when you are ready.")
        return 0

    stage(7, total, "re-analysing the realigned movies, and collecting them")
    print("A realigned movie counts as stale, so this redoes exactly those, plus any other "
          "movie whose products are out of date.\n")
    nested = argparse.Namespace(folders=[str(r) for r in roots], no_upload=args.no_upload,
                                no_update=True, include_bad=args.include_bad)
    return run_steps(nested, log_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('setup', help='one-time setup of this PC')
    run_parser = sub.add_parser('run', help='re-analyse, collect and upload')
    run_parser.add_argument('folders', nargs='*',
                            help='one experiment folder, or a folder of experiments (asked if omitted)')
    run_parser.add_argument('--no-upload', action='store_true',
                            help='re-analyse and collect on this PC only')
    run_parser.add_argument('--no-update', action='store_true',
                            help='run with this copy of the code even if the server has newer')
    run_parser.add_argument('--include-bad', action='store_true',
                            help='also collect and upload movies from bad_signal/bad_wings '
                                 'folders (they are always re-analysed; by default they are '
                                 'not collected). Each goes under its experiment\'s <bad '
                                 'folder>/ subfolder, never among its usable movies')
    realign_parser = sub.add_parser(
        'realign', help="re-run the ensemble of movies whose wings are labelled inconsistently, "
                        "on the cluster, then re-analyse them here")
    realign_parser.add_argument('folders', nargs='*',
                                help='one experiment folder, or a folder of experiments (asked if omitted)')
    realign_parser.add_argument('--check-only', action='store_true',
                                help='only say which movies would change; touch nothing')
    realign_parser.add_argument('--retry-blocked', action='store_true',
                                help='offer again the movies an earlier round was refused')
    realign_parser.add_argument('--at-once', type=int, default=20,
                                help='how many movies the cluster works on at a time (default 20)')
    realign_parser.add_argument('--poll-seconds', type=int, default=120,
                                help='how often to ask the cluster how it is going (default 120)')
    realign_parser.add_argument('--restart', action='store_true',
                                help='start a new round instead of continuing the last unfinished one')
    realign_parser.add_argument('--keep-on-server', action='store_true',
                                help='leave the uploaded members and results on the cluster')
    realign_parser.add_argument('--no-reanalyse', action='store_true',
                                help='install the new ensembles but do not re-analyse them yet')
    realign_parser.add_argument('--no-upload', action='store_true',
                                help='re-analyse and collect on this PC only')
    realign_parser.add_argument('--no-update', action='store_true',
                                help='run with this copy of the code even if the server has newer')
    realign_parser.add_argument('--include-bad', action='store_true',
                                help='also collect movies from bad_signal/bad_wings folders')
    sub.add_parser('update', help='download the latest committed code from the server')
    args = parser.parse_args()
    handlers = {'setup': setup, 'run': run, 'update': update, 'realign': realign}
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
