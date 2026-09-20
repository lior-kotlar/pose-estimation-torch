"""Re-analyse predicted movies on a PC, then collect and upload their analysis h5 files.

The plain-language guide is LOCAL_REANALYSIS.md; this is the program behind the three
double-click files in local_reanalysis/:

    python code/local_reanalysis.py setup             once per PC: server username, ssh key, check
    python code/local_reanalysis.py run [FOLDER ...]   re-analyse, collect, upload
    python code/local_reanalysis.py update            download the latest committed code

A run takes one or more folders -- a single experiment, or a folder holding many -- and:

  1. finds every predicted movie in them (a folder with points_3D_smoothed_ensemble_best_method.npy)
  2. downloads, from the cluster, the live perturbation.json of every experiment they belong to
  3. checks every movie: where its trigger and declaration come from, and whether it is stale
  4. re-analyses the stale ones in parallel with reanalyse_movies.py; a re-run resumes
  5. collects the analysis h5 of every up-to-date movie into collected_h5/<experiment>/
  6. uploads whatever the cluster does not have yet; the cluster checks every file's checksum
     before installing it, and keeps the version it replaces in superseded_<time>/

Everything it talks to the cluster about goes through ssh to code/local_reanalysis_server.py in
the cluster's copy of the project, one connection per step. Movies of experiments the cluster
does not know are fine: they keep the declaration their old h5 recorded, and are collected under
the experiment their own records name, or local_only/<folder> when they name none.
"""
import argparse
import datetime as dt
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
    # empty: collected_h5 next to the code on this PC
    'collected_h5': '',
    # 0: half the processor threads, one movie each
    'jobs': 0,
}
SERVER_HELPER = 'code/local_reanalysis_server.py'
UPLOAD_LEDGER = '.uploaded.json'


class Problem(Exception):
    """Something the user has to fix; printed without a traceback."""


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
                      "and the local collection are kept; run again to retry the upload")
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
    upload_step = not args.no_upload
    total = 6 if upload_step else 5

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
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = rm.write_report(rows, os.path.join(REPORTS_DIR, f'reanalyse_report_{stamp}.csv'))
    failed_any = rm.print_run_summary(rows)

    stage(5, total, "collecting the analysis h5 files")
    excluded = collector.DEFAULT_EXCLUDE_DIRS
    ready = [r['movie_dir'] for r in rows if r.get('status') in ('done', 'current')]
    bad = [d for d in ready if is_bad(d, movie_root[d], excluded)]
    ready = [d for d in ready if d not in bad]
    collected = collect_movies(settings, ready, movie_root, groups, collector)
    print(f"{len(collected)} file(s) into {collected_root(settings)}: "
          f"{count(r['status'] for r in collected) or 'none'}")
    if bad:
        print(f"not collected, being in {'/'.join(sorted(excluded))} folders: {len(bad)} movie(s)")
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
    print(f"collected on PC  : {collected_root(settings)}")
    if upload_step:
        print(f"on the server    : {settings['server_host']}:{upload_destination(settings)}")
    if not_ready:
        print("\nSome movies FAILED -- see the messages above and the report.")
    return 1 if failed_any or not_ready else 0


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
    sub.add_parser('update', help='download the latest committed code from the server')
    args = parser.parse_args()
    handlers = {'setup': setup, 'run': run, 'update': update}
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
