"""One round of work the cluster does for a PC: send, run, watch, bring home.

Two jobs have the same shape and differ only in what travels and what comes back:

    realign  the movie's ensemble members go up, the cluster re-runs the ensemble and keeps the
             result only where nothing got worse, and new 3D points come back
    video    a shrunk box h5 and the analysis h5 go up, the cluster reprojects and renders, and
             the overlay mp4 comes back

So the state machine lives here once -- find the movies, upload, submit an array job, watch it,
fetch, install beside the movie, clear the round off the cluster -- and each kind supplies only
what is particular to it. A round is written down as it goes, so closing the window and running
the same command again picks it up rather than starting over.
"""
import datetime as dt
import io
import json
import os
import posixpath
import shutil
import subprocess
import tarfile
import time

from cluster_link import (Problem, count, helper_command, helper_json, remote, server_answer,
                          sha256)

# where a PC writes down the rounds it has going
JOBS_DIRNAME = 'realign_jobs'


def job_state_path(jobs_dir, job):
    return os.path.join(jobs_dir, f'{job}.json')


def save_job_state(jobs_dir, state):
    os.makedirs(jobs_dir, exist_ok=True)
    path = job_state_path(jobs_dir, state['job'])
    staged = path + '.partial'
    with open(staged, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=1)
    os.replace(staged, path)


def unfinished_job(jobs_dir, folders, kind):
    """The newest round of this kind over these same folders that has not been finished yet.

    Keyed on the kind as well as the folders, so a half-done realignment and a half-done render
    over the same experiment are two rounds and never each other's."""
    if not os.path.isdir(jobs_dir):
        return None
    for name in sorted(os.listdir(jobs_dir), reverse=True):
        if not name.endswith('.json'):
            continue
        try:
            with open(os.path.join(jobs_dir, name), encoding='utf-8') as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if (not state.get('finished') and state.get('folders') == folders
                and state.get('kind', 'realign') == kind):
            return state
    return None


def new_job_state(folders, movies, kind):
    # the round is named after this PC, so several PCs' rounds never share a folder on the cluster
    pc = os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME') or 'pc'
    name = ''.join(c if c.isalnum() else '_' for c in pc)[:24]
    job = f"{name or 'pc'}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return {'job': f'{kind}_{job}', 'kind': kind,
            'created': dt.datetime.now().isoformat(timespec='seconds'),
            'folders': folders, 'movies': movies, 'uploaded': False, 'job_id': '',
            'installed': {}, 'finished': False}


def movie_keys(rows):
    """A short name per movie for the job tree: <experiment>/<movie>, kept unique.

    Takes survey rows, which already carry the experiment a movie belongs to."""
    keys, used = {}, set()
    for entry in sorted(rows, key=lambda e: e['movie_dir']):
        movie_dir = entry['movie_dir']
        base = f"{entry['group']}/{os.path.basename(movie_dir)}"
        key, number = base, 1
        while key in used:
            number += 1
            key = f"{base}_{number}"
        used.add(key)
        keys[key] = movie_dir
    return keys


def upload_round(settings, state, files_for):
    """Send each movie's inputs to its folder of the job on the cluster.

    files_for(movie_dir) yields (name under the movie, path on this PC) -- the kind decides what
    that is: a realignment sends the ensemble members, a render sends a shrunk box and the h5."""
    destination = job_inputs(settings, state['job'])
    pending = {}
    for key, movie_dir in sorted(state['movies'].items()):
        for name, source in files_for(movie_dir):
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
    answer = server_answer(proc, 'the server did not accept the files')
    print(f"the server has every file: {count(answer['results'].values())}")


def wait_for_job(settings, state, poll_seconds):
    """Watch the array job until every movie has a result, or the queue says none is coming."""
    job = state['job']
    started, last, shown, warned, settled = time.time(), None, 0, False, False
    while True:
        answer = helper_json(settings, 'round-status', '--job', job, '--kind', state['kind'],
                             what='the server could not say how the round is going')
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


def fetch_results(settings, jobs_dir, state):
    """Download what the cluster made into a staging folder here; returns the relative paths."""
    staging = os.path.join(jobs_dir, state['job'], 'incoming')
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    proc = remote(settings, helper_command(settings, 'round-fetch', '--job', state['job'],
                                           '--kind', state['kind']), stdout=subprocess.PIPE)
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
        raise Problem("the server did not send the results; nothing on this PC was touched. Run "
                      "the same command again to retry it")
    for rel, digest in manifest.items():
        path = os.path.join(staging, *rel.split('/'))
        if not os.path.isfile(path):
            raise Problem(f"{rel} is missing from the download; nothing on this PC was touched")
        if sha256(path) != digest:
            raise Problem(f"{rel} arrived damaged; nothing on this PC was touched. Run the same "
                          "command again to download it afresh")
    print(f"{len(received)} file(s) downloaded and checked")
    return staging, sorted(manifest)


def job_inputs(settings, job):
    """Where a round's uploaded files live on the cluster."""
    return posixpath.join(settings['server_project'], JOBS_DIRNAME, job, 'inputs')


def submit(settings, state, at_once):
    """Ask the cluster to start the array job for this round; returns its slurm id."""
    answer = helper_json(settings, 'round-submit', '--job', state['job'], '--kind', state['kind'],
                         '--throttle', str(at_once),
                         what='the cluster would not start this round')
    return str(answer['job_id']), int(answer.get('movies', 0))


def clean(settings, state):
    """Delete the round from the cluster, once its results are safely home."""
    helper_json(settings, 'round-clean', '--job', state['job'], '--kind', state['kind'],
                what='the cluster could not delete this round')


def by_movie(state, relpaths):
    """The downloaded files grouped under the movie of this round they belong to."""
    grouped = {}
    for rel in relpaths:
        for key in state['movies']:
            if rel.startswith(key + '/'):
                grouped.setdefault(key, []).append(rel[len(key) + 1:])
                break
    return grouped


def record(jobs_dir, state, key, outcome):
    """Write down what became of one movie, as soon as it is known.

    A round taken up again then installs only what it had not installed yet, rather than archiving
    a movie's files a second time."""
    installed = dict(state.get('installed') or {})
    installed[key] = outcome
    state['installed'] = installed
    save_job_state(jobs_dir, state)
