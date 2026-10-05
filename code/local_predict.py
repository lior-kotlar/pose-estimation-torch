"""Prep and predict, on the cluster, an experiment whose raw movies are on this PC -- and bring
the results home. The plain-language guide is LOCAL_PREDICT.md.

Drop an experiment folder onto local_reanalysis\\predict.bat (python code/local_reanalysis.py
predict FOLDER) and this, with no further input unless the data leaves a real choice:

  1. looks at the movies here (code/predict_prep.py): the experiment's units, prep's prescan of
     every movie, and -- the first time only -- the easyWand, mirror camera and run name, which
     it works out from the mats where it can and saves as prep.json beside the data;
  2. leaves out what needs nothing: movies the prescan turns away, movies already predicted into
     the output folder, and movies the cluster turned away before under the same prep.json;
  3. sends the rest a few GB at a time, every movie cut down to the frames prep reads
     (code/sparse_trim.py); each unit's prep starts on the cluster as soon as it is up;
  4. watches the cluster -- prep (code/process_experiment.py via pipeline.sh, one prep at a time),
     then one GPU task per movie -- retrying by itself a task that dies, and a prep that
     crashes, once;
  5. brings each movie home as soon as it is done: the prediction folder to
     <output>/<run>/<movie>/, and the render-only box, raw movie and prescan sidecar beside its
     mats; the unit's calibration.h5 and process report beside its movies. Then it deletes that
     movie from the cluster, so a long round never holds the lab's disk;
  6. deletes the round from the cluster, and writes a report (reports\\predict_report_*.csv).

If the cluster's disk cannot take every movie at once, the round takes the ones that fit and the
next round starts by itself when it is done. Closing the window is safe at any point: the round
is written down as it goes (realign_jobs\\predict_*.json) and the same command picks it up.
"""
import csv
import datetime as dt
import json
import os
import posixpath
import shutil
import sys
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import cluster_round as rounds  # noqa: E402
import local_reanalysis as lr  # noqa: E402
import predict_prep as pp  # noqa: E402
from cluster_link import Problem, count, helper_json, stage  # noqa: E402

KIND = 'predict'
JOBS_DIR = lr.REALIGN_JOBS_DIR
PRESCAN_CACHE = os.path.join(lr.HOME, '.prescan_cache.json')
# left in a local movie folder once the cluster has had its say, so a movie prep turned away is
# not sent again under the same prep.json
OUTCOME_FILE = '.predict_outcome.json'
ANALYSIS_SUFFIX = '_analysis_smoothed.h5'
RENDER_SUFFIX = '_render.h5'
RAW_MOVIE_GLOB = '*_raw_fr*_skip*.mp4'
# what a movie's state on the server means for this PC
WAITING = ('uploaded', 'waiting', 'queued', 'prepping', 'predicting', 'missing')
FINAL = ('predicted', 'rejected', 'prep_failed', 'prep_stopped', 'failed', 'prep_crashed',
         'released', 'lost', 'not_sent')
TURNED_AWAY = ('rejected', 'prep_failed', 'prep_stopped', 'failed', 'prep_crashed', 'lost',
               'not_sent')
RETRIES = 1
# rough sizes, to plan with before anything is measured
SERVER_MB_PER_FRAME = 0.45    # box, predictions, render box, and the box cache while predicting
HOME_MB = (40.0, 0.12)        # what comes back: per movie, plus per frame
PRESCAN_FLAGS = {'min_intersection': '--prescan-min-intersection',
                 'pixel_threshold': '--prescan-pixel-threshold',
                 'blob_ratio': '--prescan-blob-ratio',
                 'blob_distance': '--prescan-blob-distance',
                 'min_edge_margin': '--prescan-min-edge-margin',
                 'min_cams_in_frame': '--prescan-min-cams-in-frame'}


def out_root_of(settings, args):
    root = args.out or (settings.get('predict_output') or '').strip()
    return os.path.abspath(root or os.path.join(lr.HOME, 'predict_output'))


def job_path(settings, job):
    return posixpath.join(settings['server_project'], rounds.JOBS_DIRNAME, job)


# -- what is already here -------------------------------------------------------------------------

def predicted_index(out_root, depth=5):
    """{prediction folder name: [folders]} for every predicted movie under the output folder,
    including any sorted into bad_signal/ and the like."""
    index = {}
    if not os.path.isdir(out_root):
        return index
    base_depth = out_root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(out_root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(('superseded_', '.')))
        name = os.path.basename(dirpath)
        if name + ANALYSIS_SUFFIX in filenames:
            index.setdefault(name, []).append(dirpath)
            dirnames[:] = []
        elif dirpath.count(os.sep) - base_depth >= depth:
            dirnames[:] = []
    return index


def already_predicted(index, stem, experiment, run_name):
    """The folder this movie was already predicted into, or None. A movie name is not unique
    across experiments, so the folder must also record this experiment (or, recording none, sit
    in this run's folder)."""
    from collect_analysis_h5 import normalise_experiment, recorded_experiment
    wanted = normalise_experiment(experiment)
    for folder in index.get(stem, []):
        recorded = recorded_experiment(folder)
        if recorded == wanted:
            return folder
        if recorded is None and run_name in folder.replace('\\', '/').split('/'):
            return folder
    return None


def run_folder(out_root, run_name):
    """Where a run's movies go: an existing folder of that name near the top of the output
    folder (e.g. <output>/roni_dark/roni_dark_2023_08_07_5ms), else <output>/<run>."""
    import glob
    for depth_glob in (run_name, os.path.join('*', run_name), os.path.join('*', '*', run_name)):
        found = sorted(p for p in glob.glob(os.path.join(glob.escape(out_root), depth_glob))
                       if os.path.isdir(p))
        if found:
            return found[0]
    return os.path.join(out_root, run_name)


def existing_run(index, experiment):
    """The run folder this experiment's movies were predicted into before, if any are in the
    output folder -- so new movies of it land beside them, under the name already in use."""
    from collect_analysis_h5 import WORKFLOW_DIRS, normalise_experiment, recorded_experiment
    wanted = normalise_experiment(experiment)
    for folders in index.values():
        for folder in folders:
            if recorded_experiment(folder) != wanted:
                continue
            parent = os.path.dirname(folder)
            # movies sorted into bad_signal/<reason>/ and the like still belong to the run above
            while (os.path.basename(parent) in WORKFLOW_DIRS
                   or os.path.basename(os.path.dirname(parent)) in WORKFLOW_DIRS):
                parent = os.path.dirname(parent)
            return parent
    return None


def read_outcome(movie_dir):
    try:
        with open(os.path.join(movie_dir, OUTCOME_FILE), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_outcome(movie_dir, record):
    try:
        with open(os.path.join(movie_dir, OUTCOME_FILE), 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=1)
    except OSError:
        pass


# -- the survey -----------------------------------------------------------------------------------

def camera_count(movies, declared):
    """(cameras, {movie: why it is left out}) -- prep's rule: 3 or 4 cameras are recognised, 2
    only when declared, and an experiment's movies must agree."""
    counts = {m: len(pp.sparse_mats(m)) for m in movies}
    if declared:
        n = int(declared)
    else:
        usual = {c for c in counts.values() if c in (3, 4)}
        if len(usual) > 1:
            raise Problem(f"the movies under {os.path.dirname(movies[0])} hold both 3 and 4 "
                          f"cameras' mats; prep needs one camera count per experiment")
        n = usual.pop() if usual else (2 if set(counts.values()) == {2} else 0)
    left_out = {m: f"has {c} camera mats, the experiment {n}" for m, c in counts.items()
                if c != n}
    return n, left_out


def survey(settings, roots, args, out_root, interactive):
    """Every movie under the folders, and what it needs. Asks only what the data cannot settle,
    the first time an experiment is seen."""
    from scan_sparse_movies import PRESCAN_DEFAULTS
    units = []
    for root in roots:
        units += pp.find_units(root)
    if not units:
        raise Problem(f"no movie folders (mov<N> holding *_sparse.mat files) under "
                      f"{', '.join(roots)}")
    by_experiment = {}
    for unit_dir, movies in units:
        by_experiment.setdefault(pp.experiment_dir(unit_dir), []).append((unit_dir, movies))
    cache = pp.PrescanCache(PRESCAN_CACHE)
    workers = max(1, min(4, int(settings.get('jobs') or 0) or (os.cpu_count() or 2) // 2))
    index = predicted_index(out_root)
    dataset_root = lr.dataset_roots(settings)
    plan = {'experiments': {}, 'movies': []}
    for exp_dir, exp_units in sorted(by_experiment.items()):
        declaration = None if args.redeclare else pp.load_declaration(exp_dir)
        params = dict(PRESCAN_DEFAULTS)
        params.update((declaration or {}).get('prescan') or {})
        all_movies = [m for _, movies in exp_units for m in movies]
        n_cams, left_out = camera_count(all_movies, (declaration or {}).get('num_cams'))
        print(f"\n{exp_dir}: {len(all_movies)} movie(s) in {len(exp_units)} folder(s), "
              f"{n_cams} cameras", flush=True)
        scans = pp.prescan_movies([m for m in all_movies if m not in left_out], params, cache,
                                  workers)
        first_parts = pp.staged_path(exp_units[0][0], dataset_root)
        new = declaration is None
        if new:
            declaration = pp.declare(
                exp_dir, [(u, [m for m in ms if m not in left_out]) for u, ms in exp_units],
                scans, interactive, lr.ask, n_cams)
        if new:
            before = existing_run(index, pp.experiment_name(first_parts))
            if before:
                declaration['run_name'] = os.path.basename(before)
                print(f"run name: {declaration['run_name']} (this experiment's movies already "
                      f"predicted in {before})")
        declaration['run_name'] = pp.settle_run_name(declaration, first_parts, interactive,
                                                     lr.ask)
        declaration['experiment'] = pp.experiment_name(first_parts)
        if new or args.redeclare:
            pp.save_declaration(exp_dir, declaration)
        easywand = os.path.normpath(os.path.join(exp_dir, declaration['easywand']))
        if not os.path.isfile(easywand):
            raise Problem(f"{exp_dir}\\{pp.PREP_FILE} names the easyWand {declaration['easywand']}"
                          f", which is not there")
        signature = pp.declaration_hash(declaration)
        run_name = declaration['run_name']
        install_root = run_folder(out_root, run_name)
        plan['experiments'][exp_dir] = {'declaration': declaration, 'easywand': easywand,
                                        'signature': signature, 'params': params,
                                        'run_name': run_name, 'install_root': install_root}
        for unit_dir, movies in exp_units:
            parts = pp.staged_path(unit_dir, dataset_root)
            for movie_dir in movies:
                scan = scans.get(movie_dir)
                row = {'dir': movie_dir, 'unit_dir': unit_dir, 'exp_dir': exp_dir,
                       'parts': parts, 'name': f"mov{pp.movie_number(movie_dir)}",
                       'scan': scan, 'stem': None, 'need': 'send', 'reason': ''}
                plan['movies'].append(row)
                if movie_dir in left_out:
                    row.update(need='skip', reason=left_out[movie_dir])
                    continue
                if scan['verdict'] != 'OK':
                    row.update(need='bad', reason=(scan.get('error') or
                                                   f"prescan: only {scan['good_run_length']} "
                                                   f"usable frames in a row"))
                    continue
                row['stem'] = pp.box_stem(movie_dir, scan)
                done = already_predicted(index, row['stem'], declaration['experiment'],
                                         run_name)
                if done and not args.repredict:
                    row.update(need='predicted', reason=done)
                    continue
                outcome = read_outcome(movie_dir)
                if (outcome and outcome.get('state') in TURNED_AWAY and not args.retry_failed
                        and outcome.get('declaration') == signature):
                    row.update(need='turned_away',
                               reason=f"{outcome['state']}: {outcome.get('reason', '')}")
    return plan


def mats_bytes(movie_dir):
    return sum(os.path.getsize(p) for p in pp.sparse_mats(movie_dir))


def estimate_upload(row):
    """Bytes the movie will take on the wire: the kept share of its frames' pixels, roughly."""
    scan = row['scan']
    span = (scan['good_end'] - scan['good_start'] + 14 + 100) / max(scan['n_frames'], 1)
    return mats_bytes(row['dir']) * min(1.0, 0.25 + span)


def frames_of(row):
    scan = row['scan'] or {}
    return max(0, scan.get('good_end', 0) - scan.get('good_start', 0))


def print_plan(plan):
    print("\n=== what each movie needs ===")
    rows = {}
    for row in plan['movies']:
        key = os.path.relpath(row['unit_dir'], os.path.dirname(row['exp_dir']))
        rows.setdefault(key, {}).setdefault(row['need'], []).append(row)
    print(f"{'folder':44} {'movies':>6} {'send':>5} {'done':>5} {'bad':>4} {'away':>5} {'skip':>5}")
    for key, needs in sorted(rows.items()):
        total = sum(len(v) for v in needs.values())
        print(f"{key[-44:]:44} {total:6} {len(needs.get('send', [])):5} "
              f"{len(needs.get('predicted', [])):5} {len(needs.get('bad', [])):4} "
              f"{len(needs.get('turned_away', [])):5} {len(needs.get('skip', [])):5}")
    send = [r for r in plan['movies'] if r['need'] == 'send']
    for row in plan['movies']:
        if row['need'] in ('bad', 'skip', 'turned_away'):
            print(f"  {row['need']:11} {os.path.basename(row['unit_dir'])}/{row['name']}: "
                  f"{row['reason']}")
    if send:
        up = sum(estimate_upload(r) for r in send) / 1e9
        home = sum(HOME_MB[0] + HOME_MB[1] * frames_of(r) for r in send) / 1e3
        print(f"\nto predict: {len(send)} movie(s); about {up:.1f} GB to send and "
              f"{home:.1f} GB to bring back")
    else:
        print("\nnothing to predict: every movie is either predicted already or turned away")
    print("('done' = already predicted in the output folder; 'away' = the cluster turned it "
          "away before under this prep.json -- --retry-failed sends it again)")


# -- the round ------------------------------------------------------------------------------------

def new_round(settings, plan, args, out_root, budget_bytes):
    """A round over the movies that need sending, as many as the cluster's disk can take."""
    pc = os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME') or 'pc'
    name = ''.join(c if c.isalnum() else '_' for c in pc)[:24] or 'pc'
    state = {'job': f"{KIND}_{name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
             'kind': KIND, 'created': dt.datetime.now().isoformat(timespec='seconds'),
             'folders': args.roots, 'out_root': out_root, 'finished': False,
             'deferred': 0, 'units': {}, 'movies': {}}
    used, unit_of = 0, {}
    for row in plan['movies']:
        if row['need'] != 'send':
            continue
        cost = estimate_upload(row) + SERVER_MB_PER_FRAME * 1e6 * frames_of(row)
        if state['movies'] and used + cost > budget_bytes:
            state['deferred'] += 1
            continue
        used += cost
        exp = plan['experiments'][row['exp_dir']]
        if row['unit_dir'] not in unit_of:
            unit = f"u{len(unit_of) + 1}"
            unit_of[row['unit_dir']] = unit
            declaration = exp['declaration']
            prep_args = list(declaration.get('prep_args') or [])
            from scan_sparse_movies import PRESCAN_DEFAULTS
            for key, flag in PRESCAN_FLAGS.items():
                if exp['params'].get(key) != PRESCAN_DEFAULTS.get(key):
                    prep_args += [flag, str(exp['params'][key])]
            state['units'][unit] = {
                'local': row['unit_dir'], 'exp_dir': row['exp_dir'], 'parts': row['parts'],
                'run_name': exp['run_name'], 'install_root': exp['install_root'],
                'easywand': exp['easywand'], 'cam': declaration['cam'], 'prep_args': prep_args,
                'params': exp['params'], 'signature': exp['signature'],
                'skip_raw_movies': False, 'spec_sent': False, 'extras_sent': False,
                'prep_job_id': '', 'unit_installed': False, 'resets': 0}
        unit = unit_of[row['unit_dir']]
        key = f"{unit}/{row['name']}"
        state['movies'][key] = {'dir': row['dir'], 'unit': unit, 'name': row['name'],
                                'stem': row['stem'], 'scan': row['scan'],
                                'frames': frames_of(row), 'uploaded': False, 'state': '',
                                'reason': '', 'retries': 0, 'installed': '', 'released': False}
    for unit, info in state['units'].items():
        movies = [m['dir'] for m in state['movies'].values() if m['unit'] == unit]
        # prep makes a raw movie only where there is none; when every movie here already has
        # one, the step (6 min a movie of a serialised prep) is skipped outright
        info['skip_raw_movies'] = all(_raw_movie(d) for d in movies)
    return state


def _raw_movie(movie_dir):
    import glob
    found = glob.glob(os.path.join(glob.escape(movie_dir), RAW_MOVIE_GLOB))
    return found[0] if found else None


def save(state):
    rounds.save_job_state(JOBS_DIR, state)


def unit_spec(state, unit):
    info = state['units'][unit]
    folder = 'inference_datasets/' + '/'.join(info['parts'])
    movies = {m['name']: {'stem': m['stem'], 'frames': m['frames']}
              for m in state['movies'].values() if m['unit'] == unit and m['uploaded']}
    prep_args = list(info['prep_args'])
    if info['skip_raw_movies']:
        prep_args.append('--skip-raw-movies')
    return {'unit': unit, 'input': folder,
            'easywand': f"{folder}/{os.path.basename(info['easywand'])}",
            'cam': info['cam'], 'run_name': info['run_name'], 'prep_args': prep_args,
            'predict_config': os.path.basename(state.get('predict_config') or 'config1.json'),
            'movies': movies}


def unit_extras(state, unit, outgoing):
    """The unit's easyWand and declarations, as {name on the cluster: path here}. A
    perturbation.json above a batch folder is sent into the batch: predict reads it only from
    a movie's folder or the one above."""
    info = state['units'][unit]
    folder = 'inference_datasets/' + '/'.join(info['parts'])
    files = {f"{folder}/{os.path.basename(info['easywand'])}": info['easywand']}
    for candidate in (os.path.join(info['local'], 'perturbation.json'),
                      os.path.join(info['exp_dir'], 'perturbation.json')):
        if os.path.isfile(candidate):
            if os.path.dirname(candidate) != info['local']:
                print(f"  {unit}: using the experiment's perturbation.json for this batch folder")
            files[f"{folder}/perturbation.json"] = candidate
            break
    return files


def defer_rest(state, keys):
    """Leave movies out of this round (the cluster's disk is short); the next round takes them."""
    for key in keys:
        state['movies'].pop(key, None)
        state['deferred'] = state.get('deferred', 0) + 1
    for unit in [u for u in state['units']
                 if not any(m['unit'] == u for m in state['movies'].values())]:
        state['units'].pop(unit)


def send_round(settings, state, args):
    """Upload every movie not yet on the cluster, a chunk at a time, and start each unit's prep as
    soon as all of it is there."""
    import sparse_trim
    chunk_limit = float(settings.get('upload_chunk_mb') or 2000) * 1e6
    reserve = float(settings.get('server_reserve_gb') or 30) * 1e9
    destination = job_path(settings, state['job'])
    outgoing = os.path.join(JOBS_DIR, state['job'], 'outgoing')
    for unit in sorted(state['units'], key=lambda u: int(u[1:])):
        info = state['units'].get(unit)
        if info is None or info['prep_job_id']:
            continue
        keys = sorted((k for k, m in state['movies'].items()
                       if m['unit'] == unit and not m['uploaded'] and m['state'] != 'not_sent'),
                      key=lambda k: int(state['movies'][k]['name'][3:]))
        files, chunk, chunk_bytes = {}, [], 0
        if not info['extras_sent']:
            files.update(unit_extras(state, unit, outgoing))

        def flush():
            nonlocal files, chunk, chunk_bytes
            if not files:
                return
            free = helper_json(settings, 'predict-space', '--job', state['job'],
                               what='the cluster could not say how much space it has')
            if free['free_bytes'] - chunk_bytes < reserve:
                print(f"  the cluster's disk is nearly full ({free['free_bytes'] / 1e9:.0f} GB "
                      f"free); the remaining movies wait for the next round", flush=True)
                return False
            rounds.send_files(settings, destination, files)
            for key in chunk:
                state['movies'][key]['uploaded'] = True
                shutil.rmtree(os.path.join(outgoing, *key.split('/')), ignore_errors=True)
            info['extras_sent'] = True
            save(state)
            files, chunk, chunk_bytes = {}, [], 0
            return True

        stopped = False
        for number, key in enumerate(keys, 1):
            movie = state['movies'][key]
            here = os.path.join(outgoing, *key.split('/'))
            print(f"  [{unit} {number}/{len(keys)}] {movie['name']}: keeping the frames prep "
                  f"reads", flush=True)
            try:
                record = sparse_trim.blank_movie(movie['dir'], here, movie['scan'],
                                                 info['params'])
            except (ValueError, OSError) as e:
                movie.update(state='not_sent', reason=f"could not be cut down: {e}")
                print(f"      NOT SENT: {e}", flush=True)
                save(state)
                continue
            before = sum(m['bytes_original'] for m in record['mats'].values())
            after = sum(m['bytes'] for m in record['mats'].values())
            print(f"      {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB", flush=True)
            folder = f"inference_datasets/{'/'.join(info['parts'])}/{movie['name']}"
            for name in sorted(os.listdir(here)):
                files[f"{folder}/{name}"] = os.path.join(here, name)
            local_declaration = os.path.join(movie['dir'], 'perturbation.json')
            if os.path.isfile(local_declaration):
                files[f"{folder}/perturbation.json"] = local_declaration
            chunk.append(key)
            chunk_bytes += after
            if chunk_bytes >= chunk_limit and flush() is False:
                stopped = True
                break
        if not stopped and flush() is False:
            stopped = True
        if stopped:
            defer_rest(state, [k for k, m in state['movies'].items()
                               if not m['uploaded'] and m['state'] != 'not_sent'])
            save(state)
            if unit not in state['units']:
                continue
        if not any(m['uploaded'] for m in state['movies'].values() if m['unit'] == unit):
            continue
        if not info['spec_sent']:
            spec_path = os.path.join(outgoing, f"{unit}.json")
            os.makedirs(outgoing, exist_ok=True)
            with open(spec_path, 'w', encoding='utf-8') as f:
                json.dump(unit_spec(state, unit), f, indent=1)
            rounds.send_files(settings, destination, {f"units/{unit}.json": spec_path})
            info['spec_sent'] = True
            save(state)
        answer = helper_json(settings, 'predict-submit', '--job', state['job'], '--unit', unit,
                             '--throttle', str(args.at_once),
                             what=f'the cluster would not start prep for {unit}')
        info['prep_job_id'] = str(answer['prep_job_id'])
        save(state)
        n = sum(1 for m in state['movies'].values() if m['unit'] == unit and m['uploaded'])
        print(f"  {unit}: prep queued on the cluster (slurm job {info['prep_job_id']}, "
              f"{n} movie(s)); preps run one at a time", flush=True)
        if stopped:
            break


# -- bringing results home ------------------------------------------------------------------------

def move_into(source, target, archive):
    """Move one file into place, keeping whatever it replaces in `archive` (unless identical)."""
    from cluster_link import sha256
    if os.path.isfile(target):
        if sha256(target) == sha256(source):
            os.remove(source)
            return 'unchanged'
        os.makedirs(archive, exist_ok=True)
        shutil.move(target, os.path.join(archive, os.path.basename(target)))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.move(source, target)
    return 'installed'


def install_movie(state, key, staging, names, server_row):
    """Put one movie's results in place on this PC; returns what was done."""
    movie = state['movies'][key]
    info = state['units'][movie['unit']]
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    source = os.path.join(staging, *key.split('/'))
    prep = [n[len('prep/'):] for n in names if n.startswith('prep/')]
    predicted = [n[len('predicted/'):] for n in names if n.startswith('predicted/')]
    archive = os.path.join(movie['dir'], f'superseded_{stamp}')
    has_raw = _raw_movie(movie['dir'])
    for name in prep:
        # a raw movie made from cut-down mats shows only the kept frames; one made here from the
        # full mats is better, so it is never replaced
        if has_raw and (name.endswith('.mp4') or name == 'raw_movie.log'):
            continue
        move_into(os.path.join(source, 'prep', name), os.path.join(movie['dir'], name), archive)
    state_name = server_row.get('state', '')
    folder = ''
    if state_name == 'predicted' and predicted:
        stem = server_row.get('stem') or movie['stem']
        folder = os.path.join(info['install_root'], stem)
        if os.path.isdir(folder):
            older = os.path.join(info['install_root'], f'superseded_{stamp}', stem)
            os.makedirs(os.path.dirname(older), exist_ok=True)
            shutil.move(folder, older)
            print(f"      the earlier prediction is kept in {older}")
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        shutil.move(os.path.join(source, 'predicted'), folder)
    write_outcome(movie['dir'], {'state': state_name, 'reason': server_row.get('reason', ''),
                                 'declaration': info['signature'], 'round': state['job'],
                                 'prediction': folder,
                                 'at': dt.datetime.now().isoformat(timespec='seconds')})
    movie.update(state=state_name, reason=server_row.get('reason', ''),
                 installed=folder or state_name, stem=server_row.get('stem') or movie['stem'])
    return folder


def install_unit(state, unit, staging, names):
    info = state['units'][unit]
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    source = os.path.join(staging, unit, 'unit')
    archive = os.path.join(info['local'], f'superseded_{stamp}')
    for name in names:
        move_into(os.path.join(source, name), os.path.join(info['local'], name), archive)
    info['unit_installed'] = True


def fetch(settings, state, keys, units, rows):
    """Download, check and install these movies and units, then free them on the cluster."""
    staging = os.path.join(JOBS_DIR, state['job'], 'incoming')
    verb = ['predict-fetch', '--job', state['job']]
    if keys:
        verb += ['--movies', ','.join(keys)]
    if units:
        verb += ['--units', ','.join(units)]
    prefixes = tuple(f"{k}/" for k in keys) + tuple(f"{u}/unit/" for u in units)
    relpaths = rounds.fetch_tar(settings, verb, staging, lambda rel: rel.startswith(prefixes))
    for key in keys:
        names = [r[len(key) + 1:] for r in relpaths if r.startswith(key + '/')]
        folder = install_movie(state, key, staging, names, rows.get(key, {}))
        movie = state['movies'][key]
        print(f"  {movie['unit']}/{movie['name']}: {movie['state']}"
              + (f" -> {folder}" if folder else f" ({movie['reason']})"), flush=True)
        save(state)
    for unit in units:
        names = [r[len(unit) + 6:] for r in relpaths if r.startswith(unit + '/unit/')]
        install_unit(state, unit, staging, names)
        print(f"  {unit}: calibration.h5 and prep's report are beside the movies in "
              f"{state['units'][unit]['local']}", flush=True)
        save(state)
    shutil.rmtree(staging, ignore_errors=True)
    if keys and not state.get('keep_on_server'):
        helper_json(settings, 'predict-release', '--job', state['job'], '--movies',
                    ','.join(keys), what='the cluster could not free the movies brought home')
        for key in keys:
            state['movies'][key]['released'] = True
        save(state)


def batches(keys, rows, limit):
    batch, size = [], 0
    for key in keys:
        cost = rows.get(key, {}).get('bytes') or 0
        if batch and size + cost > limit:
            yield batch
            batch, size = [], 0
        batch.append(key)
        size += cost
    if batch:
        yield batch


# -- watching -------------------------------------------------------------------------------------

def ensure_keeper(settings, state):
    """Make sure the round has a keeper on the cluster: a small job that retries what fails while
    this PC is off, and has slurm email the owner when the round is done (local_predict_server)."""
    verb = ['predict-keeper', '--job', state['job']]
    if not settings.get('email_when_done', True):
        verb.append('--no-mail')
    try:
        answer = helper_json(settings, *verb, what='the cluster would not start the round\'s keeper')
    except Problem as e:
        print(f"  ({e}; this PC will look after the round itself while it is on)", flush=True)
        return None
    if answer.get('not_needed'):
        return None
    state['keeper_job_id'] = str(answer['keeper_job_id'])
    save(state)
    return answer


def watch(settings, state, args):
    """Poll the cluster until every movie of the round is home, acting on what it says.

    While the round's keeper runs on the cluster, retrying failed tasks and resubmitting preps
    that never started is left to it (the server keeps the counts, so the two never act twice);
    what only this PC can do -- bring movies home, send a crashed prep's movies again -- is done
    here either way."""
    fetch_limit = float(settings.get('fetch_chunk_mb') or 2000) * 1e6
    started, last, shown, idle, trouble = time.time(), None, 0.0, 0, 0
    while True:
        try:
            answer = helper_json(settings, 'predict-status', '--job', state['job'],
                                 what='the cluster could not say how the round is going')
            trouble = 0
        except Problem as e:
            # a dropped connection or a busy gateway is no reason to stop a round of hours
            trouble += 1
            if trouble >= 6:
                raise
            print(f"  ({e}; asking again in {args.poll_seconds // 60 or 1} min)", flush=True)
            time.sleep(args.poll_seconds)
            continue
        rows = answer.get('movies') or {}
        units = answer.get('units') or {}
        keeper_active = bool((answer.get('keeper') or {}).get('active'))
        for key, movie in state['movies'].items():
            if key in rows and not movie['installed']:
                movie['state'] = rows[key]['state']
                movie['reason'] = rows[key].get('reason', '')
                movie['retries'] = rows[key].get('retries', movie.get('retries', 0))
        save(state)
        tally = count(m['state'] for m in state['movies'].values() if not m['installed'])
        done = sum(1 for m in state['movies'].values() if m['installed'])
        minutes = (time.time() - started) / 60
        line = f"{tally or 'nothing left'}; home: {done}/{len(state['movies'])}"
        if line != last or minutes - shown >= 10:
            print(f"  [{minutes:5.1f} min] {line}", flush=True)
            last, shown = line, minutes
        for unit, report in units.items():
            if report.get('stopped') and not state['units'].get(unit, {}).get('told'):
                print(f"\n  {unit}: PREP STOPPED -- {report['stopped']}\n  (fix prep.json, e.g. "
                      f"with --redeclare, and run again)", flush=True)
                state['units'][unit]['told'] = True

        def waits_for_retry(key):
            row = rows.get(key, {})
            if row.get('state') == 'failed':
                return (bool(row.get('task') or row.get('no_array'))
                        and row.get('retries', 0) < RETRIES)
            if row.get('state') == 'prep_crashed' and row.get('instant'):
                unit = key.split('/')[0]
                return units.get(unit, {}).get('prep_resubmits', 0) < 3
            return False

        acted = False
        if not keeper_active:
            # a prep that never ran (a node without the lab's disk) is simply submitted again
            lost = sorted({k.split('/')[0] for k, m in state['movies'].items()
                           if not m['installed'] and m['state'] == 'prep_crashed'
                           and waits_for_retry(k)})
            for unit in lost:
                print(f"  {unit}: the prep job never started properly (a node without the "
                      f"lab's disk); submitting it again", flush=True)
                reply = helper_json(settings, 'predict-submit', '--job', state['job'], '--unit',
                                    unit, '--throttle', str(args.at_once), '--again',
                                    what=f'the cluster would not start prep for {unit} again')
                state['units'][unit]['prep_job_id'] = str(reply['prep_job_id'])
                acted = True
            # a predict task that died: tried once more
            retry = sorted(k for k, m in state['movies'].items()
                           if not m['installed'] and m['state'] == 'failed' and waits_for_retry(k))
            if retry:
                print(f"  retrying {len(retry)} movie(s) whose GPU task failed: "
                      f"{', '.join(retry[:8])}", flush=True)
                helper_json(settings, 'predict-retry', '--job', state['job'],
                            '--movies', ','.join(retry), what='the cluster would not retry')
                acted = True
            if acted:
                save(state)
                ensure_keeper(settings, state)
                continue

        # a prep that crashed part-way: its mats may be half flipped, so the unit goes up again,
        # once -- only this PC has the movies to send
        crashed = sorted({m['unit'] for k, m in state['movies'].items()
                          if m['state'] == 'prep_crashed' and not m['installed']
                          and not rows.get(k, {}).get('instant')
                          and state['units'][m['unit']]['resets'] < RETRIES})
        for unit in crashed:
            info = state['units'][unit]
            print(f"  {unit}: prep crashed on the cluster; sending the unit again", flush=True)
            helper_json(settings, 'predict-reset', '--job', state['job'], '--unit', unit,
                        what=f'the cluster could not reset {unit}')
            info.update(resets=info['resets'] + 1, prep_job_id='', spec_sent=False,
                        extras_sent=False)
            for movie in state['movies'].values():
                if movie['unit'] == unit:
                    movie.update(uploaded=False, state='')
            save(state)
            send_round(settings, state, args)
        if crashed:
            ensure_keeper(settings, state)
            continue

        ready = sorted(k for k, m in state['movies'].items()
                       if not m['installed'] and m['state'] in FINAL
                       and m['state'] not in ('prep_crashed', 'released', 'not_sent')
                       and not waits_for_retry(k))
        for batch in batches(ready, rows, fetch_limit):
            print(f"bringing home {len(batch)} movie(s) ...", flush=True)
            try:
                fetch(settings, state, batch, [], rows)
            except Problem as e:
                print(f"  ({e}; trying again at the next look)", flush=True)
                break
        for key, movie in state['movies'].items():
            if (not movie['installed'] and movie['state'] in ('prep_crashed', 'not_sent', 'lost')
                    and not waits_for_retry(key)):
                movie['installed'] = movie['state']
                info = state['units'][movie['unit']]
                write_outcome(movie['dir'], {'state': movie['state'], 'reason': movie['reason'],
                                             'declaration': info['signature'],
                                             'round': state['job'],
                                             'at': dt.datetime.now().isoformat(timespec='seconds')})
        finished_units = [u for u, info in state['units'].items()
                          if not info['unit_installed'] and info['prep_job_id']
                          and all(m['installed'] for m in state['movies'].values()
                                  if m['unit'] == u)]
        if finished_units:
            try:
                fetch(settings, state, [], finished_units, rows)
            except Problem as e:
                print(f"  ({e}; trying again at the next look)", flush=True)
        save(state)
        if all(m['installed'] for m in state['movies'].values()) and \
                all(i['unit_installed'] or not i['prep_job_id'] for i in state['units'].values()):
            return
        # nothing running on the cluster, no keeper, and nothing moving: those movies are not
        # coming
        waiting = [k for k, m in state['movies'].items() if not m['installed']]
        idle = (idle + 1 if answer.get('working') is False and not keeper_active and waiting
                else 0)
        if idle >= 3:
            print(f"\n{len(waiting)} movie(s) have nothing running for them on the cluster; "
                  f"giving up on them", flush=True)
            for key in waiting:
                state['movies'][key].update(state='lost', reason='nothing ran for it on the '
                                                                 'cluster')
            save(state)
            continue
        time.sleep(args.poll_seconds)


# -- the command ----------------------------------------------------------------------------------

def report(state, out_root):
    os.makedirs(lr.REPORTS_DIR, exist_ok=True)
    path = os.path.join(lr.REPORTS_DIR,
                        f"predict_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['movie', 'folder', 'state', 'reason', 'prediction', 'round'])
        for key, movie in sorted(state['movies'].items()):
            writer.writerow([movie['name'], movie['dir'], movie['state'], movie['reason'],
                             movie['installed'] if movie['state'] == 'predicted' else '',
                             state['job']])
    return path


def do_round(settings, state, args):
    pending = sum(1 for m in state['movies'].values()
                  if not m['uploaded'] and m['state'] != 'not_sent')
    if pending or not all(u['prep_job_id'] for u in state['units'].values()):
        stage(2, 4, f"sending {pending} movie(s) to the cluster (round {state['job']})")
        send_round(settings, state, args)
    else:
        stage(2, 4, f"the cluster already has every movie of this round ({state['job']})")
    stage(3, 4, "prep, then prediction, on the cluster -- each movie comes home as soon as it "
                "is done")
    keeper = ensure_keeper(settings, state)
    print("Prep takes a few minutes a movie, one experiment at a time; a GPU task about 20 "
          "minutes to 3 hours a movie, many at once.", flush=True)
    if keeper:
        mail = (" and slurm emails you when it is done (\"predictions_<run>_ready\")"
                if settings.get('email_when_done', True) else "")
        print(f"\nEVERYTHING IS ON THE CLUSTER. You may close this window and turn this PC off: "
              f"the cluster looks after the round by itself (slurm job {keeper['keeper_job_id']})"
              f"{mail}.\nAfterwards drag the same folder onto predict.bat again to bring the "
              f"results home.\n", flush=True)
    else:
        print("You can close this window and run the same command later to pick the round up "
              "again.", flush=True)
    watch(settings, state, args)
    stage(4, 4, "clearing the round off the cluster")
    if args.keep_on_server:
        print(f"kept, as asked: {job_path(settings, state['job'])}")
    else:
        helper_json(settings, 'predict-clean', '--job', state['job'],
                    what='the cluster could not delete the round')
        print("the round is gone from the cluster")
    state['finished'] = True
    save(state)
    shutil.rmtree(os.path.join(JOBS_DIR, state['job']), ignore_errors=True)
    tally = count(m['state'] for m in state['movies'].values())
    print(f"\nround {state['job']}: {tally}")
    print(f"report: {report(state, state['out_root'])}")
    for movie in state['movies'].values():
        if movie['state'] != 'predicted':
            print(f"  {movie['unit']}/{movie['name']} ({os.path.basename(movie['dir'])}): "
                  f"{movie['state']} -- {movie['reason']}")


def give_up(settings, roots):
    state = rounds.unfinished_job(JOBS_DIR, roots, KIND)
    if not state:
        print("there is no unfinished predict round over these folders")
        return 0
    helper_json(settings, 'predict-clean', '--job', state['job'],
                what='the cluster could not cancel the round')
    state['finished'] = True
    state['given_up'] = True
    save(state)
    print(f"round {state['job']} cancelled and deleted from the cluster; movies brought home "
          f"so far stay")
    return 0


def predict_steps(args, log_path):
    settings = lr.load_settings()
    roots = lr.wanted_folders(args, what='predict')
    args.roots = roots
    rerun = lr.update_if_outdated(args, settings)
    if rerun is not None:
        return rerun
    if args.give_up:
        return give_up(settings, roots)
    out_root = out_root_of(settings, args)
    interactive = sys.stdin.isatty() and not args.yes
    print(f"predictions go to: {out_root}")
    while True:
        state = None if args.restart else rounds.unfinished_job(JOBS_DIR, roots, KIND)
        if state:
            print(f"\ncontinuing the predict round started {state['created']} "
                  f"({len(state['movies'])} movie(s), job {state['job']})")
            state['keep_on_server'] = args.keep_on_server
            do_round(settings, state, args)
        else:
            stage(1, 4, "looking at the movies on this PC")
            plan = survey(settings, roots, args, out_root, interactive)
            print_plan(plan)
            if args.check or not any(r['need'] == 'send' for r in plan['movies']):
                return 0
            if not lr.may_upload(settings):
                raise Problem("predicting runs jobs on the cluster in the pipeline owner's "
                              "account, and this PC is set up without it (setup found it cannot "
                              "write there); predict_check.bat still works")
            free = helper_json(settings, 'predict-space', '--job', 'predict_survey_check',
                               what='the cluster could not say how much space it has')
            reserve = float(settings.get('server_reserve_gb') or 30) * 1e9
            budget = free['free_bytes'] - reserve
            if budget <= 0:
                raise Problem(f"the cluster's disk has only {free['free_bytes'] / 1e9:.0f} GB "
                              f"free (this PC keeps {reserve / 1e9:.0f} GB of it for others); "
                              f"try again later")
            state = new_round(settings, plan, args, out_root, budget)
            state['predict_config'] = settings.get('predict_config') or 'config1.json'
            state['keep_on_server'] = args.keep_on_server
            save(state)
            if state['deferred']:
                print(f"the cluster's disk takes {len(state['movies'])} movie(s) now; the other "
                      f"{state['deferred']} follow in the next round, by themselves")
            do_round(settings, state, args)
            if not state.get('deferred'):
                return 0
        if args.restart:
            args.restart = False


def predict(args):
    os.makedirs(lr.REPORTS_DIR, exist_ok=True)
    log_path = os.path.join(lr.REPORTS_DIR,
                            f"predict_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_path, 'w', encoding='utf-8') as handle:
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = lr.Tee(out, handle), lr.Tee(err, handle)
        try:
            code = predict_steps(args, log_path)
            print(f"\nlog of this run: {log_path}", flush=True)
            return code
        except Problem as e:
            print(f"\nSTOPPED: {e}", flush=True)
            print(f"log of this run: {log_path}", flush=True)
            return 1
        finally:
            sys.stdout, sys.stderr = out, err


def add_parser(sub):
    parser = sub.add_parser(
        'predict', help='prep and predict, on the cluster, the experiments in a folder on this PC, '
                        'and bring the results home')
    parser.add_argument('folders', nargs='*',
                        help='an experiment folder, a batch folder, a movie folder, or a folder of '
                             'experiments (asked if omitted)')
    parser.add_argument('--out', default='',
                        help='where predictions go (default: the predict_output setting, else '
                             'predict_output next to the tool)')
    parser.add_argument('--check', action='store_true',
                        help='only say what would be sent; nothing is uploaded')
    parser.add_argument('--redeclare', action='store_true',
                        help='work out prep.json again (easyWand, mirror camera, run name)')
    parser.add_argument('--repredict', action='store_true',
                        help='also send movies already predicted in the output folder (the old '
                             'prediction is kept in superseded_<time>/)')
    parser.add_argument('--retry-failed', action='store_true',
                        help='also send movies the cluster turned away before under the same '
                             'prep.json')
    parser.add_argument('--yes', action='store_true',
                        help='never ask; stop where the data cannot decide')
    parser.add_argument('--at-once', type=int, default=32,
                        help='how many GPU tasks may run at a time (default 32)')
    parser.add_argument('--poll-seconds', type=int, default=300,
                        help='how often to ask the cluster how the round is going (default 300)')
    parser.add_argument('--restart', action='store_true',
                        help='start a new round instead of continuing the unfinished one')
    parser.add_argument('--give-up', action='store_true',
                        help='cancel the unfinished round over these folders and delete it from '
                             'the cluster')
    parser.add_argument('--keep-on-server', action='store_true',
                        help="leave the round's files on the cluster (for debugging)")
    parser.add_argument('--no-update', action='store_true',
                        help='run with this copy of the code even if the server has newer')
    return parser
