"""What a PC works out about an experiment before sending it to be predicted, with no cluster.

code/local_predict.py runs this first. It reads only the raw `*_sparse.mat` files on this PC:

  units       the folders prep runs on: any folder holding mov<N>/ folders of sparse mats (an
              experiment, or one of its batch folders such as 1to30)
  prescan     prep's own prescan (scan_sparse_movies.scan_movie, with prep's own thresholds) on
              every movie -- which movies prep will turn away, and the frames it will build from
              the rest. Cached, so a second run over the same folder costs nothing
  declaration what prep needs to be told about an experiment: which easyWand, which camera films
              through the mirror, the run name. Saved as prep.json in the experiment folder the
              first time and reused from then on. The data decides wherever it can: each easyWand
              found near the experiment is tried with prep's mirror check on the raw mats, so a
              wrong calibration or a wrong mirror camera shows up here, in seconds, before
              anything is uploaded. Only what the data cannot settle is asked.

Where a unit goes on the cluster is decided here too: the part of its path below the PC's
dataset_root (or below an `inference_datasets` folder), so the predictions' provenance names the
same experiment a manual cluster run would, and the reanalysis tool finds the data again.
"""
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from cluster_link import Problem  # noqa: E402

PREP_FILE = 'prep.json'
SCHEMA = 1
MOVIE_DIR = re.compile(r'^mov(\d+)$', re.IGNORECASE)
BATCH_DIR = re.compile(r'^\d+to\d+$')
# a cluster path component: MATLAB and the predict scripts quote paths with ', so keep them plain
SAFE_PART = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
DATED = re.compile(r'\d{4}_\d{2}_\d{2}')
DATASETS_SEGMENT = 'inference_datasets'
LOCAL_ONLY = 'local_only'
SPARSE_GLOB = '*_sparse.mat'
# an easyWand whose mirror check leaves every camera this close to the calibration is one prep
# would pass (verify fails a movie above 15 px)
GOOD_FIT_PX = 15.0
# two calibrations that both fit, and are this close, are a choice for a person, not the data
CLEAR_WIN = 1.5
SKIP_DIRS = ('superseded_', '.', '_superseded')


# -- finding units and movies ---------------------------------------------------------------------

def sparse_mats(movie_dir):
    return sorted(glob.glob(os.path.join(glob.escape(movie_dir), SPARSE_GLOB)))


def is_movie_dir(path):
    return bool(MOVIE_DIR.match(os.path.basename(path))) and len(sparse_mats(path)) >= 2


def movie_number(path):
    m = MOVIE_DIR.match(os.path.basename(path))
    return int(m.group(1)) if m else None


def find_units(root):
    """[(unit folder, [movie folders])] under `root`: every folder holding mov<N>/ folders of
    sparse mats. A movie folder dropped on its own is a unit of one, in its parent."""
    root = os.path.abspath(root)
    if is_movie_dir(root):
        return [(os.path.dirname(root), [root])]
    units = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(SKIP_DIRS))
        movies = [os.path.join(dirpath, d) for d in dirnames
                  if is_movie_dir(os.path.join(dirpath, d))]
        if movies:
            units.append((dirpath, sorted(movies, key=movie_number)))
            dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in movies]
    return units


def experiment_dir(unit_dir):
    """The experiment a unit belongs to: the unit itself, or the folder above a batch folder."""
    return os.path.dirname(unit_dir) if BATCH_DIR.match(os.path.basename(unit_dir)) else unit_dir


def staged_path(unit_dir, dataset_root):
    """The unit's path on the cluster below inference_datasets/, as a list of components.

    Below dataset_root when the unit is there; below the last `inference_datasets` folder of its
    path otherwise; local_only/<experiment>[/<batch>] for a folder that is in neither."""
    unit_dir = os.path.abspath(unit_dir)
    parts = None
    if dataset_root:
        root = os.path.abspath(dataset_root)
        rel = os.path.relpath(unit_dir, root)
        if not rel.startswith('..') and rel != '.':
            parts = rel.replace('\\', '/').split('/')
    if parts is None:
        pieces = unit_dir.replace('\\', '/').split('/')
        lowered = [p.lower() for p in pieces]
        if DATASETS_SEGMENT in lowered:
            after = len(lowered) - 1 - lowered[::-1].index(DATASETS_SEGMENT)
            parts = [p for p in pieces[after + 1:] if p] or None
    if parts is None:
        exp = experiment_dir(unit_dir)
        parts = [LOCAL_ONLY, os.path.basename(exp)]
        if exp != unit_dir:
            parts.append(os.path.basename(unit_dir))
    bad = [p for p in parts if not SAFE_PART.match(p)]
    if bad:
        raise Problem(f"the folder name {bad[0]!r} (in {unit_dir}) holds characters the cluster "
                      f"tools cannot take; rename it to letters, digits, '.', '-' and '_' only")
    return parts


def experiment_name(parts):
    """roni_dark/2023_08_07_5ms/1to30 -> roni_dark/2023_08_07_5ms (batches are not experiments)."""
    parts = list(parts)
    while parts and BATCH_DIR.match(parts[-1]):
        parts.pop()
    return '/'.join(parts)


def default_run_name(parts):
    return experiment_name(parts).replace('/', '_')


# -- the prescan, cached --------------------------------------------------------------------------

def _code_hash():
    import scan_sparse_movies
    with open(scan_sparse_movies.__file__, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def movie_signature(movie_dir, params):
    mats = [(os.path.basename(p), os.path.getsize(p), int(os.path.getmtime(p)))
            for p in sparse_mats(movie_dir)]
    return json.dumps([mats, params, _code_hash()], sort_keys=True)


def scan_one(movie_dir, params):
    """Prep's prescan of one movie, reduced to what the PC needs. Top level, for a process pool."""
    from scan_sparse_movies import scan_movie
    result = scan_movie(movie_dir, params['pixel_threshold'], params['blob_ratio'],
                        params['blob_distance'], params['min_edge_margin'],
                        params['min_cams_in_frame'])
    if 'error' in result:
        return {'verdict': 'ERR', 'error': result['error']}
    ok = result['good_run_length'] >= params['min_intersection']
    return {'verdict': 'OK' if ok else 'BAD', 'good_start': result['good_start'],
            'good_end': result['good_end'], 'n_frames': result['n_frames'],
            'n_cams': result['n_cams'], 'good_run_length': result['good_run_length']}


class PrescanCache:
    def __init__(self, path):
        self.path = path
        try:
            with open(path, encoding='utf-8') as f:
                self.data = json.load(f)
        except (OSError, ValueError):
            self.data = {}

    def get(self, movie_dir, params):
        entry = self.data.get(os.path.abspath(movie_dir))
        if entry and entry.get('signature') == movie_signature(movie_dir, params):
            return entry['result']
        return None

    def put(self, movie_dir, params, result):
        self.data[os.path.abspath(movie_dir)] = {'signature': movie_signature(movie_dir, params),
                                                 'result': result}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        staged = self.path + '.partial'
        with open(staged, 'w', encoding='utf-8') as f:
            json.dump(self.data, f)
        os.replace(staged, self.path)


def prescan_movies(movie_dirs, params, cache, workers=1):
    """{movie folder: scan_one result}, from the cache where the mats have not changed."""
    results, todo = {}, []
    for movie_dir in movie_dirs:
        cached = cache.get(movie_dir, params)
        if cached is not None:
            results[movie_dir] = cached
        else:
            todo.append(movie_dir)
    if not todo:
        return results
    print(f"prescanning {len(todo)} movie(s) on this PC (cached afterwards) ...", flush=True)
    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {movie_dir: pool.submit(scan_one, movie_dir, params) for movie_dir in todo}
            for number, movie_dir in enumerate(todo, 1):
                results[movie_dir] = _settle(futures[movie_dir])
                cache.put(movie_dir, params, results[movie_dir])
                _progress(number, len(todo), movie_dir, results[movie_dir])
    else:
        for number, movie_dir in enumerate(todo, 1):
            try:
                results[movie_dir] = scan_one(movie_dir, params)
            except Exception as e:
                results[movie_dir] = {'verdict': 'ERR', 'error': f'{type(e).__name__}: {e}'}
            cache.put(movie_dir, params, results[movie_dir])
            _progress(number, len(todo), movie_dir, results[movie_dir])
    cache.save()
    return results


def _settle(future):
    try:
        return future.result()
    except Exception as e:
        return {'verdict': 'ERR', 'error': f'{type(e).__name__}: {e}'}


def _progress(number, total, movie_dir, result):
    if result['verdict'] == 'OK':
        detail = (f"frames {result['good_start']}-{result['good_end']} of "
                  f"{result['n_frames']}")
    elif result['verdict'] == 'BAD':
        detail = f"only {result['good_run_length']} usable frames in a row"
    else:
        detail = result.get('error', '')
    print(f"  [{number}/{total}] {os.path.basename(movie_dir)}: {result['verdict']}  {detail}",
          flush=True)


def box_stem(movie_dir, result):
    """The name prep will give the movie's box h5 -- and so its prediction folder."""
    from scan_sparse_movies import build_range
    rng = build_range(result['good_start'], result['good_end'], result['n_frames'])
    if rng is None:
        return None
    return f"mov_{movie_number(movie_dir)}_{rng[0]}_{rng[1]}_ds_3tc_7tj"


# -- the declaration ------------------------------------------------------------------------------

def load_declaration(exp_dir):
    path = os.path.join(exp_dir, PREP_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def easywand_candidates(exp_dir, unit_dirs):
    """Every easyWand .mat in the experiment, its units, and two folders up."""
    folders = [exp_dir] + list(unit_dirs)
    up = exp_dir
    for _ in range(2):
        up = os.path.dirname(up)
        folders.append(up)
    found = []
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(glob.escape(folder), '*.mat'))):
            name = os.path.basename(path).lower()
            if 'easywand' in name.replace('_', '') and path not in found:
                found.append(path)
    return found


def mirror_verdict(movie_dirs, easywand):
    """prep's mirror check on these movies against one easyWand, quietly; None if it cannot run."""
    from find_mirror_cam import PREP_SAMPLES, detect_mirror_cam
    try:
        return detect_mirror_cam(movie_dirs, easywand=easywand, samples=PREP_SAMPLES,
                                 verbose=False)
    except SystemExit as e:        # dlt_from_easywand exits on a mat that is not an easyWand
        return {'conclusive': False, 'flip': [], 'worst': float('inf'),
                'runner_up': float('inf'), 'reason': str(e), 'results': [], 'cam_names': []}
    except Exception as e:
        return {'conclusive': False, 'flip': [], 'worst': float('inf'),
                'runner_up': float('inf'), 'reason': f'{type(e).__name__}: {e}',
                'results': [], 'cam_names': []}


def fits(verdict):
    return bool(verdict) and verdict.get('conclusive') and verdict['worst'] <= GOOD_FIT_PX


def describe(verdict):
    if not verdict:
        return 'could not be checked'
    if verdict.get('conclusive'):
        flip = ('flip ' + ', '.join(verdict['flip'])) if verdict['flip'] else 'no flip'
        return (f"{flip:12} worst camera {verdict['worst']:6.1f} px "
                f"(next hypothesis {verdict['runner_up']:.1f} px)")
    return f"inconclusive -- {verdict.get('reason', '')}"


def relative_to(path, folder):
    return os.path.relpath(path, folder).replace('\\', '/')


def declare(exp_dir, units, scans, interactive, ask, n_cams):
    """Work out prep.json for an experiment, asking only what the data cannot settle."""
    from find_mirror_cam import sample_movies
    from scan_sparse_movies import PRESCAN_DEFAULTS
    print(f"\n--- declaring {exp_dir} (first time; the answers are saved as {PREP_FILE}) ---")
    ok_movies = [m for _, movies in units for m in movies if scans[m]['verdict'] == 'OK']
    if not ok_movies:
        raise Problem(f"no movie in {exp_dir} passes the prescan, so there is nothing to predict")
    sample = [d for d, _ in sample_movies([(m, movie_number(m)) for m in ok_movies])]
    print(f"cameras: {n_cams}; mirror check on {', '.join(os.path.basename(m) for m in sample)}")
    candidates = easywand_candidates(exp_dir, [u for u, _ in units])
    if not candidates:
        raise Problem(f"no easyWand .mat was found in {exp_dir}, its batch folders or the two "
                      f"folders above it. Put the experiment's easyWand there and run again")
    verdicts = {}
    for number, path in enumerate(candidates, 1):
        verdicts[path] = mirror_verdict(sample, path)
        print(f"  {number}. {relative_to(path, exp_dir):55} {describe(verdicts[path])}",
              flush=True)
    good = sorted((p for p in candidates if fits(verdicts[p])), key=lambda p: verdicts[p]['worst'])
    chosen, how = None, ''
    if len(good) == 1:
        chosen, how = good[0], 'the only easyWand that fits these movies'
    elif len(good) > 1 and verdicts[good[1]]['worst'] >= CLEAR_WIN * verdicts[good[0]]['worst']:
        chosen, how = good[0], 'clearly the best fit'
    if chosen is None:
        if not interactive:
            raise Problem("the easyWand cannot be chosen from the data (see the list above); "
                          "run predict.bat in a window to choose, or write prep.json")
        default = str(candidates.index(good[0]) + 1) if good else ''
        if n_cams == 2:
            print("  (a 2-camera rig cannot check its mirror camera; pick by the calibration "
                  "you know is right)")
        while chosen is None:
            answer = ask("Which easyWand is this experiment's? (number)", default)
            if answer.isdigit() and 1 <= int(answer) <= len(candidates):
                chosen, how = candidates[int(answer) - 1], 'chosen by you'
    verdict = verdicts[chosen]
    print(f"easyWand: {relative_to(chosen, exp_dir)} ({how})")

    cam, cam_how = None, ''
    if verdict.get('conclusive') and len(verdict['flip']) <= 1:
        cam = verdict['flip'][0] if verdict['flip'] else 'none'
        cam_how = 'from the mirror check'
    else:
        if not interactive:
            raise Problem("the mirror camera cannot be told from the data; run predict.bat in a "
                          "window to name it, or write prep.json")
        print("The mirror check cannot tell which camera films through the mirror "
              f"({verdict.get('reason') or 'more than one camera looks flipped'}).")
        while cam is None:
            answer = ask("Mirror camera to flip (e.g. cam1), or none").lower()
            if re.match(r'^(cam\d+|none)$', answer):
                cam, cam_how = answer, 'named by you'
    print(f"mirror camera: {cam} ({cam_how})")

    prep_args, bottom = [], 'auto'
    if n_cams == 2:
        if not interactive:
            raise Problem("a 2-camera rig must say which camera films from below; run "
                          "predict.bat in a window, or write prep.json")
        while True:
            answer = ask("Which camera films from below? (0-based index in file-name order)")
            if answer.isdigit() and int(answer) < 2:
                bottom = int(answer)
                break
        prep_args = ['--num-cams', '2', '--bottom-cam', str(bottom)]

    return {
        'schema': SCHEMA,
        'easywand': relative_to(chosen, exp_dir),
        'cam': cam,
        'num_cams': n_cams,
        'bottom_cam': bottom,
        'prep_args': prep_args,
        'prescan': dict(PRESCAN_DEFAULTS),
        'declared_at': dt.datetime.now().isoformat(timespec='seconds'),
        'declared_how': {'easywand': how, 'cam': cam_how},
        'mirror_check': {'movies': [os.path.basename(m) for m in sample],
                         'conclusive': bool(verdict.get('conclusive')),
                         'flip': list(verdict.get('flip') or []),
                         'worst_px': verdict.get('worst'), 'runner_up_px': verdict.get('runner_up'),
                         'candidates': {relative_to(p, exp_dir): describe(v)
                                        for p, v in verdicts.items()}},
    }


def settle_run_name(declaration, parts, interactive, ask):
    """The run name: the declared one, else the experiment's path joined with '_'. A name with no
    YYYY_MM_DD date breaks the consumer's rule that each experiment's folder is dated, so that is
    the one case worth a question."""
    if declaration.get('run_name'):
        return declaration['run_name']
    name = default_run_name(parts)
    if not DATED.search(name):
        print(f"note: the run name {name!r} carries no date (YYYY_MM_DD); the consumer expects "
              f"each experiment's folder to be dated")
        if interactive:
            answer = ask("Run name (Enter keeps it)", name)
            if re.match(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$', answer):
                name = answer
    return name


def save_declaration(exp_dir, declaration):
    path = os.path.join(exp_dir, PREP_FILE)
    staged = path + '.partial'
    with open(staged, 'w', encoding='utf-8') as f:
        json.dump(declaration, f, indent=1)
    os.replace(staged, path)
    print(f"saved {path}")
    return path


def declaration_hash(declaration):
    """What a movie's outcome is remembered against: the parts of prep.json that change how it
    preps, so a new easyWand or mirror camera gives movies prep turned away another chance."""
    keys = ('easywand', 'cam', 'num_cams', 'bottom_cam', 'prep_args', 'prescan')
    return hashlib.sha256(json.dumps({k: declaration.get(k) for k in keys},
                                     sort_keys=True).encode()).hexdigest()[:16]
