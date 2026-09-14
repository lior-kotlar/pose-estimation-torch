"""Re-run the ensemble step for already-predicted movies with the wing labels aligned first, and install
the new 3D points only where nothing got worse.

Some pose models come out with the two wings' labels the other way round from the rest, and the ensemble
step used to take the median of them together, putting both wings (and both hinges) on one physical wing.
Predictor2D.harmonize_wing_labels now aligns them before they are combined. Every movie's per-model
candidates (<member>/points_3D_all.npy) are on disk, so this needs no GPU and no re-prediction.

Per movie:
  1. skip it when harmonize_wing_labels finds nothing to exchange: its ensemble would come out identical;
  2. re-run find_3D_points_from_ensemble into <movie>/.realign_staging (symlinks to the members);
  3. compare old and new: frames with both wings on one wing, wing-shape error, and -- through
     FlightAnalysis -- out-of-range wing stroke angles (phi) and body pitch/yaw;
  4. if none of those got worse (and, where nothing got better, the shape error and the smoothed score did
     not rise either), move the old ensemble files into superseded_ensemble_<timestamp>/, put
     the new ones in place and re-analyse the movie with reanalyse_movies.py --with-mp4; otherwise leave
     the movie untouched and report it (the staging dir is kept with a BLOCKED.json for inspection).
The pipeline's own score (From2Dto3D.get_validation_score) is reported but does not block: it cannot tell
the wings apart, so one wing counted twice scores as a perfectly rigid wing.

usage:
    .env/bin/python code/realign_ensemble.py <movie_dir> [<movie_dir> ...] [--list FILE]
                                             [--max-models 3] [--dry-run] [--no-reanalyse] [--force]
"""
import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import traceback

import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (CODE_DIR, os.path.join(CODE_DIR, 'prediction_code_lior')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')

from Predictor import Predictor2D
from From_2D_to_3D import From2Dto3D
from predict import find_3D_points_from_ensemble
from extract_flight_data import FlightAnalysis

STAGING = '.realign_staging'
MARKER = '.realigned_ensemble.json'
RAW = 'points_3D_ensemble_best_method.npy'
SMOOTHED = 'points_3D_smoothed_ensemble_best_method.npy'
FA_DIRS = ('before_fa', 'after_fa')
# a movie counts as worse when any of these rises by more than its tolerance
TOL_COLLAPSE_PCT = 0.05     # percentage points of frames
TOL_PHI_WRONG_PCT = 0.05    # percentage points of phi values
TOL_SHAPE_REL = 0.02        # relative
# a movie that nothing got better in is worse when its shape error or smoothed score rises by more than
TOL_NO_GAIN_REL = 0.001     # relative


def members(movie_dir):
    return sorted(d for d in glob.glob(os.path.join(movie_dir, '*'))
                  if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'points_3D_all.npy')))


def collapse_pct(points):
    """Share of frames with both wings on one wing: tips < 1 mm and wing centres < 0.5 mm apart."""
    per_wing = (points.shape[1] - 2) // 2
    tips = np.linalg.norm(points[:, 2] - points[:, 2 + per_wing], axis=1)
    centres = np.linalg.norm(points[:, :per_wing - 1].mean(1) - points[:, per_wing:2 * per_wing - 1].mean(1), axis=1)
    return 100.0 * float(np.mean((tips < 1e-3) & (centres < 5e-4)))


def shape_error(points):
    """How much each wing's shape wobbles from frame to frame: the median |distance - its median| between
    neighbouring wing points, averaged over both wings (metres). A few garbage frames do not move it."""
    per_wing = (points.shape[1] - 2) // 2
    errors = []
    for start in (0, per_wing):
        idx = list(range(start, start + per_wing - 1))
        for a, b in zip(idx, idx[1:] + idx[:1]):
            d = np.linalg.norm(points[:, a] - points[:, b], axis=1)
            errors.append(np.nanmedian(np.abs(d - np.nanmedian(d))))
    return float(np.mean(errors))


def analysis_check(smoothed, work_dir, movie_dir):
    """Out-of-range phi share inside the roll window, plus body pitch and yaw, from FlightAnalysis."""
    os.makedirs(work_dir, exist_ok=True)
    for f in glob.glob(os.path.join(movie_dir, 'README_mov*')):   # start frame / cut / flip live here
        shutil.copy2(f, work_dir)
    path = os.path.join(work_dir, SMOOTHED)
    np.save(path, smoothed)
    fa = FlightAnalysis(points_3D_path=path, find_auto_correlation=False)
    window = np.isfinite(fa.roll_angle)
    phi = np.concatenate([fa.wings_phi_left[window], fa.wings_phi_right[window]])
    wrong = 100.0 * float(np.mean(~np.isfinite(phi) | (phi < 0) | (phi > 200)))
    return wrong, fa.pitch_angle, fa.yaw_angle


def install(movie_dir, staging, stamp):
    """Move the old ensemble files aside and the new ones in. Returns (archive dir, entries replaced)."""
    archive = os.path.join(movie_dir, 'superseded_ensemble_' + stamp)
    os.makedirs(archive)
    moved = []
    for entry in sorted(os.listdir(staging)):
        src = os.path.join(staging, entry)
        if os.path.islink(src) or entry in FA_DIRS:
            continue
        dst = os.path.join(movie_dir, entry)
        if os.path.exists(dst):
            shutil.move(dst, os.path.join(archive, entry))
        shutil.move(src, dst)
        moved.append(entry)
    return archive, moved


def realign(movie_dir, max_models=3, dry_run=False, reanalyse=True, force=False):
    movie_dir = os.path.abspath(movie_dir.rstrip('/'))
    result = {'movie': movie_dir}
    if os.path.exists(os.path.join(movie_dir, MARKER)) and not force:
        result['status'] = 'already realigned'
        return result
    dirs = members(movie_dir)
    if len(dirs) < 2:
        result['status'] = 'fewer than 2 models, nothing to align'
        return result
    _, num_swapped = Predictor2D.harmonize_wing_labels([np.load(os.path.join(d, 'points_3D_all.npy')) for d in dirs])
    result['exchanged_pairs'] = num_swapped
    if num_swapped == 0:
        result['status'] = 'nothing to align, the ensemble would come out identical'
        return result
    if dry_run:
        result['status'] = 'would re-run'
        return result

    staging = os.path.join(movie_dir, STAGING)
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    for d in dirs:
        os.symlink(d, os.path.join(staging, os.path.basename(d)))
    find_3D_points_from_ensemble(staging, max_models=max_models)

    old_raw, old_sm = np.load(os.path.join(movie_dir, RAW)), np.load(os.path.join(movie_dir, SMOOTHED))
    new_raw, new_sm = np.load(os.path.join(staging, RAW)), np.load(os.path.join(staging, SMOOTHED))
    old_wrong, old_pitch, old_yaw = analysis_check(old_sm, os.path.join(staging, FA_DIRS[0]), movie_dir)
    new_wrong, new_pitch, new_yaw = analysis_check(new_sm, os.path.join(staging, FA_DIRS[1]), movie_dir)
    score = From2Dto3D.get_validation_score
    result.update(
        collapse_pct=[collapse_pct(old_sm), collapse_pct(new_sm)],
        shape_error_um=[1e6 * shape_error(old_sm), 1e6 * shape_error(new_sm)],
        phi_wrong_pct=[old_wrong, new_wrong],
        score_raw=[float(score(old_raw)), float(score(new_raw))],
        score_smoothed=[float(score(old_sm)), float(score(new_sm))],
        score_smoothed_without_edges=[float(score(old_sm[5:-5])), float(score(new_sm[5:-5]))])
    pitch_yaw_same = (np.allclose(old_pitch, new_pitch, equal_nan=True, atol=1e-9)
                      and np.allclose(old_yaw, new_yaw, equal_nan=True, atol=1e-9))
    result['pitch_yaw_unchanged'] = bool(pitch_yaw_same)

    worse = []
    if result['collapse_pct'][1] > result['collapse_pct'][0] + TOL_COLLAPSE_PCT:
        worse.append('more frames with both wings on one wing')
    if result['phi_wrong_pct'][1] > result['phi_wrong_pct'][0] + TOL_PHI_WRONG_PCT:
        worse.append('more out-of-range phi')
    if result['shape_error_um'][1] > result['shape_error_um'][0] * (1 + TOL_SHAPE_REL):
        worse.append('wing shapes wobble more')
    if not pitch_yaw_same:
        worse.append('body pitch/yaw changed')
    # the tolerances above are for movies the realignment rescues, where frames that held one wing
    # twice now hold two real wings. A movie it does not improve must not get worse at all
    improved = (result['collapse_pct'][1] < result['collapse_pct'][0]
                or result['phi_wrong_pct'][1] < result['phi_wrong_pct'][0])
    if not improved and (result['collapse_pct'][1] > result['collapse_pct'][0]
                         or result['phi_wrong_pct'][1] > result['phi_wrong_pct'][0]
                         or result['shape_error_um'][1] > result['shape_error_um'][0] * (1 + TOL_NO_GAIN_REL)
                         or result['score_smoothed'][1] > result['score_smoothed'][0] * (1 + TOL_NO_GAIN_REL)):
        worse.append('nothing fixed and slightly worse')
    if worse:
        result['status'] = 'BLOCKED: ' + ', '.join(worse)
        with open(os.path.join(staging, 'BLOCKED.json'), 'w') as f:
            json.dump(result, f, indent=1)
        return result

    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive, moved = install(movie_dir, staging, stamp)
    shutil.rmtree(staging)
    result.update(archive=archive, replaced=moved, realigned_at=stamp)
    with open(os.path.join(movie_dir, MARKER), 'w') as f:
        json.dump(result, f, indent=1)
    result['status'] = 'realigned'
    if reanalyse:
        rc = subprocess.run([sys.executable, os.path.join(CODE_DIR, 'reanalyse_movies.py'), movie_dir, '--with-mp4']).returncode
        result['reanalyse_exit_code'] = rc
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dirs', nargs='*', help='movie dirs')
    parser.add_argument('--list', help='file with one movie dir per line (further tab-separated columns are ignored)')
    parser.add_argument('--max-models', type=int, default=3,
                        help="the predict config's 'max ensemble models' (3 in predict_configurations/config1.json)")
    parser.add_argument('--dry-run', action='store_true', help='only report which movies would be re-run')
    parser.add_argument('--no-reanalyse', action='store_true', help='install the new points but do not re-analyse')
    parser.add_argument('--force', action='store_true', help='re-run a movie even if it was already realigned')
    args = parser.parse_args()
    dirs = list(args.dirs)
    if args.list:
        dirs += [line.split('\t')[0].strip() for line in open(args.list) if line.strip()]
    for d in dirs:
        try:
            r = realign(d, args.max_models, args.dry_run, not args.no_reanalyse, args.force)
        except Exception as e:
            traceback.print_exc()
            r = {'movie': d, 'status': 'FAILED: %r' % e}
        print('REALIGN ' + json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
