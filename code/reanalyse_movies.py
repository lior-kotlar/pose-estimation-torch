"""Re-run only the analysis stage of the pipeline on already-predicted movies.

The prediction stage (2D nets, ensemble selection, triangulation, 3D smoothing) ends at
points_3D_smoothed_ensemble_best_method.npy. Everything after that -- wing and body angles,
the analysis h5, the CSV, the wing-angle plot -- is derived from that one file on the CPU.

So a fix that only touches angle extraction does not need the GPU, the calibration or the
source box h5: this rewrites the derived products in place from the npy that is already there.
By default the 3D points are left untouched, and so are
points_ensemble_smoothed_reprojected.npy and "movie 2D and 3D.mp4". Those stay geometrically
correct -- they hold the same set of 3D locations either way -- but FlightAnalysis decides
left from right before reprojecting, so on a movie where that decision changes the stale
reprojection keeps the old index order and the mp4 colours the wings the other way round from
the new h5. Pass --with-mp4 to rewrite those two as well; that needs the calibration and the
source box h5 (both recovered from the saved member config), but still no GPU and no
re-prediction. Movies whose source movie has since been deleted keep their old video and are
listed at the end -- their angles are up to date either way.

The trigger offset, frame rate and provenance are read back out of the existing analysis h5,
so no other input is needed. The superseded h5/csv/png are moved into superseded_<timestamp>/
rather than deleted, so a run can be compared against what it replaced.

Usage:
    .env/bin/python code/reanalyse_movies.py <dir> [<dir> ...] [--with-mp4] [--dry-run]

<dir> may be a single movie directory, or any directory above one -- every movie dir holding
a points_3D_smoothed_ensemble_best_method.npy underneath it is re-analysed.
"""
import argparse
import datetime as dt
import os
import shutil
import sys
import traceback

import glob
import json

import h5py
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (CODE_DIR, os.path.join(CODE_DIR, 'prediction_code_lior')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')

from Triangulator import Triangulator
from Visualizer import Visualizer

from extract_flight_data import FlightAnalysis, create_movie_analysis_h5, export_analysis_csv
from plot_wing_and_body import plot_one as plot_movie_figures, FIGURE_NAMES
from plot_flight_viewer import (make_viewer as make_flight_viewer,
                                OUT_SUFFIX as VIEWER_SUFFIX)

POINTS_NAME = 'points_3D_smoothed_ensemble_best_method.npy'
PROVENANCE_KEYS = ("experiment", "movie_dir", "source_movie_dir", "box_h5")
REPROJECTED_NAME = 'points_ensemble_smoothed_reprojected.npy'
MP4_NAME = 'movie 2D and 3D.mp4'


class NoSourceMovie(Exception):
    """The source box h5 this movie was predicted from is no longer on disk."""


def read_prediction_config(movie_dir):
    """Box h5 and calibration, read back out of any ensemble member's saved config.

    Each member wrote the fully resolved configuration it ran with, so the source movie and
    the calibration are recoverable from the output directory alone -- no manifest, no config
    file, and nothing that has to be passed in on the command line.
    """
    for pattern in ('*/specific_configuration.json', '*/configuration.json'):
        for cfg_path in sorted(glob.glob(os.path.join(movie_dir, pattern))):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
            except (OSError, ValueError):
                continue
            box = cfg.get('movie path')
            calibration = cfg.get('calibration path')
            if box and calibration:
                return (box, calibration,
                        int(cfg.get('IMAGE HEIGHT', 800)), int(cfg.get('IMAGE WIDTH', 1280)))
    raise NoSourceMovie(f'no member config with a movie/calibration path under {movie_dir}')


def regenerate_video(movie_dir, analysis, h5_path, trigger_offset, frame_rate,
                    stamp=None, archive=True, force=False):
    """Rewrite the reprojected 2D points and the overlay mp4 from the re-analysed points.

    FlightAnalysis decides left from right, so its points_3D can come out in a different
    index order than the run that produced the existing reprojection -- the same 3D locations,
    but the wings labelled the other way round. The mp4 colours by index, so leaving it alone
    after that decision changes would show the wings swapped relative to the new h5. This is
    the same reprojection predict.py does; it needs the calibration and the source movie, but
    no GPU and no re-prediction.
    """
    box_path, calibration_path, image_height, image_width = read_prediction_config(movie_dir)
    if not os.path.isfile(box_path):
        raise NoSourceMovie(box_path)

    # read the cropzone straight out of the h5 rather than importing Predictor2D, which would
    # drag torch in for a two-line array read
    with h5py.File(box_path, 'r') as box:
        cropzone = box['/cropzone'][:] if '/cropzone' in box else box['/cropZone'][:]

    triangulator = Triangulator(calibration_path, image_height, image_width)
    reprojected = triangulator.get_reprojections(
        analysis.points_3D[analysis.first_analysed_frame:], cropzone)

    # reprojecting is seconds, rendering the mp4 is minutes, and on most movies the left/right
    # decision did not change -- so the new reprojection is identical to the one already on
    # disk and re-rendering would burn minutes to produce the same video. compare first
    reprojected_path = os.path.join(movie_dir, REPROJECTED_NAME)
    # an mp4 that is missing has to be rendered whatever the reprojection says -- and so does
    # one left behind by an interrupted render, which is why the npy is written only after the
    # render succeeds: an npy on disk means its mp4 was finished
    mp4_path = os.path.join(movie_dir, MP4_NAME)
    if not force and os.path.isfile(reprojected_path) and os.path.isfile(mp4_path):
        previous = np.load(reprojected_path)
        if previous.shape == reprojected.shape and np.allclose(previous, reprojected,
                                                               atol=1e-6, equal_nan=True):
            return 'unchanged, mp4 left as is'

    if archive and stamp is not None:
        archive_previous(movie_dir, stamp, names=(REPROJECTED_NAME, MP4_NAME))
    # create_movie_mp4 reads the points back from disk, so the npy has to exist first; write it
    # to a temporary name and only move it into place once the render has finished, so an
    # interrupted run leaves no npy and the next run redoes both. np.save appends '.npy' to any
    # path that does not already end in it, so the staged name has to carry the suffix itself --
    # otherwise the move below goes looking for a file that was never written under that name.
    staged_path = reprojected_path[:-len('.npy')] + '.partial.npy'
    np.save(staged_path, reprojected)
    try:
        Visualizer.create_movie_mp4(h5_path, save_frames=None, mode='SAVE',
                                    reprojected_points_path=staged_path,
                                    box_path=box_path,
                                    save_path=mp4_path, rotate=True,
                                    trigger_offset=trigger_offset, frame_rate=frame_rate)
    except BaseException:
        for leftover in (staged_path, mp4_path):
            if os.path.exists(leftover):
                os.remove(leftover)
        raise
    shutil.move(staged_path, reprojected_path)
    return 'rewritten'


def find_movie_dirs(root):
    """Every directory under root (or root itself) holding a smoothed 3D points file."""
    root = os.path.abspath(root)
    if os.path.exists(os.path.join(root, POINTS_NAME)):
        return [root]
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if POINTS_NAME in filenames:
            found.append(dirpath)
    return sorted(found)


def read_existing_context(movie_dir):
    """Pull trigger offset, frame rate and provenance back out of the current analysis h5.

    Returns (trigger_offset, frame_rate, source, perturbation, h5_path); h5_path is None when
    the movie has never been analysed, in which case the analysis still runs, just without
    trigger numbering.

    The perturbation window is recovered here too, in the shape utils.load_perturbation
    returns, so a re-analysed movie keeps the band it declared instead of silently losing it.
    Reading it back out of the h5 rather than off the experiment's perturbation.json keeps
    this script's promise of needing no input beyond the movie directory -- and a movie
    analysed before the perturbation flags existed simply has nothing to restore.
    """
    matches = [f for f in os.listdir(movie_dir) if f.endswith('_analysis_smoothed.h5')]
    if not matches:
        return None, None, None, None
    h5_path = os.path.join(movie_dir, matches[0])
    trigger_offset = frame_rate = None
    source = {}
    with h5py.File(h5_path, 'r') as hdf:
        if 'trigger_offset' in hdf:
            trigger_offset = int(hdf['trigger_offset'][()])
        if 'frame_rate' in hdf:
            frame_rate = float(hdf['frame_rate'][()])
        for key in PROVENANCE_KEYS:
            if key in hdf:
                value = hdf[key][()]
                source[key] = value.decode() if isinstance(value, bytes) else str(value)
    perturbation = read_perturbation(h5_path)
    if perturbation is None:
        # A re-analysis run by a version that did not carry the window forward leaves an h5
        # with no perturbation datasets, so the live file cannot always answer. The archives
        # hold every superseded h5, so walk them newest-first for one that still declares it
        # rather than dropping the band permanently on the second run.
        for archived in sorted(glob.glob(os.path.join(movie_dir, 'superseded_*', '*_analysis_smoothed.h5')),
                               reverse=True):
            perturbation = read_perturbation(archived)
            if perturbation is not None:
                break
    return trigger_offset, frame_rate, (source or None), perturbation, h5_path


def read_perturbation(h5_path):
    """The perturbation window declared in one analysis h5, or None.

    Shaped the way utils.load_perturbation returns it, so it can be handed straight back to
    create_movie_analysis_h5 and export_analysis_csv.
    """
    try:
        with h5py.File(h5_path, 'r') as hdf:
            if 'perturbation' not in hdf or not int(hdf['perturbation'][()]):
                return None
            kind = hdf['perturbation_type'][()] if 'perturbation_type' in hdf else b'unspecified'
            # An absent duration is how "the log never recorded one" is expressed, so carry the
            # absence through rather than defaulting it to zero -- see utils.PERT_UNKNOWN. The
            # onset is known whenever an experiment is declared perturbed, so it is read flat.
            end_known = (bool(int(hdf['perturbation_end_known'][()]))
                         if 'perturbation_end_known' in hdf else False)
            return {
                'type': kind.decode() if isinstance(kind, bytes) else str(kind),
                'onset_frame': int(hdf['perturbation_start_frame'][()]),
                'duration_ms': (float(hdf['perturbation_duration_ms'][()])
                                if end_known and 'perturbation_duration_ms' in hdf else None),
                'end_frame': (int(hdf['perturbation_end_frame'][()])
                              if end_known and 'perturbation_end_frame' in hdf else None),
                'end_known': end_known,
                'source': f'restored from {os.path.basename(h5_path)}',
            }
    except (OSError, KeyError):
        return None


def archive_previous(movie_dir, stamp, names=None):
    """Move the products this script is about to overwrite into superseded_<stamp>/."""
    if names is None:
        doomed = [f for f in os.listdir(movie_dir)
                  if f.endswith('_analysis_smoothed.h5')
                  or f.endswith('_analysis_smoothed.csv')
                  or f in FIGURE_NAMES
                  or f.endswith(VIEWER_SUFFIX)
                  or f == 'All body data.html']
    else:
        doomed = [f for f in names if os.path.exists(os.path.join(movie_dir, f))]
    if not doomed:
        return None
    archive_dir = os.path.join(movie_dir, f'superseded_{stamp}')
    os.makedirs(archive_dir, exist_ok=True)
    for name in doomed:
        shutil.move(os.path.join(movie_dir, name), os.path.join(archive_dir, name))
    return archive_dir


def reanalyse(movie_dir, stamp, archive=True, with_video=False, force_video=False):
    points_path = os.path.join(movie_dir, POINTS_NAME)
    movie = os.path.basename(movie_dir.rstrip(os.sep))
    trigger_offset, frame_rate, source, perturbation, _ = read_existing_context(movie_dir)

    # build the analysis first, so a movie that fails keeps the products it already had
    analysis = FlightAnalysis(points_3D_path=points_path, find_auto_correlation=True,
                              create_html=False, create_mp4=False, create_h5=False)
    if archive:
        archive_previous(movie_dir, stamp)

    h5_path, _ = create_movie_analysis_h5(movie, movie_dir, points_path, smooth=True,
                                          analysis_object=analysis,
                                          trigger_offset=trigger_offset,
                                          frame_rate=frame_rate, source=source,
                                          perturbation=perturbation)
    export_analysis_csv(analysis, h5_path.replace('.h5', '.csv'), trigger_offset or 0, frame_rate,
                        perturbation=perturbation)
    plot_movie_figures(h5_path, units="frames")
    make_flight_viewer(h5_path)

    video = None
    if with_video:
        # a missing source movie costs the mp4, not the analysis that already succeeded
        try:
            video = regenerate_video(movie_dir, analysis, h5_path, trigger_offset, frame_rate,
                                     stamp=stamp, archive=archive, force=force_video)
        except NoSourceMovie as e:
            video = f'skipped (source movie gone: {e})'

    lo, hi = analysis.first_y_body_frame, analysis.end_frame
    spans = []
    for wing in ('left', 'right'):
        psi = getattr(analysis, f'wings_psi_{wing}')[lo:hi]
        psi = psi[np.isfinite(psi)]
        spans.append(psi.max() - psi.min() if psi.size else np.nan)
    return h5_path, spans, video


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dirs', nargs='+', help='movie dir, or any dir above movie dirs')
    parser.add_argument('--dry-run', action='store_true',
                        help='list the movies that would be re-analysed and stop')
    parser.add_argument('--no-archive', action='store_true',
                        help='overwrite the previous products instead of moving them aside')
    parser.add_argument('--force-mp4', action='store_true',
                        help='with --with-mp4, re-render even when the reprojection is '
                             'unchanged (by default an identical reprojection skips the render)')
    parser.add_argument('--with-mp4', action='store_true',
                        help='also rewrite points_ensemble_smoothed_reprojected.npy and the '
                             'overlay mp4. needs the source box h5 and the calibration, both '
                             'read from the saved member config; movies whose source movie is '
                             'no longer on disk keep their old video and are reported')
    args = parser.parse_args()

    movie_dirs = []
    for root in args.dirs:
        movie_dirs.extend(find_movie_dirs(root))
    movie_dirs = sorted(set(movie_dirs))
    if not movie_dirs:
        print(f"no movie dirs with a {POINTS_NAME} under: {', '.join(args.dirs)}")
        return 1

    print(f"{len(movie_dirs)} movie(s) to re-analyse")
    if args.dry_run:
        for d in movie_dirs:
            print(f"  {d}")
        return 0

    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    failures = []
    no_video = []
    for i, movie_dir in enumerate(movie_dirs, 1):
        name = os.path.basename(movie_dir.rstrip(os.sep))
        print(f"\n[{i}/{len(movie_dirs)}] {name}", flush=True)
        try:
            _, (span_l, span_r), video = reanalyse(movie_dir, stamp,
                                                   archive=not args.no_archive,
                                                   with_video=args.with_mp4,
                                                   force_video=args.force_mp4)
            print(f"  psi span: left {span_l:.0f} deg, right {span_r:.0f} deg", flush=True)
            if video is not None:
                print(f"  video: {video}", flush=True)
                if video.startswith('skipped'):
                    no_video.append(name)
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    print(f"\nre-analysed {len(movie_dirs) - len(failures)} of {len(movie_dirs)}")
    for name, err in failures:
        print(f"  FAILED {name}: {err}")
    if no_video:
        print(f"\n{len(no_video)} movie(s) kept their old mp4 because the source movie is gone;"
              f" their angles are still up to date:")
        for name in no_video:
            print(f"  {name}")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
