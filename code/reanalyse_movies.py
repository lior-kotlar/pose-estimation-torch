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
so no other input is needed. The declaration (pulse window AND lighting) is taken from the live
perturbation.json when one applies (--perturbation-source auto, the default), so an edit to the
declaration reaches every product without re-predicting; --perturbation-source h5 keeps what the
old analysis h5 recorded instead. The superseded h5/csv/png are moved into superseded_<timestamp>/
rather than deleted, so a run can be compared against what it replaced.

Off the cluster. A movie's analysis reads three files -- the smoothed 3D points, the previous
analysis h5 and source.json -- plus its experiment's perturbation.json, so an experiment kept on
another machine is re-analysed where it lives (LOCAL_REANALYSIS.md). The paths recorded at
predict time are cluster paths; --path-map SERVER=LOCAL rewrites them wherever one is opened and
maps the declaration's path back before it is written, so the products read the same whichever
machine made them. A movie whose trigger cannot be found is refused rather than numbered from
box frame 0 (--allow-no-trigger overrides).

Staleness. Every re-analysed movie is stamped -- in its h5 and in source.json -- with a
fingerprint of the analysis code, of the declaration it used and of the 3D points it read.
--only-stale skips a movie whose stamp still matches, so re-running the same command resumes an
interrupted run and brings a whole tree up to date without redoing what already is. New points
(code/realign_ensemble.py installs some) make a movie stale like a code change does.

Usage:
    .env/bin/python code/reanalyse_movies.py <dir> [<dir> ...] [--jobs N] [--only-stale]
        [--path-map SERVER=LOCAL] [--with-mp4] [--dry-run]

<dir> may be a single movie directory, or any directory above one -- every movie dir holding
a points_3D_smoothed_ensemble_best_method.npy underneath it is re-analysed, except the archived
copies under superseded_*/ and hidden directories such as .realign_staging/. --dry-run is a
preflight: it reads, per movie, where the trigger and the declaration would come from and
whether the products are stale, and writes nothing. A run over a directory above the movies
leaves reanalyse_report_<timestamp>.csv in that directory.
"""
import argparse
import contextlib
import csv
import datetime as dt
import glob
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

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

from extract_flight_data import (FlightAnalysis, create_movie_analysis_h5, export_analysis_csv,
                                 _h5_text)
from plot_wing_and_body import plot_one as plot_movie_figures, FIGURE_NAMES, lighting_info
from utils import load_perturbation, stamp_declaration, get_trigger_frame_info, pitch_read_sign
from plot_flight_viewer import (make_viewer as make_flight_viewer,
                                OUT_SUFFIX as VIEWER_SUFFIX)

POINTS_NAME = 'points_3D_smoothed_ensemble_best_method.npy'
PROVENANCE_KEYS = ("experiment", "movie_dir", "source_movie_dir", "box_h5")
REPROJECTED_NAME = 'points_ensemble_smoothed_reprojected.npy'
MP4_NAME = 'movie 2D and 3D.mp4'
SOURCE_JSON = 'source.json'
# code/realign_ensemble.py leaves this behind in a movie whose ensemble it re-ran
REALIGN_MARKER = '.realigned_ensemble.json'

# The sources that decide what the analysis products contain. An edit to any of them makes
# every movie stale for --only-stale. Deliberately broad: redoing a movie costs seconds, and
# shipping one made by older code is the failure this exists to prevent.
ANALYSIS_CODE = ('reanalyse_movies.py', 'utils.py', 'plot_wing_and_body.py',
                 'plot_flight_viewer.py', 'prediction_code_lior/extract_flight_data.py',
                 'prediction_code_lior/Visualizer.py')

# Parallel workers each run one movie on one core; left to themselves, numpy's BLAS threads in
# every worker would fight over the same cores.
BLAS_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS')

# Run report: the before/after check against the superseded h5. Yaw and pitch come from the
# head-tail axis alone, which no analysis fix touches, so movement beyond this is a surprise
# worth a look; roll is expected to move and is reported without a flag.
PITCH_YAW_CHANGE_DEG = 0.5
# A rise in the share of NaN wing-angle frames beyond this marks frames the new code rejected.
WING_NAN_RISE = 0.01
BODY_ANGLE_KEYS = ('yaw_angle', 'pitch_angle', 'roll_angle')
WING_ANGLE_KEYS = tuple(f'wings_{angle}_{side}' for angle in ('phi', 'theta', 'psi')
                        for side in ('left', 'right'))
REPORT_FIELDS = ('experiment', 'movie', 'status', 'seconds', 'frames', 'declaration', 'lighting',
                 'trigger', 'psi_span_left_deg', 'psi_span_right_deg',
                 'max_abs_change_yaw_deg', 'max_abs_change_pitch_deg', 'max_abs_change_roll_deg',
                 'wing_nan_frac_old', 'wing_nan_frac_new', 'check', 'video', 'error', 'movie_dir')


class NoSourceMovie(Exception):
    """The source box h5 this movie was predicted from is no longer on disk."""


class NoTrigger(Exception):
    """Neither the analysis h5 nor the source movie's sparse mats place the camera trigger."""


def parse_path_maps(specs):
    """--path-map SERVER=LOCAL values as (server prefix, local prefix) pairs."""
    maps = []
    for spec in specs or ():
        server, sep, local = spec.partition('=')
        if not sep or not server or not local:
            raise SystemExit(f"--path-map wants SERVER=LOCAL, got {spec!r}")
        maps.append((server.rstrip('/'), local))
    return tuple(maps)


def map_path(path, path_maps):
    """A path recorded on the cluster, as it is reachable on this machine.

    The longest matching prefix wins, so a specific map (an experiment kept outside the
    mirrored tree, say) can sit next to a general one in any order."""
    if not path:
        return path
    matches = [(server, local) for server, local in path_maps
               if path == server or path.startswith(server + '/')]
    if not matches:
        return path
    server, local = max(matches, key=lambda pair: len(pair[0]))
    rest = path[len(server):].lstrip('/')
    return os.path.normpath(os.path.join(local, *rest.split('/')) if rest else local)


def unmap_path(path, path_maps):
    """The inverse of map_path: the cluster path a local one stands for.

    Applied to every path this script writes into a product, so an h5 made on a laptop names
    the same perturbation.json as one made on the cluster."""
    if not path:
        return path
    here = os.path.normpath(path)
    matches = []
    for server, local in path_maps:
        root = os.path.normpath(local).rstrip(os.sep)
        if (os.path.normcase(here) == os.path.normcase(root)
                or os.path.normcase(here).startswith(os.path.normcase(root) + os.sep)):
            matches.append((server, root))
    if not matches:
        return path
    # a recorded path can be relative to the cluster project (old member configs hold
    # 'inference_datasets/...'), so a relative map may cover the same local files as an absolute
    # one; the absolute cluster path is the one worth recording
    server, root = max(matches, key=lambda pair: (pair[0].startswith('/'), len(pair[1])))
    rest = here[len(root):].lstrip(os.sep)
    return server + ('/' + rest.replace(os.sep, '/') if rest else '')


def code_fingerprint():
    """A short hash of the analysis code, identical for a Windows and a Linux checkout."""
    digest = hashlib.sha256()
    for rel in ANALYSIS_CODE:
        with open(os.path.join(CODE_DIR, *rel.split('/')), 'rb') as f:
            # git may check the sources out with CRLF line ends on Windows
            text = f.read().replace(b'\r\n', b'\n')
        digest.update(rel.encode() + b'\0' + text + b'\0')
    return digest.hexdigest()[:16]


def points_fingerprint(movie_dir):
    """A short hash of the 3D points the analysis reads, or '' when they are not there.

    The points are an input, not code, so nothing above notices when they change. They do change:
    code/realign_ensemble.py re-runs a movie's ensemble and installs new ones. Stamping this makes
    such a movie stale by itself, and lets a shipped h5 name the points it was made from."""
    path = os.path.join(movie_dir, POINTS_NAME)
    digest = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                digest.update(block)
    except OSError:
        return ''
    return digest.hexdigest()[:16]


def realigned_at(movie_dir):
    """When code/realign_ensemble.py last installed a new ensemble here, or '' if it never did."""
    try:
        with open(os.path.join(movie_dir, REALIGN_MARKER), encoding='utf-8') as f:
            return str(json.load(f).get('realigned_at') or '')
    except (OSError, json.JSONDecodeError, AttributeError):
        return ''


def declaration_fingerprint(perturbation):
    """A short hash of the resolved declaration, minus the machine-specific path it came from."""
    if perturbation is None:
        payload = 'none'
    else:
        payload = json.dumps({k: v for k, v in perturbation.items() if k != 'source'},
                             sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# local_reanalysis.py update writes the cluster commit a downloaded copy of the code came from
BUNDLE_COMMIT_FILE = os.path.join(os.path.dirname(CODE_DIR), 'BUNDLE_COMMIT')


def git_commit():
    """The checkout's short commit, '-dirty' when the analysis code differs from it.

    A copy downloaded by local_reanalysis.py reports the commit it was downloaded at; any other
    copy that is not a git checkout reports 'unknown'."""
    try:
        head = subprocess.run(['git', '-C', CODE_DIR, 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, timeout=30)
        if head.returncode != 0:
            return bundle_commit()
        dirty = subprocess.run(['git', '-C', CODE_DIR, 'status', '--porcelain', '--',
                                *ANALYSIS_CODE], capture_output=True, text=True, timeout=30)
        return head.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')
    except (OSError, subprocess.SubprocessError):
        return bundle_commit()


def bundle_commit():
    try:
        with open(BUNDLE_COMMIT_FILE, encoding='utf-8') as f:
            return f.read().strip() or 'unknown'
    except OSError:
        return 'unknown'


def read_prediction_config(movie_dir):
    """Box h5 and calibration, read back out of any ensemble member's saved config.

    Each member wrote the fully resolved configuration it ran with, so the source movie and
    the calibration are recoverable from the output directory alone -- no manifest, no config
    file, and nothing that has to be passed in on the command line.
    """
    for pattern in ('*/specific_configuration.json', '*/configuration.json'):
        for cfg_path in sorted(glob.glob(os.path.join(movie_dir, pattern))):
            try:
                with open(cfg_path, encoding='utf-8') as f:
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
                     stamp=None, archive=True, force=False, perturbation=None, path_maps=()):
    """Rewrite the reprojected 2D points and the overlay mp4 from the re-analysed points.

    FlightAnalysis decides left from right, so its points_3D can come out in a different
    index order than the run that produced the existing reprojection -- the same 3D locations,
    but the wings labelled the other way round. The mp4 colours by index, so leaving it alone
    after that decision changes would show the wings swapped relative to the new h5. This is
    the same reprojection predict.py does; it needs the calibration and the source movie, but
    no GPU and no re-prediction.
    """
    box_path, calibration_path, image_height, image_width = read_prediction_config(movie_dir)
    box_path, calibration_path = map_path(box_path, path_maps), map_path(calibration_path, path_maps)
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
                                    trigger_offset=trigger_offset, frame_rate=frame_rate,
                                    perturbation=perturbation)
    except BaseException:
        for leftover in (staged_path, mp4_path):
            if os.path.exists(leftover):
                os.remove(leftover)
        raise
    shutil.move(staged_path, reprojected_path)
    return 'rewritten'


def is_archive_dir(name):
    """superseded_<stamp>/ and dot-dirs hold replaced copies of a movie's files, not movies."""
    return name.startswith('superseded_') or name.startswith('.')


def find_movie_dirs(root):
    """Every directory under root (or root itself) holding a smoothed 3D points file.

    Archives are pruned from the walk: realign_ensemble.py moves a movie's previous
    points file into superseded_ensemble_<stamp>/ (and stages candidates in
    .realign_staging/), and re-analysing that copy would write products from the
    points the movie no longer uses."""
    root = os.path.abspath(root)
    if os.path.exists(os.path.join(root, POINTS_NAME)):
        return [root]
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not is_archive_dir(d))
        if POINTS_NAME in filenames:
            found.append(dirpath)
    return sorted(found)


def read_existing_context(movie_dir):
    """Pull trigger offset, frame rate and provenance back out of the current analysis h5.

    Returns (trigger_offset, frame_rate, source, perturbation, h5_path); h5_path is None when
    the movie has never been analysed.

    The perturbation window is recovered here too, in the shape utils.load_perturbation
    returns, so a re-analysed movie keeps the band it declared instead of silently losing it.
    Reading it back out of the h5 rather than off the experiment's perturbation.json keeps
    this script's promise of needing no input beyond the movie directory -- and a movie
    analysed before the perturbation flags existed simply has nothing to restore.
    """
    live = sorted(os.path.join(movie_dir, f) for f in os.listdir(movie_dir)
                  if f.endswith('_analysis_smoothed.h5'))
    archived = sorted(glob.glob(os.path.join(movie_dir, 'superseded_*', '*_analysis_smoothed.h5')),
                      reverse=True)
    if not live and not archived:
        # Five values, matching the unpack in resolve_context(): a movie dir with the
        # 3D points but no analysis h5 must return cleanly, not ValueError.
        return None, None, None, None, None
    h5_path = live[0] if live else None
    trigger_offset = frame_rate = None
    source = {}
    # The live h5 first, then the archives newest-first. A run interrupted after moving the old
    # products aside but before writing the new h5 leaves no live h5, or a half-written one that
    # will not open; the trigger and provenance are properties of the movie, so the superseded
    # copy answers just as well -- and off the cluster it is the only thing that can.
    for candidate in live + archived:
        try:
            with h5py.File(candidate, 'r') as hdf:
                if trigger_offset is None and 'trigger_offset' in hdf:
                    trigger_offset = int(hdf['trigger_offset'][()])
                    if 'frame_rate' in hdf:
                        frame_rate = float(hdf['frame_rate'][()])
                if not source:
                    for key in PROVENANCE_KEYS:
                        if key in hdf:
                            value = hdf[key][()]
                            source[key] = value.decode() if isinstance(value, bytes) else str(value)
        except OSError:
            continue
        if trigger_offset is not None and source:
            break
    perturbation = read_perturbation(h5_path) if h5_path else None
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
            # Key off `perturbation_declared`, not `perturbation`: a movie
            # declared as an unperturbed CONTROL has perturbation == 0 but is
            # still declared, and dropping that on every re-analysis would turn
            # a known control back into an undeclared movie.
            declared = ('perturbation_declared' in hdf
                        and bool(int(hdf['perturbation_declared'][()])))
            has_window = ('perturbation' in hdf and bool(int(hdf['perturbation'][()])))
            if not declared and not has_window:
                return None
            status = (hdf['perturbation_status'][()].decode(errors='replace')
                      if 'perturbation_status' in hdf
                      else ('perturbed' if has_window else 'unknown'))
            if status != 'perturbed':
                return {'status': status, 'type': 'unknown', 'type_known': False,
                        'onset_frame': None, 'duration_ms': None,
                        'duration_source': 'n/a', 'duration_assumed_ms': None,
                        'duration_note': None, 'end_frame': None,
                        'end_known': False, 'frame_rate': None,
                        'movie_key': os.path.basename(os.path.dirname(h5_path)),
                        'source': f'restored from {os.path.basename(h5_path)}',
                        **lighting_info(hdf)}
            kind = hdf['perturbation_type'][()] if 'perturbation_type' in hdf else b'unspecified'
            # An absent duration is how "the log never recorded one" is expressed, so carry the
            # absence through rather than defaulting it to zero -- see utils.PERT_UNKNOWN. The
            # onset is known whenever an experiment is declared perturbed, so it is read flat.
            end_known = (bool(int(hdf['perturbation_end_known'][()]))
                         if 'perturbation_end_known' in hdf else False)
            kind_text = kind.decode() if isinstance(kind, bytes) else str(kind)
            dsrc = hdf['perturbation_duration_source'][()] if 'perturbation_duration_source' in hdf else None
            dsrc = (dsrc.decode() if isinstance(dsrc, bytes) else dsrc) if dsrc is not None else None
            return {
                # Without 'status' the writer would read this as not perturbed and
                # drop the window it is restoring.
                'status': 'perturbed',
                'type_known': kind_text.strip().lower() not in ('', 'unspecified', 'unknown'),
                'duration_source': dsrc or ('recorded' if end_known else 'unrecorded'),
                'movie_key': os.path.basename(os.path.dirname(h5_path)),
                'type': kind_text,
                'onset_frame': int(hdf['perturbation_start_frame'][()]),
                'duration_ms': (float(hdf['perturbation_duration_ms'][()])
                                if end_known and 'perturbation_duration_ms' in hdf else None),
                'end_frame': (int(hdf['perturbation_end_frame'][()])
                              if end_known and 'perturbation_end_frame' in hdf else None),
                'end_known': end_known,
                'source': f'restored from {os.path.basename(h5_path)}',
                **lighting_info(hdf),
            }
    except (OSError, KeyError):
        return None


def saved_source(movie_dir):
    """The provenance keys source.json holds, or None.

    Provenance normally rides in the analysis h5; an h5 left behind by an interrupted analysis
    can lack it, and writing a new one from that would drop it for good. source.json, written
    at predict time, holds the same keys."""
    try:
        with open(os.path.join(movie_dir, SOURCE_JSON), encoding='utf-8') as f:
            saved = json.load(f)
        return {k: str(saved[k]) for k in PROVENANCE_KEYS if saved.get(k)} or None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def recorded_source_box(movie_dir):
    """The source box h5 path a movie recorded at predict time (a cluster path), or None."""
    source = read_existing_context(movie_dir)[2] or saved_source(movie_dir)
    return recorded_box_path(movie_dir, source)


def recorded_box_path(movie_dir, source):
    """The source box h5 recorded at predict time -- a cluster path -- or None.

    From the analysis h5's provenance, else source.json, else a saved member config."""
    box = (source or {}).get('box_h5')
    if not box:
        try:
            with open(os.path.join(movie_dir, SOURCE_JSON), encoding='utf-8') as f:
                box = json.load(f).get('box_h5')
        except (OSError, json.JSONDecodeError):
            box = None
    if not box:
        try:
            box = read_prediction_config(movie_dir)[0]
        except Exception:
            box = None
    return box


def declaration_from_json(movie_dir, source, frame_rate, path_maps=()):
    """The live perturbation.json's declaration for this movie, or None.

    Found through the source movie's recorded path; load_perturbation needs only that path to
    find the experiment dir, not the movie file itself, so off the cluster a mirrored
    perturbation.json under the mapped path is enough. The path it names is mapped back, so
    the products record where the declaration lives on the cluster."""
    box = recorded_box_path(movie_dir, source)
    if not box:
        return None
    perturbation = load_perturbation(map_path(box, path_maps), frame_rate)
    if perturbation is not None and perturbation.get('source'):
        perturbation['source'] = unmap_path(perturbation['source'], path_maps)
    return perturbation


def resolve_context(movie_dir, pert_source='auto', path_maps=()):
    """Everything a movie's re-analysis takes besides its 3D points. Writes nothing.

    Shared by the run and by --dry-run, so the preflight reports exactly what a run would use.
    """
    trigger_offset, frame_rate, source, h5_perturbation, h5_path = read_existing_context(movie_dir)
    source = source or saved_source(movie_dir)

    # Without a trigger every frame number would be a box index, breaking the rule that frame
    # 0 is the camera trigger in every product. An h5 that never recorded it can still be
    # placed from the source movie's sparse mats, when those are reachable.
    trigger = 'analysis h5' if trigger_offset is not None else 'MISSING'
    if trigger_offset is None:
        box = recorded_box_path(movie_dir, source)
        if box:
            recovered_offset, recovered_rate = get_trigger_frame_info(map_path(box, path_maps))
            if recovered_offset is not None:
                trigger_offset, trigger = recovered_offset, 'sparse mat'
                frame_rate = frame_rate or recovered_rate

    # The live declaration wins under 'auto': it is the only place a lighting
    # block added after prediction can come from.
    perturbation, used = h5_perturbation, ('analysis h5' if h5_perturbation else 'none')
    if pert_source in ('auto', 'json'):
        live = declaration_from_json(movie_dir, source, frame_rate, path_maps)
        if live is not None:
            perturbation, used = live, live.get('source') or 'perturbation.json'
        elif pert_source == 'json':
            perturbation, used = None, 'none (no perturbation.json applies)'
    return {'trigger_offset': trigger_offset, 'frame_rate': frame_rate, 'trigger': trigger,
            'source': source, 'perturbation': perturbation, 'declaration': used,
            'h5_path': h5_path}


def archive_previous(movie_dir, stamp, names=None):
    """Move the products this script is about to overwrite into superseded_<stamp>/."""
    if names is None:
        doomed = [f for f in os.listdir(movie_dir)
                  if f.endswith('_analysis_smoothed.h5')
                  or f.endswith('_analysis_smoothed.csv')
                  or f in FIGURE_NAMES
                  or f.endswith(VIEWER_SUFFIX)
                  or f in ('All body data.html', 'movie_html.html')]
    else:
        doomed = [f for f in names if os.path.exists(os.path.join(movie_dir, f))]
    if not doomed:
        return None
    archive_dir = os.path.join(movie_dir, f'superseded_{stamp}')
    os.makedirs(archive_dir, exist_ok=True)
    for name in doomed:
        shutil.move(os.path.join(movie_dir, name), os.path.join(archive_dir, name))
    return archive_dir


def read_analysis_stamp(movie_dir):
    """source.json's "analysis" block, or {}."""
    try:
        with open(os.path.join(movie_dir, SOURCE_JSON), encoding='utf-8') as f:
            return json.load(f).get('analysis') or {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def is_current(movie_dir, code_fp, decl_fp):
    """True when this code, this declaration and these points already made every product on disk.

    The stamp is written only after the last product, so an interrupted movie never matches;
    a product removed by hand afterwards makes the movie stale again too, and so do new 3D
    points -- a stamp from before points_fingerprint existed records none and never matches."""
    stamp = read_analysis_stamp(movie_dir)
    if stamp.get('code_fingerprint') != code_fp or stamp.get('declaration_fingerprint') != decl_fp:
        return False
    if stamp.get('points_fingerprint') != points_fingerprint(movie_dir):
        return False
    names = os.listdir(movie_dir)
    return (any(f.endswith('_analysis_smoothed.h5') for f in names)
            and any(f.endswith('_analysis_smoothed.csv') for f in names)
            and any(f.endswith(VIEWER_SUFFIX) for f in names)
            and all(f in names for f in FIGURE_NAMES))


# FlightAnalysis attributes that the h5 stores as the directory the analysis ran in
RUN_PATH_KEYS = ('dir', 'points_3D_path')


def stamp_h5(h5_path, stamps, path_maps=()):
    """Record which code made an analysis h5 inside the h5, since it is shipped on its own.

    Off the cluster, the run's own location is mapped back too, so a laptop-made h5 names the
    cluster directory of its movie like every other one does."""
    with h5py.File(h5_path, 'a') as hdf:
        for key, value in stamps.items():
            if key in hdf:
                del hdf[key]
            hdf.create_dataset(key, data=_h5_text(value))
        for key in RUN_PATH_KEYS if path_maps else ():
            if key not in hdf:
                continue
            value = hdf[key][()]
            value = value.decode() if isinstance(value, bytes) else str(value)
            recorded = unmap_path(value, path_maps)
            if recorded != value:
                # a plain str, as create_movie_analysis_h5 wrote it, keeps the dataset's dtype
                del hdf[key]
                hdf.create_dataset(key, data=recorded)


def stamp_analysis(movie_dir, info):
    """Write source.json's "analysis" block, atomically, as the movie's last step."""
    path = os.path.join(movie_dir, SOURCE_JSON)
    doc = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding='utf-8') as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            doc = {}
    doc['analysis'] = info
    staged = path + '.partial'
    with open(staged, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=4, default=str)
    os.replace(staged, path)


def previous_h5(archive_dir):
    """The newest readable analysis h5 among a movie's archives, this run's own first.

    What this run moved aside is normally the comparison. When that is an h5 an interrupted
    run left half-written, the one before it is."""
    if not archive_dir:
        return None
    movie_dir = os.path.dirname(archive_dir)
    candidates = sorted(glob.glob(os.path.join(archive_dir, '*_analysis_smoothed.h5')))
    candidates += sorted((path for path in glob.glob(os.path.join(movie_dir, 'superseded_*',
                                                                  '*_analysis_smoothed.h5'))
                          if os.path.dirname(path) != archive_dir), reverse=True)
    for path in candidates:
        try:
            with h5py.File(path, 'r'):
                return path
        except OSError:
            continue
    return None


def compare_with_previous(archive_dir, h5_path):
    """Before/after numbers for the run report, against the h5 this run superseded."""
    previous = previous_h5(archive_dir)
    if previous is None:
        return {}
    out, checks = {}, []
    try:
        with h5py.File(previous, 'r') as old, h5py.File(h5_path, 'r') as new:
            for key in BODY_ANGLE_KEYS:
                if key not in old or key not in new:
                    continue
                # an h5 written before the nose-down convention holds the opposite pitch sign
                a = np.asarray(old[key][()], dtype=float).ravel() * pitch_read_sign(old, key)
                b = np.asarray(new[key][()], dtype=float).ravel() * pitch_read_sign(new, key)
                n = min(a.size, b.size)
                change = np.abs(a[:n] - b[:n])
                if np.isfinite(change).any():
                    out[f"max_abs_change_{key.split('_')[0]}_deg"] = round(float(np.nanmax(change)), 4)

            def wing_nan_fraction(hdf):
                values = [np.asarray(hdf[k][()], dtype=float).ravel() for k in WING_ANGLE_KEYS if k in hdf]
                return round(float(np.mean(np.isnan(np.concatenate(values)))), 4) if values else None

            out['wing_nan_frac_old'] = wing_nan_fraction(old)
            out['wing_nan_frac_new'] = wing_nan_fraction(new)
    except (OSError, KeyError, ValueError) as e:
        out['check'] = f'comparison failed: {e}'
        return out
    if max(out.get('max_abs_change_yaw_deg', 0), out.get('max_abs_change_pitch_deg', 0)) > PITCH_YAW_CHANGE_DEG:
        checks.append('yaw/pitch moved')
    if (out.get('wing_nan_frac_old') is not None and out.get('wing_nan_frac_new') is not None
            and out['wing_nan_frac_new'] > out['wing_nan_frac_old'] + WING_NAN_RISE):
        checks.append('more NaN wing-angle frames')
    out['check'] = '; '.join(checks)
    return out


def reanalyse(movie_dir, stamp, archive=True, with_video=False, force_video=False,
              pert_source='auto', path_maps=(), allow_no_trigger=False, only_stale=False,
              code_fp=None, commit='unknown'):
    """Re-analyse one movie; returns its row for the run report."""
    points_path = os.path.join(movie_dir, POINTS_NAME)
    movie = os.path.basename(movie_dir.rstrip(os.sep))
    ctx = resolve_context(movie_dir, pert_source, path_maps)
    perturbation, trigger_offset, frame_rate = ctx['perturbation'], ctx['trigger_offset'], ctx['frame_rate']
    print(f"  declaration: {ctx['declaration']}"
          + (f" | status {perturbation.get('status')} | lighting "
             f"{perturbation.get('lighting_regime', 'unknown')}" if perturbation else ""),
          flush=True)
    print(f"  trigger: {ctx['trigger']}", flush=True)
    row = {'declaration': ctx['declaration'], 'trigger': ctx['trigger'],
           'lighting': (perturbation or {}).get('lighting_regime', '')}
    if ctx['trigger'] == 'MISSING' and not allow_no_trigger:
        raise NoTrigger('its previous analysis h5 does not record where the camera trigger is, '
                        'and the source movie is not reachable to find it, so its frames could '
                        'not be numbered from the trigger; skipped (--path-map can reach the '
                        'source movie, --allow-no-trigger numbers from box frame 0 instead)')

    code_fp = code_fp or code_fingerprint()
    decl_fp = declaration_fingerprint(perturbation)
    points_fp, realigned = points_fingerprint(movie_dir), realigned_at(movie_dir)
    if only_stale and is_current(movie_dir, code_fp, decl_fp):
        print("  current: already made by this code and this declaration, skipped", flush=True)
        return {**row, 'status': 'current'}

    # build the analysis first, so a movie that fails keeps the products it already had
    analysis = FlightAnalysis(points_3D_path=points_path, find_auto_correlation=True,
                              create_html=False, create_mp4=False, create_h5=False)
    archive_dir = archive_previous(movie_dir, stamp) if archive else None

    analysed_at = dt.datetime.now().isoformat(timespec='seconds')
    h5_path, _ = create_movie_analysis_h5(movie, movie_dir, points_path, smooth=True,
                                          analysis_object=analysis,
                                          trigger_offset=trigger_offset,
                                          frame_rate=frame_rate, source=ctx['source'],
                                          perturbation=perturbation)
    stamps = {'analysis_code_fingerprint': code_fp, 'analysis_git_commit': commit,
              'analysed_at': analysed_at, 'points_fingerprint': points_fp}
    if realigned:
        # so a shipped h5 says its points came from a re-run ensemble, not the predicted one
        stamps['ensemble_realigned_at'] = realigned
    stamp_h5(h5_path, stamps, path_maps)
    export_analysis_csv(analysis, h5_path.replace('.h5', '.csv'), trigger_offset, frame_rate,
                        perturbation=perturbation)
    plot_movie_figures(h5_path, units="frames")
    make_flight_viewer(h5_path)
    stamp_declaration(movie_dir, perturbation)

    video = None
    if with_video:
        # a missing source movie costs the mp4, not the analysis that already succeeded
        try:
            video = regenerate_video(movie_dir, analysis, h5_path, trigger_offset, frame_rate,
                                     stamp=stamp, archive=archive, force=force_video,
                                     perturbation=perturbation, path_maps=path_maps)
        except NoSourceMovie as e:
            video = f'skipped (source movie gone: {e})'
        print(f"  video: {video}", flush=True)

    lo, hi = analysis.first_y_body_frame, analysis.end_frame
    spans = []
    for wing in ('left', 'right'):
        psi = getattr(analysis, f'wings_psi_{wing}')[lo:hi]
        psi = psi[np.isfinite(psi)]
        spans.append(round(float(psi.max() - psi.min()), 1) if psi.size else np.nan)
    print(f"  psi span: left {spans[0]:.0f} deg, right {spans[1]:.0f} deg", flush=True)

    comparison = compare_with_previous(archive_dir, h5_path)
    if comparison.get('check'):
        print(f"  CHECK: {comparison['check']}", flush=True)

    # last, so only a movie whose every product was written counts as current
    stamp_analysis(movie_dir, {'code_fingerprint': code_fp, 'declaration_fingerprint': decl_fp,
                               'points_fingerprint': points_fp, 'ensemble_realigned_at': realigned,
                               'git_commit': commit, 'analysed_at': analysed_at,
                               'host': socket.gethostname(), 'declaration': ctx['declaration'],
                               'trigger': ctx['trigger']})
    return {**row, **comparison, 'status': 'done', 'frames': analysis.num_frames,
            'psi_span_left_deg': spans[0], 'psi_span_right_deg': spans[1], 'video': video or ''}


def run_one(movie_dir, stamp, opts, capture):
    """One movie, never raising: (report row, its console output when captured).

    A parallel worker's prints are held back and handed to the parent, which prints each
    movie's output as one block instead of interleaving every worker's lines."""
    row = {'experiment': os.path.basename(os.path.dirname(movie_dir)),
           'movie': os.path.basename(movie_dir), 'movie_dir': movie_dir}
    buffer = io.StringIO()
    started = time.time()
    with contextlib.ExitStack() as stack:
        if capture:
            stack.enter_context(contextlib.redirect_stdout(buffer))
            stack.enter_context(contextlib.redirect_stderr(buffer))
        try:
            row.update(reanalyse(movie_dir, stamp, **opts))
        except Exception as e:
            row.update(status='failed', error=f'{type(e).__name__}: {e}')
            traceback.print_exc()
    row['seconds'] = round(time.time() - started, 1)
    return row, buffer.getvalue()


def movie_label(movie_dir):
    """experiment/movie: movie numbers repeat across experiments."""
    return f"{os.path.basename(os.path.dirname(movie_dir))}/{os.path.basename(movie_dir)}"


def status_line(row):
    text = f"  -> {row.get('status', '?')} in {row.get('seconds', 0):.0f} s"
    if row.get('error'):
        text += f"  ({row['error']})"
    return text


def preflight_rows(movie_dirs, pert_source, path_maps, code_fp, group_of=None):
    """What a run would use and redo, one row per movie, printed as it goes. Writes nothing.

    group_of names the group a movie is summarised under; by default its parent folder."""
    rows = []
    for movie_dir in movie_dirs:
        name = os.path.basename(movie_dir)
        row = {'movie_dir': movie_dir, 'movie': name,
               'group': group_of(movie_dir) if group_of else os.path.basename(os.path.dirname(movie_dir))}
        rows.append(row)
        try:
            ctx = resolve_context(movie_dir, pert_source, path_maps)
        except Exception as e:
            print(f"  {name:<34} ERROR {type(e).__name__}: {e}")
            row.update(state='error', trigger='?', declaration='?', error=f'{type(e).__name__}: {e}')
            continue
        perturbation = ctx['perturbation']
        state = ('current' if is_current(movie_dir, code_fp, declaration_fingerprint(perturbation))
                 else 'stale')
        declaration = ctx['declaration']
        kind = ('perturbation.json' if declaration.endswith('perturbation.json') else declaration)
        detail = (f"{perturbation.get('status')}, {perturbation.get('lighting_regime', 'unknown')}"
                  if perturbation else 'nothing declared')
        print(f"  {name:<34} trigger {ctx['trigger']:<11} {state:<7} declaration {kind} ({detail})")
        row.update(state=state, trigger=ctx['trigger'], declaration=kind)
    return rows


def print_preflight_summary(rows):
    def counts(key, subset):
        return ', '.join(f'{k} {v}' for k, v in sorted(Counter(r[key] for r in subset).items()))

    print('\npreflight summary')
    for group in sorted({r['group'] for r in rows}):
        subset = [r for r in rows if r['group'] == group]
        print(f"  {group}: {len(subset)} movie(s)")
        print(f"    to redo     : {counts('state', subset)}")
        print(f"    trigger     : {counts('trigger', subset)}")
        print(f"    declaration : {counts('declaration', subset)}")
    missing = sum(r['trigger'] == 'MISSING' for r in rows)
    if missing:
        print(f"\n{missing} movie(s) have no trigger and would FAIL; see --path-map / --allow-no-trigger")


def preflight(movie_dirs, pert_source, path_maps, code_fp):
    """--dry-run: what a run would use and redo, per movie and per experiment. Writes nothing."""
    print_preflight_summary(preflight_rows(movie_dirs, pert_source, path_maps, code_fp))
    return 0


def write_report(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.get('experiment', ''), r.get('movie', ''))):
            writer.writerow({k: ('' if row.get(k) is None else row.get(k)) for k in REPORT_FIELDS})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dirs', nargs='+', help='movie dir, or any dir above movie dirs')
    parser.add_argument('--dry-run', action='store_true',
                        help='preflight: per movie, where the trigger and declaration would come '
                             'from and whether it is stale; writes nothing')
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
    parser.add_argument('--perturbation-source', choices=('auto', 'json', 'h5'), default='auto',
                        help="where the declaration (pulse window + lighting) comes from: "
                             "'json' the experiment's live perturbation.json, 'h5' what the "
                             "existing analysis h5 recorded, 'auto' (default) the json when "
                             "one applies, else the h5")
    parser.add_argument('--jobs', type=int, default=1,
                        help='movies re-analysed at once, one core each (default 1). each needs '
                             '~1-1.5 GB of memory')
    parser.add_argument('--only-stale', action='store_true',
                        help='skip movies whose products were already made by this analysis '
                             'code and this declaration; re-running a command then resumes it')
    parser.add_argument('--path-map', action='append', metavar='SERVER=LOCAL',
                        help='where a cluster path prefix recorded at predict time lives on this '
                             'machine, e.g. /cs/labs/tsevi/lior.kotlar/pose-estimation-torch='
                             'E:\\pose-estimation-torch (repeatable)')
    parser.add_argument('--allow-no-trigger', action='store_true',
                        help='re-analyse a movie whose trigger cannot be found, numbering its '
                             'frames from box frame 0 (refused by default)')
    parser.add_argument('--report', metavar='CSV',
                        help='where to write the run report (default: '
                             'reanalyse_report_<timestamp>.csv in the first dir, when that dir '
                             'is above the movies)')
    args = parser.parse_args()
    path_maps = parse_path_maps(args.path_map)

    movie_dirs = []
    for root in args.dirs:
        movie_dirs.extend(find_movie_dirs(root))
    movie_dirs = sorted(set(movie_dirs))
    if not movie_dirs:
        print(f"no movie dirs with a {POINTS_NAME} under: {', '.join(args.dirs)}")
        return 1

    code_fp, commit = code_fingerprint(), git_commit()
    print(f"{len(movie_dirs)} movie(s) | analysis code {code_fp} ({commit})", flush=True)
    for server, local in path_maps:
        print(f"path map: {server} -> {local}", flush=True)
    if args.dry_run:
        return preflight(movie_dirs, args.perturbation_source, path_maps, code_fp)

    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    opts = dict(archive=not args.no_archive, with_video=args.with_mp4,
                force_video=args.force_mp4, pert_source=args.perturbation_source,
                path_maps=path_maps, allow_no_trigger=args.allow_no_trigger,
                only_stale=args.only_stale, code_fp=code_fp, commit=commit)
    rows = run_movies(movie_dirs, stamp, opts, args.jobs)

    report = args.report
    first = os.path.abspath(args.dirs[0])
    if report is None and not os.path.exists(os.path.join(first, POINTS_NAME)):
        report = os.path.join(first, f'reanalyse_report_{stamp}.csv')
    if report:
        print(f"\nreport: {write_report(rows, report)}")
    return print_run_summary(rows)


def run_movies(movie_dirs, stamp, opts, jobs=1):
    """Re-analyse movie_dirs, jobs at a time; returns one report row per movie."""
    rows = []
    total = len(movie_dirs)
    if jobs <= 1:
        for i, movie_dir in enumerate(movie_dirs, 1):
            print(f"\n[{i}/{total}] {movie_label(movie_dir)}", flush=True)
            row, _ = run_one(movie_dir, stamp, opts, capture=False)
            print(status_line(row), flush=True)
            rows.append(row)
    else:
        # set before the pool starts: spawned workers read them when they import numpy
        for var in BLAS_THREAD_VARS:
            os.environ.setdefault(var, '1')
        # spawn on every platform: it is the only start method on Windows, and forking a
        # parent that already holds h5py and matplotlib state is not safe on Linux either
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
            futures = {pool.submit(run_one, d, stamp, opts, True): d for d in movie_dirs}
            for i, future in enumerate(as_completed(futures), 1):
                movie_dir = futures[future]
                try:
                    row, output = future.result()
                except Exception as e:
                    # a worker killed outright (out of memory, say) leaves no row behind
                    row = {'experiment': os.path.basename(os.path.dirname(movie_dir)),
                           'movie': os.path.basename(movie_dir), 'movie_dir': movie_dir,
                           'status': 'failed', 'error': f'worker died: {type(e).__name__}: {e}'}
                    output = ''
                print(f"\n[{i}/{total}] {movie_label(movie_dir)}\n{output.rstrip()}", flush=True)
                print(status_line(row), flush=True)
                rows.append(row)
    return rows


def print_run_summary(rows):
    """Counts, failures and flags at the end of a run; 1 when any movie failed."""
    by_status = Counter(row.get('status') for row in rows)
    print(f"\n{len(rows)} movie(s): " + ', '.join(f'{k} {v}' for k, v in sorted(by_status.items())))
    for row in rows:
        if row.get('status') == 'failed':
            print(f"  FAILED {row['experiment']}/{row['movie']}: {row.get('error')}")
    flagged = [row for row in rows if row.get('check')]
    if flagged:
        print(f"\n{len(flagged)} movie(s) to look at against their previous h5:")
        for row in flagged:
            print(f"  {row['experiment']}/{row['movie']}: {row['check']}")
    no_video = [row['movie'] for row in rows if str(row.get('video', '')).startswith('skipped')]
    if no_video:
        print(f"\n{len(no_video)} movie(s) kept their old mp4 because the source movie is gone;"
              f" their angles are still up to date:")
        for name in no_video:
            print(f"  {name}")
    return 1 if by_status.get('failed') else 0


if __name__ == '__main__':
    sys.exit(main())
