"""Render a movie's overlay video ("movie 2D and 3D.mp4") in pieces, then join them.

A long movie's render takes hours (about 20-25 min per 1000 frames), and done at the end of a
prediction job it can run into the job's time limit -- mov_72 (6204 frames) was cut off at
frame ~3500 that way. Rendered as a SLURM array, each task draws one contiguous stretch of
frames into its own file; a second job joins them (ffmpeg's concat demuxer, no re-encoding)
and puts the result where reanalyse_movies.py --only-video would have, with the same stamp.

Everything is read back out of the movie's analysis h5 and its members' saved config, exactly as
reanalyse_movies.render_only does: the analysed 3D points, the trigger, the declaration, the
source box and the calibration. Each piece reprojects the points itself (seconds), so the pieces
need nothing from each other; the join checks they all reprojected the same.

    # one piece per array task (index and count from SLURM_ARRAY_TASK_ID / _COUNT):
    sbatch --array=0-7 -p glacier --gres=gpu:0 --mem=32g --cpus-per-task=2 --time=04:00:00 \\
        -J video_pieces sbatch_files/sbatch_configurable.sh code/render_video_pieces.py piece <movie_dir>
    # then, once they have all finished:
    sbatch --dependency=afterok:<array job id> -p glacier --gres=gpu:0 --mem=8g --cpus-per-task=1 \\
        --time=00:30:00 -J video_join sbatch_files/sbatch_configurable.sh \\
        code/render_video_pieces.py join <movie_dir> --pieces 8

The pieces are kept in <movie_dir>/.video_pieces/ (a dot directory, so no movie walk visits it)
until the join succeeds. The previous video and reprojection are moved into superseded_<stamp>/.
Only movies analysed from their first frame (first_analysed_frame 0) are supported.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_code_lior'))
import reanalyse_movies as rm  # noqa: E402
from Triangulator import Triangulator  # noqa: E402
from Visualizer import Visualizer  # noqa: E402

PIECES_DIR = '.video_pieces'


def piece_name(index, pieces, suffix):
    return f'piece_{index:03d}_of_{pieces:03d}{suffix}'


def context(movie_dir):
    """The analysis h5's points, trigger and declaration, and the source box and calibration."""
    h5_path = rm.analysis_h5(movie_dir)
    if not h5_path:
        sys.exit(f'{movie_dir}: no analysis h5 to render from')
    with h5py.File(h5_path, 'r') as hdf:
        points = hdf['points_3D'][()]
        first = int(hdf['first_analysed_frame'][()]) if 'first_analysed_frame' in hdf else 0
        trigger_offset = int(hdf['trigger_offset'][()]) if 'trigger_offset' in hdf else None
        frame_rate = float(hdf['frame_rate'][()]) if 'frame_rate' in hdf else None
    if first != 0:
        sys.exit(f'{movie_dir}: first_analysed_frame is {first}; only movies analysed from frame 0 '
                 'can be rendered in pieces')
    recorded_box, recorded_calibration, height, width = rm.read_prediction_config(movie_dir)
    box_path, calibration_path = rm.reachable(recorded_box), rm.reachable(recorded_calibration)
    if not os.path.isfile(box_path):
        sys.exit(f'{movie_dir}: source box not found: {box_path}')
    return dict(h5_path=h5_path, points=points, trigger_offset=trigger_offset, frame_rate=frame_rate,
                perturbation=rm.read_perturbation(h5_path), recorded_box=recorded_box, box_path=box_path,
                calibration_path=calibration_path, height=height, width=width)


def reproject(ctx):
    with h5py.File(ctx['box_path'], 'r') as box:
        cropzone = box['/cropzone'][:] if '/cropzone' in box else box['/cropZone'][:]
    return Triangulator(ctx['calibration_path'], ctx['height'], ctx['width']).get_reprojections(
        ctx['points'], cropzone)


def task(value, variable):
    if value is not None:
        return value
    if variable not in os.environ:
        sys.exit(f'give --{variable.split("_")[-1].lower()} or run as a SLURM array ({variable})')
    return int(os.environ[variable])


def piece(movie_dir, pieces, index):
    ctx = context(movie_dir)
    frames = np.array_split(np.arange(len(ctx['points'])), pieces)[index]
    work = os.path.join(movie_dir, PIECES_DIR)
    os.makedirs(work, exist_ok=True)
    points_path = os.path.join(work, piece_name(index, pieces, '_reprojected.npy'))
    np.save(points_path, reproject(ctx))
    staged = os.path.join(work, piece_name(index, pieces, '.partial.mp4'))
    print(f'{movie_dir}: piece {index + 1} of {pieces}, box frames {frames[0]}-{frames[-1]} '
          f'({len(frames)} frames)', flush=True)
    Visualizer.create_movie_mp4(ctx['h5_path'], save_frames=frames, mode='SAVE',
                                reprojected_points_path=points_path, box_path=ctx['box_path'],
                                save_path=staged, rotate=True, trigger_offset=ctx['trigger_offset'],
                                frame_rate=ctx['frame_rate'], perturbation=ctx['perturbation'])
    os.replace(staged, os.path.join(work, piece_name(index, pieces, '.mp4')))
    print('done', flush=True)


def count_frames(mp4):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                          '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', mp4],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def join(movie_dir, pieces):
    ctx = context(movie_dir)
    work = os.path.join(movie_dir, PIECES_DIR)
    parts = [os.path.join(work, piece_name(k, pieces, '.mp4')) for k in range(pieces)]
    missing = [p for p in parts if not os.path.isfile(p)]
    if missing:
        sys.exit(f'{len(missing)} of {pieces} pieces are missing, e.g. {missing[0]}')
    reprojections = [np.load(os.path.join(work, piece_name(k, pieces, '_reprojected.npy'))) for k in range(pieces)]
    if not all(np.array_equal(reprojections[0], r, equal_nan=True) for r in reprojections[1:]):
        sys.exit('the pieces reprojected the points differently; not joining')
    expected = [len(f) for f in np.array_split(np.arange(len(ctx['points'])), pieces)]
    got = [count_frames(p) for p in parts]
    if got != expected:
        sys.exit(f'frames per piece {got}, expected {expected}')
    listing = os.path.join(work, 'pieces.txt')
    with open(listing, 'w') as fh:
        fh.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    joined = os.path.join(work, 'joined.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', listing, '-c', 'copy',
                    joined], check=True)
    if count_frames(joined) != len(ctx['points']):
        sys.exit(f'the joined video has {count_frames(joined)} frames, the movie {len(ctx["points"])}')
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    archived = rm.archive_previous(movie_dir, stamp, names=(rm.REPROJECTED_NAME, rm.MP4_NAME))
    shutil.move(joined, os.path.join(movie_dir, rm.MP4_NAME))
    np.save(os.path.join(movie_dir, rm.REPROJECTED_NAME), reprojections[0])
    video = rm.write_video_stamp(movie_dir, ctx['h5_path'], ctx['recorded_box'])
    with open(os.path.join(movie_dir, rm.VIDEO_STAMP), 'w', encoding='utf-8') as fh:
        json.dump({**video, 'rendered_in_pieces': pieces}, fh, indent=1)
    shutil.rmtree(work)
    print(f'{movie_dir}: joined {pieces} pieces, {len(ctx["points"])} frames -> {rm.MP4_NAME}'
          + (f'; the previous video is in {os.path.basename(archived)}/' if archived else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('piece', help='render one contiguous piece of the movie')
    p.add_argument('movie_dir')
    p.add_argument('--pieces', type=int, help='default: SLURM_ARRAY_TASK_COUNT')
    p.add_argument('--index', type=int, help='0-based; default: SLURM_ARRAY_TASK_ID')
    j = sub.add_parser('join', help='join the pieces into the movie\'s video')
    j.add_argument('movie_dir')
    j.add_argument('--pieces', type=int, required=True)
    args = ap.parse_args()
    if args.command == 'piece':
        piece(args.movie_dir, task(args.pieces, 'SLURM_ARRAY_TASK_COUNT'), task(args.index, 'SLURM_ARRAY_TASK_ID'))
    else:
        join(args.movie_dir, args.pieces)


if __name__ == '__main__':
    main()
