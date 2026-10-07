"""What a movie's prediction run keeps once its prediction is done.

During a prediction every ensemble member writes its own outputs into the run: its 2D points,
its 3D candidates from every camera pair and its copy of the crop record. They are what the
ensemble combines, and once the ensemble and the analysis are written nothing reads them
again -- yet they are half of what a run holds (about 66 of 134 kB per frame on the
four-camera rig). So by default a finished run drops them:

    removed    every member's points_3D_all.npy, points_3D.npy, points_3D_smoothed.npy, and
               every member's predicted_points_and_box.h5 but one
    kept       every member's small files (configuration.json, specific_configuration.json,
               README_scores_3D.txt), so the run still says which members ran and how; ONE
               member's crop record (predicted_points_and_box.h5), which reanalysis and
               rendering read the box and calibration back out of; everything else the run
               made: the analysis, the 3D points, the ensemble's records, the overlay video,
               the HTML pages. Recorded in member_outputs.json

KEEP_MEMBER_OUTPUTS=1 on predict_array.sh (config "keep member outputs": true) keeps them all.
A run whose member outputs are gone can not be realigned (code/realign_ensemble.py re-runs the
ensemble from them); predictions made since Predictor2D.harmonize_wing_labels do not need it.

A LEAN run (off by default; LEAN_RUN=1, config "lean run": true) goes further, for when disk is
short: about 20 kB per frame.

    not made   the overlay video, movie_html.html, All body data.html, the flight viewer,
               model_selection_visualizations/
    compressed all_frames_scores.json -> all_frames_scores.json.gz
    recorded   in lean_run.json, which reanalyse_movies.py reads

Videos and viewers are made for the lean movies asked for:
    .env/bin/python code/reanalyse_movies.py <movie dir> --with-viewers [--with-mp4]
"""
import datetime as dt
import gzip
import json
import os
import shutil

LEAN_MARKER = 'lean_run.json'
MEMBERS_MARKER = 'member_outputs.json'
SCORES_NAME = 'all_frames_scores.json'
CROP_RECORD = 'predicted_points_and_box.h5'
MEMBER_HEAVY = ('points_3D_all.npy', 'points_3D.npy', 'points_3D_smoothed.npy')
MEMBER_CONFIGS = ('configuration.json', 'specific_configuration.json')


def is_lean(movie_dir):
    return os.path.isfile(os.path.join(movie_dir, LEAN_MARKER))


def member_dirs(movie_dir):
    """The run's ensemble member folders: the subfolders holding a member's saved config."""
    out = []
    for name in sorted(os.listdir(movie_dir)):
        path = os.path.join(movie_dir, name)
        if os.path.isdir(path) and any(os.path.isfile(os.path.join(path, c)) for c in MEMBER_CONFIGS):
            out.append(path)
    return out


def write_scores(path, payload):
    """all_frames_scores, gzip-compressed; returns the path written."""
    gz = path + '.gz'
    with gzip.open(gz + '.partial', 'wt', encoding='utf-8') as fh:
        json.dump(payload, fh)
    os.replace(gz + '.partial', gz)
    return gz


def compress_scores(movie_dir):
    """Replace an all_frames_scores.json by its .gz, checked by reading it back."""
    path = os.path.join(movie_dir, SCORES_NAME)
    if not os.path.isfile(path):
        return None
    gz = path + '.gz'
    with open(path, 'rb') as src, gzip.open(gz + '.partial', 'wb') as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    with open(path, 'rb') as a, gzip.open(gz + '.partial', 'rb') as b:
        while True:
            x, y = a.read(1 << 20), b.read(1 << 20)
            if x != y:
                raise IOError(f'{gz}: the compressed copy does not read back the same')
            if not x:
                break
    os.replace(gz + '.partial', gz)
    os.remove(path)
    return gz


def slim_members(movie_dir):
    """Drop the members' own outputs from a finished run, keeping one crop record; returns what
    member_outputs.json records."""
    members = member_dirs(movie_dir)
    kept = next((m for m in members if os.path.isfile(os.path.join(m, CROP_RECORD))), None)
    if kept is None:
        raise FileNotFoundError(f'{movie_dir}: no member keeps a crop record; nothing removed')
    removed, freed = [], 0
    for member in members:
        doomed = list(MEMBER_HEAVY) + ([] if member == kept else [CROP_RECORD])
        for name in doomed:
            path = os.path.join(member, name)
            if os.path.isfile(path):
                freed += os.path.getsize(path)
                os.remove(path)
                removed.append(os.path.relpath(path, movie_dir))
    record = dict(crop_record=os.path.relpath(os.path.join(kept, CROP_RECORD), movie_dir),
                  members=len(members), files_removed=len(removed), bytes_removed=freed,
                  removed_at=dt.datetime.now().isoformat(timespec='seconds'))
    with open(os.path.join(movie_dir, MEMBERS_MARKER), 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1)
    print(f'member outputs: kept {record["crop_record"]}; removed {len(removed)} member files '
          f'({freed / 1e6:.0f} MB)', flush=True)
    return record


def slim(movie_dir):
    """Make a finished run lean: the members' outputs dropped and the scores compressed; returns
    what lean_run.json records."""
    members = slim_members(movie_dir)
    scores = compress_scores(movie_dir)
    record = dict(lean=True, crop_record=members['crop_record'],
                  scores=os.path.basename(scores) if scores else SCORES_NAME + '.gz',
                  not_made=['movie 2D and 3D.mp4', 'movie_html.html', 'All body data.html',
                            '*_flight_viewer.html', 'model_selection_visualizations/'],
                  slimmed_at=dt.datetime.now().isoformat(timespec='seconds'))
    with open(os.path.join(movie_dir, LEAN_MARKER), 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1)
    print(f'lean run: scores {record["scores"]}', flush=True)
    return record
