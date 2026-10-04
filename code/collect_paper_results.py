"""Gather the files a paper needs out of this repo's prediction and training
outputs into one self-contained folder, and turn the numbers that are currently
locked inside JSON into CSVs that can be cited.

Why this exists. The pipeline's outputs are shaped for the pipeline: per movie a
directory with one subdirectory per ensemble member, a selection report in JSON,
a `.realigned_ensemble.json` holding the before/after metrics of the wing-label
fix, and several gigabytes of weights, boxes and per-epoch debug images mixed in
with the few small files that actually say what happened. A paper cannot cite
that. It can cite a file. So this script does two things:

  1. copies a defined, tiered subset of the outputs into one destination folder,
     never touching the source tree, and
  2. writes the derived tables next to them -- pooled model-selection
     percentages, per-member scores, winning-subset sizes, a one-row-per-movie
     index and a one-row-per-training-run index -- so every number the paper
     quotes exists on disk with a header naming it.

Everything large and reproducible is left behind: `*.pt` weights, `weights/`,
`viz_pred/`, each member's `predicted_points_and_box.h5`, the per-epoch
histogram series apart from the best epoch, and the `superseded_*/` archives of
earlier runs.

Three tiers, because the three uses have different sizes:

  index     every movie, small text only -- provenance, the selection report,
            the ensemble and per-member scores. Tens of MB for the whole tree.
  analysis  index plus each movie's `*_analysis_smoothed.csv` (the kinematics a
            reader would re-plot) and `all_models_combinations.npy` (the raw
            per-frame selection mask the summaries are computed from).
  showcase  analysis plus the heavy per-movie material -- the analysis h5, the
            3D point arrays, the per-member 3D points, the full score margins,
            the figures, the mp4 and the flight viewer -- for the handful of
            movies named in --showcase-file that the paper shows as figures.

Training is always collected: per run the configuration, the full loss history,
the best-epoch record, the train/val split, the two figures for the best epoch
only, and the `train.py` that produced it (which is how the split policy of that
run is evidenced). Plus `prediction_models/<name>/model.json` for every
registered ensemble member, so the mapping from a training run to the member
index used in the selection report is in the bundle.

What is a movie directory. Any directory holding `source.json`, or holding a
subdirectory that looks like an ensemble member (`points_3D_all.npy` or
`configuration.json`). The walk stops there, so the nesting above it is free:
`predict_output/<experiment>/<movie>` and
`predict_output/debug_outputs/<experiment>/<movie>` both work, and
`<experiment>` in the output is whatever path leads to the movie. Movies under
`bad_wings/` or `bad_signal/` are found but not collected; they are listed in
skipped.csv with the reason, as is a movie with no `source.json`, no member
directories, or an unreadable one. So is a movie under a folder whose name says
SUPERSEDED: that is a whole earlier prediction run of movies a later run redid
(`Tsory_ex241220_20260823_SUPERSEDED_null_duration` holds the same 28 movies as
`ex201224_light_roll_t0`, with the null declaration the rerun fixed), so
collecting both would double-count them. `--include-superseded-runs` takes them
anyway, and `--exclude NAME` drops any other folder.

Outputs:

    <dest>/PROVENANCE.md                  how, when, from which commit
    <dest>/manifest.csv                   every copied file, size, mtime, sha256
    <dest>/index/movies_index.csv         one row per movie
    <dest>/index/selection_by_group.csv   (movie, joint-group, member) long form
    <dest>/index/member_scores.csv        per-member 3D consistency scores
    <dest>/index/subset_sizes.csv         size distribution of winning subsets
    <dest>/index/training_index.csv       one row per training run
    <dest>/index/skipped.csv              every movie not collected, and why
    <dest>/predictions/<experiment>/<movie>/...
    <dest>/training/<run_name>/...
    <dest>/models/<member_name>/model.json

Usage:
    .env/bin/python code/collect_paper_results.py --dest DIR [--tier index|analysis|showcase]
        [--experiments-root predict_output] [--showcase-file FILE] [--models-file FILE]
        [--with-superseded] [--dry-run] [--force] [--hash]

Examples:
    # what would the small tier cost, over everything?
    .env/bin/python code/collect_paper_results.py --dest /tmp/paper --dry-run --tier index

    # the real thing for one experiment
    .env/bin/python code/collect_paper_results.py --dest paper_bundle \
        --experiments-root predict_output/ex201224_dark_roll_t0_7ms \
        --tier showcase --showcase-file showcase.txt

`--showcase-file` and `--models-file` are one name per line, `#` comments and
blank lines ignored. A showcase line matches a movie by its directory name, by
`<experiment>/<movie>`, or by a path ending in either. A models line matches a
training run by its directory name or by a path ending in it.

Resumable: a destination file whose size and mtime already match the source is
left alone, so an interrupted run is finished by repeating the command; --force
re-copies. --hash adds a sha256 per file to the manifest (it reads every
collected file, so it is slower). Directories are walked in sorted order, so two
runs over an unchanged tree produce byte-identical CSVs.

h5py is optional. Without it the columns read out of the analysis h5
(`n_frames_valid_wing_angles`) are left empty and the run says so once.
"""
import argparse
import csv
import datetime as dt
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys

import numpy as np

try:
    import h5py
except ImportError:  # optional: only the wing-angle validity column needs it
    h5py = None

# Joint-group order of the leading axis of all_models_combinations.npy. Must
# match ENSEMBLE_GROUP_NAMES in code/prediction_code_lior/predict.py.
GROUPS = ["left_wing", "right_wing", "head_tail", "side_points"]

# Movie directories under these never ship (see the bad_signal memo: they are
# never handed to a consumer, so they must not reach a paper bundle either).
BAD_DIRS = {"bad_wings", "bad_signal"}

# A whole prediction run that a later one replaced is marked by SUPERSEDED in its
# folder name, the same way a single replaced product goes into superseded_*/.
# Such a folder holds the same movies as the run that replaced it -- collecting
# both would double-count them -- so it is skipped unless asked for.
SUPERSEDED_RUN = re.compile("SUPERSEDED", re.IGNORECASE)

# Never copied, whatever the tier: reproducible bulk.
NEVER_DIR_NAMES = {"weights", "viz_pred"}
NEVER_FILE_NAMES = {"predicted_points_and_box.h5"}
NEVER_FILE_SUFFIXES = (".pt",)
ARCHIVE_PREFIX = "superseded_"

TIERS = ("index", "analysis", "showcase")

# Per movie, by tier. Names are exact; GLOB_* are fnmatch-style patterns.
MOVIE_FILES_INDEX = [
    "source.json",
    ".realigned_ensemble.json",
    "ensemble_model_selection_summary.json",
    "ensemble_model_selection_summary.txt",
    "model_index_legend.json",
    "README_scores_3D_ensemble.txt",
]
MEMBER_FILES_INDEX = ["README_scores_3D.txt", "configuration.json"]

MOVIE_FILES_ANALYSIS = ["all_models_combinations.npy"]
MOVIE_GLOBS_ANALYSIS = ["*_analysis_smoothed.csv"]

MOVIE_FILES_SHOWCASE = [
    "points_3D_ensemble_best_method.npy",
    "points_3D_smoothed_ensemble_best_method.npy",
    "all_frames_scores.json",
    "wing_angles.png",
    "body_angular_acceleration.png",
    "movie 2D and 3D.mp4",
]
MOVIE_GLOBS_SHOWCASE = [
    "*_analysis_smoothed.h5",
    "points_ensemble_*reprojected*.npy",
    "*_flight_viewer.html",
]
MOVIE_DIRS_SHOWCASE = ["model_selection_visualizations"]
MEMBER_FILES_SHOWCASE = ["points_3D_all.npy"]

# The two files taken out of the newest superseded_ensemble_* under
# --with-superseded: the before state of the wing-label fix.
PRE_REALIGN_FILES = ["ensemble_model_selection_summary.json",
                     "README_scores_3D_ensemble.txt"]

TRAIN_FILES = ["configuration.json", "history.csv", "best_model_info.txt",
               "train_val_split.npz"]

CATEGORIES = ["index", "analysis", "showcase", "pre_realign", "training", "models"]


# ---------------------------------------------------------------- small readers

def read_json(path):
    """(obj, error_string). error_string is None on success."""
    try:
        with open(path) as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)


def npy_shape(path):
    """Shape of a .npy without reading its data, or None."""
    try:
        with open(path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(fh)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(fh)
            else:
                return None
        return tuple(shape)
    except Exception:
        return None


SCORE_RAW_RE = re.compile(r"score for the points was\s+(\S+)")
SCORE_SMOOTH_RE = re.compile(r"score for the smoothed points was\s+(\S+)")


def read_scores_readme(path):
    """(score_raw, score_smoothed) as strings, '' when absent."""
    try:
        with open(path) as fh:
            text = fh.read()
    except Exception:
        return "", ""
    raw = SCORE_RAW_RE.search(text)
    smooth = SCORE_SMOOTH_RE.search(text)
    return (raw.group(1) if raw else "", smooth.group(1) if smooth else "")


def count_csv_rows(path):
    """Data rows in a CSV (newlines minus the header), or '' if unreadable."""
    try:
        total = 0
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                total += chunk.count(b"\n")
        return max(total - 1, 0)
    except Exception:
        return ""


def member_name_from_legend(label):
    """'per_cam_dil3 (per_cam_dil3)' -> 'per_cam_dil3' (the member directory)."""
    return label.split(" (", 1)[0].strip()


def fmt(value):
    """Cell value: '' for None, plain repr for numbers, str otherwise."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def load_name_list(path):
    if not path:
        return None
    names = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line.rstrip("/"))
    return names


def name_matches(candidates, experiment, movie):
    """True when a --showcase-file / --models-file line names this movie/run."""
    rel = "%s/%s" % (experiment, movie) if experiment else movie
    for want in candidates:
        want = want.replace(os.sep, "/").rstrip("/")
        if want == movie or want == rel:
            return True
        if want.endswith("/" + movie) or want.endswith("/" + rel):
            return True
    return False


# ------------------------------------------------------------------- discovery

def is_member_dir(path):
    return (os.path.isfile(os.path.join(path, "points_3D_all.npy"))
            or os.path.isfile(os.path.join(path, "configuration.json")))


def member_dirs(movie_dir):
    """Ensemble-member subdirectory names of a movie directory, sorted.

    Sorted order is the model index used throughout the selection bookkeeping
    (predict.py builds the ensemble from sorted(glob(...))).
    """
    out = []
    try:
        names = sorted(os.listdir(movie_dir))
    except OSError:
        return out
    for name in names:
        if name.startswith(".") or name.startswith(ARCHIVE_PREFIX):
            continue
        if name in ("model_selection_visualizations", "pre_realign"):
            continue
        if name in NEVER_DIR_NAMES:
            continue
        path = os.path.join(movie_dir, name)
        if os.path.isdir(path) and is_member_dir(path):
            out.append(name)
    return out


def looks_like_movie_dir(path):
    return os.path.isfile(os.path.join(path, "source.json")) or bool(member_dirs(path))


def find_movie_dirs(root):
    """Every movie directory under root, in sorted order. Stops descending at
    one, and never descends into an archive, a hidden directory or the bulk
    directories that are never copied."""
    found = []
    for dirpath, dirnames, _files in os.walk(root, topdown=True):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".")
                             and not d.startswith(ARCHIVE_PREFIX)
                             and d not in NEVER_DIR_NAMES)
        if os.path.abspath(dirpath) != os.path.abspath(root) and looks_like_movie_dir(dirpath):
            found.append(dirpath)
            dirnames[:] = []
    return found


def newest_superseded_ensemble(movie_dir, realign):
    """Path of the archive holding the pre-realignment ensemble, or None.

    Prefers the archive named by .realigned_ensemble.json; otherwise the newest
    superseded_ensemble_* by name (they are timestamped).
    """
    if isinstance(realign, dict):
        archive = realign.get("archive")
        if archive and os.path.isdir(archive):
            return archive
        if archive:
            local = os.path.join(movie_dir, os.path.basename(archive))
            if os.path.isdir(local):
                return local
    try:
        names = sorted(d for d in os.listdir(movie_dir)
                       if d.startswith("superseded_ensemble_")
                       and os.path.isdir(os.path.join(movie_dir, d)))
    except OSError:
        return None
    return os.path.join(movie_dir, names[-1]) if names else None


# ------------------------------------------------------------------ copy engine

class PlanItem(object):
    __slots__ = ("category", "src", "dest_rel", "size", "mtime")

    def __init__(self, category, src, dest_rel, size, mtime):
        self.category = category
        self.src = src
        self.dest_rel = dest_rel
        self.size = size
        self.mtime = mtime


def forbidden_reason(src, allow_archive=False):
    """Why this source path must never be copied, or None."""
    parts = os.path.abspath(src).split(os.sep)
    base = parts[-1]
    for part in parts[:-1]:
        if part.startswith(ARCHIVE_PREFIX) and not allow_archive:
            return "inside %s" % part
        if part in NEVER_DIR_NAMES:
            return "inside %s/" % part
    if base in NEVER_FILE_NAMES:
        return "excluded file %s" % base
    if base.endswith(NEVER_FILE_SUFFIXES):
        return "model weights"
    return None


class Collector(object):
    def __init__(self, dest, dry_run=False, force=False, do_hash=False):
        self.dest = os.path.abspath(dest)
        self.dry_run = dry_run
        self.force = force
        self.do_hash = do_hash
        self.plan = []
        self.seen_dest = {}
        self.refused = []
        self.copied = 0
        self.reused = 0

    # -- planning

    def add_file(self, category, src, dest_rel, allow_archive=False):
        if not os.path.isfile(src):
            return False
        reason = forbidden_reason(src, allow_archive)
        if reason:
            self.refused.append((src, reason))
            return False
        dest_rel = dest_rel.replace(os.sep, "/")
        if dest_rel in self.seen_dest:
            return True
        try:
            st = os.stat(src)
        except OSError:
            return False
        item = PlanItem(category, os.path.abspath(src), dest_rel, st.st_size, st.st_mtime)
        self.seen_dest[dest_rel] = item
        self.plan.append(item)
        return True

    def add_glob(self, category, src_dir, pattern, dest_dir_rel):
        import fnmatch
        try:
            names = sorted(os.listdir(src_dir))
        except OSError:
            return 0
        n = 0
        for name in names:
            if fnmatch.fnmatch(name, pattern) and os.path.isfile(os.path.join(src_dir, name)):
                if self.add_file(category, os.path.join(src_dir, name),
                                 "%s/%s" % (dest_dir_rel, name)):
                    n += 1
        return n

    def add_dir(self, category, src_dir, dest_dir_rel):
        if not os.path.isdir(src_dir):
            return 0
        n = 0
        for dirpath, dirnames, filenames in os.walk(src_dir, topdown=True):
            dirnames[:] = sorted(d for d in dirnames
                                 if not d.startswith(ARCHIVE_PREFIX)
                                 and d not in NEVER_DIR_NAMES)
            rel = os.path.relpath(dirpath, src_dir)
            for name in sorted(filenames):
                sub = name if rel == "." else "%s/%s" % (rel.replace(os.sep, "/"), name)
                if self.add_file(category, os.path.join(dirpath, name),
                                 "%s/%s" % (dest_dir_rel, sub)):
                    n += 1
        return n

    # -- execution

    def run(self, progress_every=50):
        total = len(self.plan)
        for i, item in enumerate(self.plan, 1):
            dest = os.path.join(self.dest, item.dest_rel)
            parent = os.path.dirname(dest)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            if not self.force and os.path.exists(dest):
                st = os.stat(dest)
                if st.st_size == item.size and abs(st.st_mtime - item.mtime) < 2:
                    self.reused += 1
                    if i % progress_every == 0 or i == total:
                        print("  [%d/%d] %s" % (i, total, item.dest_rel), flush=True)
                    continue
            shutil.copy2(item.src, dest)
            self.copied += 1
            if i % progress_every == 0 or i == total:
                print("  [%d/%d] %s" % (i, total, item.dest_rel), flush=True)

    def totals(self):
        counts = dict((c, 0) for c in CATEGORIES)
        byte_totals = dict((c, 0) for c in CATEGORIES)
        for item in self.plan:
            counts[item.category] += 1
            byte_totals[item.category] += item.size
        return counts, byte_totals

    def write_manifest(self, path):
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            header = ["dest_relpath", "source_abspath", "bytes", "mtime_iso"]
            if self.do_hash:
                header.append("sha256")
            writer.writerow(header)
            for item in sorted(self.plan, key=lambda it: it.dest_rel):
                row = [item.dest_rel, item.src, item.size,
                       dt.datetime.fromtimestamp(item.mtime).isoformat(timespec="seconds")]
                if self.do_hash:
                    row.append(sha256_of(os.path.join(self.dest, item.dest_rel)))
                writer.writerow(row)


def sha256_of(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
    except Exception:
        return ""
    return digest.hexdigest()


def human(nbytes):
    step = 1024.0
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step or unit == "TiB":
            return "%.1f %s" % (value, unit) if unit != "B" else "%d B" % nbytes
        value /= step


# ------------------------------------------------------- per-movie bookkeeping

class Movie(object):
    """One prediction output directory, and everything read out of it."""

    def __init__(self, path, root):
        self.path = os.path.abspath(path)
        self.movie = os.path.basename(self.path)
        root_abs = os.path.abspath(root)
        self.rel = os.path.relpath(self.path, root_abs)
        parent = os.path.dirname(self.rel)
        # Pointing --experiments-root straight at one experiment leaves no parent
        # above the movie, so name the experiment after the root itself.
        self.experiment = (parent.replace(os.sep, "/") if parent not in ("", ".")
                           else os.path.basename(root_abs))
        self.dest_rel = "predictions/%s/%s" % (self.experiment, self.movie)
        self.members = member_dirs(self.path)
        self.skip_reason = None
        self.source = None
        self.summary = None
        self.legend = None
        self.realign = None
        self.pre_summary = None
        self.is_showcase = False

    # -- reading

    def classify(self, exclude=(), include_superseded_runs=False):
        """Fill skip_reason when this movie must not be collected."""
        parts = self.rel.split(os.sep)
        bad = [p for p in parts if p in BAD_DIRS]
        if bad:
            self.skip_reason = "under %s/" % bad[0]
            return
        hit = [p for p in parts if p in exclude]
        if hit:
            self.skip_reason = "excluded by --exclude %s" % hit[0]
            return
        if not include_superseded_runs:
            stale = [p for p in parts if SUPERSEDED_RUN.search(p)]
            if stale:
                self.skip_reason = ("superseded prediction run %s "
                                    "(--include-superseded-runs to collect it)" % stale[0])
                return
        src_path = os.path.join(self.path, "source.json")
        if not os.path.isfile(src_path):
            self.skip_reason = "missing source.json"
            return
        self.source, err = read_json(src_path)
        if err:
            self.skip_reason = "unreadable source.json (%s)" % err
            return
        if not self.members:
            self.skip_reason = "no member directories"

    def read_metadata(self, realign=True):
        self.summary, _ = read_json(os.path.join(self.path, "ensemble_model_selection_summary.json"))
        self.legend, _ = read_json(os.path.join(self.path, "model_index_legend.json"))
        if realign:
            self.realign, _ = read_json(os.path.join(self.path, ".realigned_ensemble.json"))
            archive = newest_superseded_ensemble(self.path, self.realign)
            if archive:
                self.pre_summary, _ = read_json(
                    os.path.join(archive, "ensemble_model_selection_summary.json"))
                self.pre_archive = archive
            else:
                self.pre_archive = None
        else:
            self.pre_archive = None

    # -- derived values

    def member_list(self):
        """Members in model-index order: the legend when it exists, else the
        sorted member directories."""
        if isinstance(self.legend, dict) and self.legend:
            try:
                keys = sorted(self.legend, key=lambda k: int(k))
            except (TypeError, ValueError):
                keys = sorted(self.legend)
            return [member_name_from_legend(str(self.legend[k])) for k in keys]
        return list(self.members)

    def analysis_csv(self):
        return self._one_glob("*_analysis_smoothed.csv")

    def analysis_h5(self):
        return self._one_glob("*_analysis_smoothed.h5")

    def _one_glob(self, pattern):
        import fnmatch
        try:
            names = sorted(os.listdir(self.path))
        except OSError:
            return None
        for name in names:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(self.path, name)
        return None

    def n_cameras(self):
        """From the shape of a reprojected array (frames, cameras, joints, 2);
        else the member config's camera count; else inferred from the number of
        camera pairs."""
        import fnmatch
        try:
            names = sorted(os.listdir(self.path))
        except OSError:
            names = []
        for name in names:
            if fnmatch.fnmatch(name, "points_ensemble_*reprojected*.npy"):
                shape = npy_shape(os.path.join(self.path, name))
                if shape and len(shape) == 4:
                    return int(shape[1])
        for member in self.members:
            cfg, err = read_json(os.path.join(self.path, member, "configuration.json"))
            if not err and isinstance(cfg, dict):
                value = cfg.get("number of cameras")
                if isinstance(value, int):
                    return value
        pairs = self.n_camera_pairs()
        for n in (2, 3, 4, 5, 6):
            if pairs == n * (n - 1) // 2:
                return n
        return None

    def n_camera_pairs(self):
        if isinstance(self.summary, dict) and isinstance(self.summary.get("num_camera_pairs"), int):
            return self.summary["num_camera_pairs"]
        shape = npy_shape(os.path.join(self.path, "all_models_combinations.npy"))
        if shape and len(shape) == 4:
            return int(shape[3])
        return None

    def n_frames(self):
        if isinstance(self.summary, dict) and isinstance(self.summary.get("num_frames"), int):
            return self.summary["num_frames"]
        shape = npy_shape(os.path.join(self.path, "points_3D_ensemble_best_method.npy"))
        if shape:
            return int(shape[0])
        return None

    def frame_range(self):
        """(first_frame, last_frame), trigger-relative -- frame 0 is the camera
        trigger in every product. From the analysis h5's frame_index when h5py
        is here, else the first and last `frame` cell of the analysis CSV."""
        if h5py is not None:
            path = self.analysis_h5()
            if path:
                try:
                    with h5py.File(path, "r") as handle:
                        if "frame_index" in handle:
                            idx = handle["frame_index"]
                            if idx.shape and idx.shape[0]:
                                return int(idx[0]), int(idx[-1])
                except Exception:
                    pass
        path = self.analysis_csv()
        if path:
            try:
                with open(path) as fh:
                    reader = csv.reader(fh)
                    header = next(reader)
                    col = header.index("frame")
                    first = last = None
                    for row in reader:
                        if len(row) > col and row[col] != "":
                            if first is None:
                                first = int(float(row[col]))
                            last = int(float(row[col]))
                    if first is not None:
                        return first, last
            except Exception:
                pass
        return None, None

    def n_valid_wing_angle_frames(self):
        """Frames where both wings' phi is finite. Needs h5py; '' without it."""
        if h5py is None:
            return None
        path = self.analysis_h5()
        if not path:
            return None
        try:
            with h5py.File(path, "r") as handle:
                if "wings_phi_left" not in handle or "wings_phi_right" not in handle:
                    return None
                left = np.asarray(handle["wings_phi_left"][:], dtype=float)
                right = np.asarray(handle["wings_phi_right"][:], dtype=float)
        except Exception:
            return None
        return int(np.sum(np.isfinite(left) & np.isfinite(right)))

    def ensemble_scores(self):
        return read_scores_readme(os.path.join(self.path, "README_scores_3D_ensemble.txt"))


def realign_pair(realign, key):
    """(before, after) out of a .realigned_ensemble.json two-element list."""
    if not isinstance(realign, dict):
        return None, None
    value = realign.get(key)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0], value[1]
    return None, None


# --------------------------------------------------------------- CSV builders

MOVIES_INDEX_HEADER = [
    "experiment", "movie", "run_dir", "source_movie_dir", "box_h5", "predicted_at",
    "n_cameras", "n_camera_pairs", "n_frames", "first_frame", "last_frame",
    "members", "n_members", "ensemble_score_raw", "ensemble_score_smoothed",
    "perturbation_status", "perturbation_type", "perturbation_onset_frame",
    "perturbation_end_frame", "perturbation_duration_ms",
    "lighting_regime", "light_off_frame",
    "realign_exchanged_pairs", "collapse_pct_before", "collapse_pct_after",
    "shape_error_um_before", "shape_error_um_after",
    "phi_wrong_pct_before", "phi_wrong_pct_after",
    "analysis_csv_rows", "n_frames_valid_wing_angles", "has_mp4", "has_analysis_h5",
]


def movies_index_row(movie):
    src = movie.source if isinstance(movie.source, dict) else {}
    pert = src.get("perturbation") or {}
    light = src.get("lighting") or {}
    members = movie.member_list()
    score_raw, score_smooth = movie.ensemble_scores()
    first, last = movie.frame_range()
    collapse_b, collapse_a = realign_pair(movie.realign, "collapse_pct")
    shape_b, shape_a = realign_pair(movie.realign, "shape_error_um")
    phi_b, phi_a = realign_pair(movie.realign, "phi_wrong_pct")
    csv_path = movie.analysis_csv()
    h5_path = movie.analysis_h5()
    return [
        movie.experiment,
        movie.movie,
        movie.path,
        src.get("source_movie_dir"),
        src.get("box_h5"),
        src.get("predicted_at"),
        movie.n_cameras(),
        movie.n_camera_pairs(),
        movie.n_frames(),
        first,
        last,
        ";".join(members),
        len(members),
        score_raw,
        score_smooth,
        pert.get("status"),
        pert.get("type"),
        pert.get("onset_frame"),
        pert.get("end_frame"),
        pert.get("duration_ms"),
        light.get("regime"),
        light.get("light_off_frame"),
        (movie.realign or {}).get("exchanged_pairs") if isinstance(movie.realign, dict) else None,
        collapse_b, collapse_a,
        shape_b, shape_a,
        phi_b, phi_a,
        count_csv_rows(csv_path) if csv_path else "",
        movie.n_valid_wing_angle_frames(),
        os.path.isfile(os.path.join(movie.path, "movie 2D and 3D.mp4")),
        h5_path is not None,
    ]


SELECTION_HEADER = ["experiment", "movie", "group", "model",
                    "fraction_of_frames_selected", "model_x_camerapair_selections",
                    "num_frames", "num_camera_pairs", "stage"]


def selection_rows(movie):
    rows = []
    for summary, stage in ((movie.summary, "post_realign"), (movie.pre_summary, "pre_realign")):
        if not isinstance(summary, dict):
            continue
        per_group = summary.get("per_group") or {}
        num_frames = summary.get("num_frames")
        num_pairs = summary.get("num_camera_pairs")
        for group in list(GROUPS) + [g for g in sorted(per_group) if g not in GROUPS]:
            stats = per_group.get(group)
            if not isinstance(stats, dict):
                continue
            for label in sorted(stats, key=lambda lbl: member_name_from_legend(lbl)):
                entry = stats[label] or {}
                rows.append([movie.experiment, movie.movie, group,
                             member_name_from_legend(label),
                             entry.get("fraction_of_frames_selected"),
                             entry.get("model_x_camerapair_selections"),
                             num_frames, num_pairs, stage])
    return rows


MEMBER_SCORES_HEADER = ["experiment", "movie", "member", "score_raw", "score_smoothed",
                        "ensemble_score_raw", "ensemble_score_smoothed"]


def member_scores_rows(movie):
    ens_raw, ens_smooth = movie.ensemble_scores()
    rows = []
    for member in movie.members:
        raw, smooth = read_scores_readme(os.path.join(movie.path, member, "README_scores_3D.txt"))
        rows.append([movie.experiment, movie.movie, member, raw, smooth, ens_raw, ens_smooth])
    return rows


SUBSET_SIZES_HEADER = ["experiment", "movie", "group", "n_models_in_subset",
                       "n_camera_pairs_in_subset", "n_frames"]


def subset_size_rows(movie):
    """Size distribution of the winning subsets, from all_models_combinations.npy
    (groups, frames, models, camera-pairs); a positive entry marks a
    (model, camera-pair) that was part of the winning subset for that frame.

    Two blocks, as two halves of one table: first how many models the winning
    subset held, then how many camera pairs. The column not being counted is
    empty, so a reader filters on whichever is filled.
    """
    path = os.path.join(movie.path, "all_models_combinations.npy")
    if not os.path.isfile(path):
        return []
    try:
        comb = np.load(path, mmap_mode="r")
    except Exception:
        return []
    if comb.ndim != 4:
        return []
    model_rows, pair_rows = [], []
    for g in range(comb.shape[0]):
        group = GROUPS[g] if g < len(GROUPS) else "group_%d" % g
        mask = np.asarray(comb[g]) > 0                       # (frames, models, pairs)
        n_models = mask.any(axis=2).sum(axis=1)              # per frame
        n_pairs = mask.any(axis=1).sum(axis=1)               # per frame
        for size, count in zip(*np.unique(n_models, return_counts=True)):
            model_rows.append([movie.experiment, movie.movie, group,
                               int(size), "", int(count)])
        for size, count in zip(*np.unique(n_pairs, return_counts=True)):
            pair_rows.append([movie.experiment, movie.movie, group,
                              "", int(size), int(count)])
    return model_rows + pair_rows


SKIPPED_HEADER = ["experiment", "movie", "run_dir", "reason"]


# ------------------------------------------------------------------- training

TRAINING_HEADER = [
    "run_name", "prediction_model_name", "model_type", "camera_fusion", "n_cameras",
    "dilation", "base_filters", "dropout", "loss", "epochs_configured",
    "best_epoch", "best_epoch_is_1_based", "best_val_loss",
    "best_epoch_val_l2_px", "final_val_l2_px",
    "n_train_samples", "n_val_samples", "split_is_frame_grouped",
    "data_path", "config_note",
]

BEST_EPOCH_RE = re.compile(r"Epoch:\s*(\d+)")
BEST_LOSS_RE = re.compile(r"Best Validation Loss:\s*(\S+)")


def read_best_model_info(path):
    """(best_epoch, best_val_loss) as written -- the epoch counts from 1 while
    history.csv counts from 0."""
    try:
        with open(path) as fh:
            text = fh.read()
    except Exception:
        return None, None
    epoch = BEST_EPOCH_RE.search(text)
    loss = BEST_LOSS_RE.search(text)
    return (int(epoch.group(1)) if epoch else None,
            loss.group(1) if loss else None)


def read_history_l2(path, best_epoch):
    """(val l2 at the best epoch, val l2 of the last epoch). (None, None) when
    history.csv has no L2 columns."""
    try:
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return None, None
    if not rows or "val l2" not in rows[0]:
        return None, None
    final = rows[-1].get("val l2") or None
    best = None
    if best_epoch is not None:
        index = best_epoch - 1                       # best_model_info is 1-based
        if 0 <= index < len(rows):
            best = rows[index].get("val l2") or None
    return best, final


def read_split_sizes(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            train = data["train_idx"] if "train_idx" in data.files else None
            val = data["val_idx"] if "val_idx" in data.files else None
        return (int(train.shape[0]) if train is not None else None,
                int(val.shape[0]) if val is not None else None)
    except Exception:
        return None, None


def split_is_frame_grouped(run_dir):
    """True when this run's own copy of train.py groups samples by source frame
    (the `sample_group_ids` split), so the validation set cannot hold another
    camera subset of a training frame."""
    path = os.path.join(run_dir, "training code", "train.py")
    try:
        with open(path) as fh:
            return "sample_group_ids" in fh.read()
    except Exception:
        return False


def config_note(config):
    """The run's own `"// ... //"` comment string: the first such key carrying
    prose rather than the 0 used as a section divider."""
    if not isinstance(config, dict):
        return None
    for key, value in config.items():
        if key.startswith("//") and isinstance(value, str) and value.strip():
            return value
    return None


def registered_members(models_root):
    """{member_name: model.json contents} for every registered ensemble member."""
    out = {}
    if not os.path.isdir(models_root):
        return out
    for name in sorted(os.listdir(models_root)):
        path = os.path.join(models_root, name, "model.json")
        if os.path.isfile(path):
            meta, err = read_json(path)
            out[name] = meta if not err else {}
    return out


def find_training_runs(train_root, wanted):
    if not os.path.isdir(train_root):
        return []
    runs = []
    for name in sorted(os.listdir(train_root)):
        path = os.path.join(train_root, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        if not os.path.isfile(os.path.join(path, "configuration.json")):
            continue
        if wanted is not None and not name_matches(wanted, "", name):
            continue
        runs.append(path)
    return runs


def training_row(run_dir, members_by_source):
    name = os.path.basename(run_dir)
    config, _ = read_json(os.path.join(run_dir, "configuration.json"))
    config = config if isinstance(config, dict) else {}
    best_epoch, best_loss = read_best_model_info(os.path.join(run_dir, "best_model_info.txt"))
    best_l2, final_l2 = read_history_l2(os.path.join(run_dir, "history.csv"), best_epoch)
    n_train, n_val = read_split_sizes(os.path.join(run_dir, "train_val_split.npz"))
    return [
        name,
        members_by_source.get(name, ""),
        config.get("model type"),
        config.get("camera fusion"),
        config.get("number of cameras"),
        config.get("dilation rate"),
        config.get("number of base filters"),
        config.get("dropout ratio"),
        config.get("loss function"),
        config.get("epochs"),
        best_epoch,
        True,
        best_loss,
        best_l2,
        final_l2,
        n_train,
        n_val,
        split_is_frame_grouped(run_dir),
        config.get("data path"),
        config_note(config),
    ], best_epoch


def plan_training(collector, run_dirs, members_by_source):
    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        dest = "training/%s" % name
        for filename in TRAIN_FILES:
            collector.add_file("training", os.path.join(run_dir, filename),
                               "%s/%s" % (dest, filename))
        collector.add_file("training", os.path.join(run_dir, "training code", "train.py"),
                           "%s/training code/train.py" % dest)
        best_epoch, _ = read_best_model_info(os.path.join(run_dir, "best_model_info.txt"))
        if best_epoch is not None:
            # exactly the best epoch's two figures -- never the per-epoch series
            collector.add_file("training",
                               os.path.join(run_dir, "histograms",
                                            "l2_histogram_epoch_%d.png" % best_epoch),
                               "%s/histograms/l2_histogram_epoch_%d.png" % (dest, best_epoch))
            collector.add_file("training",
                               os.path.join(run_dir, "l2_histograms_per_point",
                                            "validation_epoch_%d.png" % best_epoch),
                               "%s/l2_histograms_per_point/validation_epoch_%d.png"
                               % (dest, best_epoch))


# --------------------------------------------------------------- planning: movies

def plan_movie(collector, movie, tier, with_superseded):
    dest = movie.dest_rel
    for filename in MOVIE_FILES_INDEX:
        collector.add_file("index", os.path.join(movie.path, filename),
                           "%s/%s" % (dest, filename))
    for member in movie.members:
        for filename in MEMBER_FILES_INDEX:
            collector.add_file("index", os.path.join(movie.path, member, filename),
                               "%s/%s/%s" % (dest, member, filename))

    if tier in ("analysis", "showcase"):
        for filename in MOVIE_FILES_ANALYSIS:
            collector.add_file("analysis", os.path.join(movie.path, filename),
                               "%s/%s" % (dest, filename))
        for pattern in MOVIE_GLOBS_ANALYSIS:
            collector.add_glob("analysis", movie.path, pattern, dest)

    if tier == "showcase" and movie.is_showcase:
        for filename in MOVIE_FILES_SHOWCASE:
            collector.add_file("showcase", os.path.join(movie.path, filename),
                               "%s/%s" % (dest, filename))
        for pattern in MOVIE_GLOBS_SHOWCASE:
            collector.add_glob("showcase", movie.path, pattern, dest)
        for dirname in MOVIE_DIRS_SHOWCASE:
            collector.add_dir("showcase", os.path.join(movie.path, dirname),
                              "%s/%s" % (dest, dirname))
        for member in movie.members:
            for filename in MEMBER_FILES_SHOWCASE:
                collector.add_file("showcase", os.path.join(movie.path, member, filename),
                                   "%s/%s/%s" % (dest, member, filename))

    if with_superseded and getattr(movie, "pre_archive", None):
        for filename in PRE_REALIGN_FILES:
            collector.add_file("pre_realign",
                               os.path.join(movie.pre_archive, filename),
                               "%s/pre_realign/%s" % (dest, filename),
                               allow_archive=True)


# ------------------------------------------------------------------- provenance

def git_state(repo_dir):
    def git(*args):
        try:
            out = subprocess.run(["git"] + list(args), cwd=repo_dir,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return out.stdout.decode("utf-8", "replace").strip()
        except Exception:
            return ""
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    return commit, dirty


def write_provenance(path, args, collector, counts, byte_totals,
                     n_movies, n_experiments, n_skipped, repo_dir, csv_paths,
                     unnamed=()):
    commit, dirty = git_state(repo_dir)
    total_bytes = sum(byte_totals.values())
    with open(path, "w") as fh:
        fh.write("# Paper results bundle\n\n")
        fh.write("Collected by `code/collect_paper_results.py` out of this repo's\n"
                 "prediction and training outputs. Nothing in the source tree was\n"
                 "moved, written or deleted; every file here is a copy.\n\n")
        fh.write("## Run\n\n")
        fh.write("- date: %s\n" % dt.datetime.now().isoformat(timespec="seconds"))
        fh.write("- host: %s\n" % socket.gethostname())
        fh.write("- user: %s\n" % getpass.getuser())
        fh.write("- repo: %s\n" % repo_dir)
        fh.write("- git commit: %s\n" % (commit or "unknown"))
        fh.write("- working tree: %s\n" % ("DIRTY -- uncommitted changes were present"
                                           if dirty else "clean"))
        fh.write("- command: %s\n" % " ".join(sys.argv))
        fh.write("- python: %s\n" % sys.version.split()[0])
        fh.write("- numpy: %s\n" % np.__version__)
        fh.write("- h5py: %s\n" % (h5py.__version__ if h5py is not None
                                   else "NOT INSTALLED -- h5-only columns left empty"))
        fh.write("- platform: %s\n" % platform.platform())
        fh.write("\n## Contents\n\n")
        fh.write("- experiments: %d\n" % n_experiments)
        fh.write("- movies collected: %d\n" % n_movies)
        fh.write("- movies skipped: %d (see index/skipped.csv)\n" % n_skipped)
        fh.write("- tier: %s\n" % args.tier)
        fh.write("- total: %d files, %s\n\n" % (len(collector.plan), human(total_bytes)))
        fh.write("| group | files | bytes |\n|---|---|---|\n")
        for category in CATEGORIES:
            if counts.get(category):
                fh.write("| %s | %d | %s |\n"
                         % (category, counts[category], human(byte_totals[category])))
        fh.write("\n## Never collected\n\n")
        fh.write("`*.pt` weights, `weights/`, `viz_pred/`, each member's\n"
                 "`predicted_points_and_box.h5`, the per-epoch histogram series apart\n"
                 "from the best epoch, and the `superseded_*/` archives")
        if args.with_superseded:
            fh.write(" -- except the two pre-realignment "
                     "files `--with-superseded` placed in each movie's `pre_realign/`")
        fh.write(".\n")
        fh.write("\n## Tables\n\n")
        for name in csv_paths:
            fh.write("- `%s`\n" % os.path.relpath(name, os.path.dirname(path)))
        fh.write("\nConventions carried over from the pipeline: frame 0 is the camera\n"
                 "trigger in every product, and `body_pitch_deg` is nose-down positive.\n")
        if unnamed:
            fh.write("\n## Movies whose ensemble members are unnamed\n\n")
            fh.write("These movies hold more member directories than the selection array\n"
                     "has models -- a second prediction run left extra `*_01` member\n"
                     "folders -- so `predict.py` fell back to positional names and their\n"
                     "legend says `model_0 ... model_N`. Which trained model each index is\n"
                     "cannot be recovered from what is on disk, so a table grouped by\n"
                     "member name must leave them out or the movie must be re-predicted:\n\n")
            for movie in unnamed:
                fh.write("- `%s/%s` (%d member directories on disk)\n"
                         % (movie.experiment, movie.movie, len(movie.members)))


# ------------------------------------------------------------------------ main

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect the prediction and training files a paper needs, "
                    "plus the summary CSVs that make their numbers citable.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tiers: index (small text, every movie) < analysis (+ kinematics "
               "CSV and selection mask, every movie) < showcase (+ heavy "
               "per-movie material, only --showcase-file movies).")
    parser.add_argument("--dest", required=True, help="destination folder (created)")
    parser.add_argument("--experiments-root", default="predict_output",
                        help="tree to walk for movie directories (default: predict_output)")
    parser.add_argument("--train-root", default=os.path.join("train_output", "debug_outputs"),
                        help="tree holding training runs (default: train_output/debug_outputs)")
    parser.add_argument("--models-root", default="prediction_models",
                        help="registered ensemble members (default: prediction_models)")
    parser.add_argument("--tier", choices=TIERS, default="analysis",
                        help="how much per movie (default: analysis)")
    parser.add_argument("--showcase-file",
                        help="movies that get the showcase tier, one per line")
    parser.add_argument("--models-file",
                        help="training runs to collect, one per line (default: all)")
    parser.add_argument("--exclude", action="append", default=[], metavar="NAME",
                        help="skip movies under a path component named NAME "
                             "(repeatable); bad_wings/ and bad_signal/ are always skipped")
    parser.add_argument("--include-superseded-runs", action="store_true",
                        help="also collect movies under a folder whose name says "
                             "SUPERSEDED -- an earlier prediction run of movies a later "
                             "run redid, so they are skipped by default")
    parser.add_argument("--with-superseded", action="store_true",
                        help="also take the pre-realignment ensemble summary and score "
                             "readme from the newest superseded_ensemble_*/ into "
                             "<movie>/pre_realign/")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be copied and the size per tier; copy nothing")
    parser.add_argument("--force", action="store_true",
                        help="re-copy files that already match in the destination")
    parser.add_argument("--hash", dest="do_hash", action="store_true",
                        help="add a sha256 per file to manifest.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = args.experiments_root
    if not os.path.isdir(root):
        print("ERROR: --experiments-root %s is not a directory" % root)
        return 2

    dest_abs = os.path.abspath(args.dest)
    root_abs = os.path.abspath(root)
    if dest_abs == root_abs or dest_abs.startswith(root_abs + os.sep):
        print("ERROR: --dest is inside --experiments-root; the source tree is read-only")
        return 2

    if h5py is None:
        print("NOTE: h5py is not installed -- movies_index.csv's "
              "n_frames_valid_wing_angles will be empty, and the frame range comes "
              "from the analysis CSV instead of the h5.")

    showcase_names = load_name_list(args.showcase_file)
    models_names = load_name_list(args.models_file)
    if args.tier == "showcase" and not showcase_names:
        print("NOTE: --tier showcase without --showcase-file: no movie gets the "
              "showcase extras, so this collects the analysis tier.")

    print("scanning %s ..." % root, flush=True)
    movie_dirs = find_movie_dirs(root)
    print("found %d movie directories" % len(movie_dirs), flush=True)

    kept, skipped = [], []
    for path in movie_dirs:
        movie = Movie(path, root)
        movie.classify(exclude=set(args.exclude),
                       include_superseded_runs=args.include_superseded_runs)
        if movie.skip_reason:
            skipped.append(movie)
            continue
        if showcase_names:
            movie.is_showcase = name_matches(showcase_names, movie.experiment, movie.movie)
        kept.append(movie)

    if args.tier == "showcase" and showcase_names:
        n_show = sum(1 for m in kept if m.is_showcase)
        print("showcase movies matched: %d of %d names" % (n_show, len(showcase_names)))
        unmatched = [n for n in showcase_names
                     if not any(name_matches([n], m.experiment, m.movie) for m in kept)]
        for name in unmatched:
            print("  WARNING: --showcase-file name matched no collected movie: %s" % name)

    collector = Collector(args.dest, dry_run=args.dry_run, force=args.force,
                          do_hash=args.do_hash)

    print("planning ...", flush=True)
    for i, movie in enumerate(kept, 1):
        if args.with_superseded:
            realign, _ = read_json(os.path.join(movie.path, ".realigned_ensemble.json"))
            movie.pre_archive = newest_superseded_ensemble(movie.path, realign)
        plan_movie(collector, movie, args.tier, args.with_superseded)
        if i % 100 == 0 or i == len(kept):
            print("  planned %d/%d movies" % (i, len(kept)), flush=True)

    members = registered_members(args.models_root)
    for name in sorted(members):
        collector.add_file("models", os.path.join(args.models_root, name, "model.json"),
                           "models/%s/model.json" % name)
    # registered member name for a training run, via model.json's "source"
    members_by_source = {}
    for name, meta in members.items():
        source = (meta or {}).get("source")
        if source:
            members_by_source[os.path.basename(str(source).rstrip("/"))] = name

    run_dirs = find_training_runs(args.train_root, models_names)
    plan_training(collector, run_dirs, members_by_source)
    print("planned %d training runs and %d registered members"
          % (len(run_dirs), len(members)), flush=True)

    counts, byte_totals = collector.totals()
    experiments = sorted(set(m.experiment for m in kept))

    if args.dry_run:
        print("\n--- dry run: nothing copied ---")
        for item in collector.plan:
            print("  %-12s %10d  %s" % (item.category, item.size, item.dest_rel))
        print("\nper group:")
        for category in CATEGORIES:
            if counts.get(category):
                print("  %-12s %6d files  %12s" % (category, counts[category],
                                                   human(byte_totals[category])))
        print("  %-12s %6d files  %12s" % ("TOTAL", len(collector.plan),
                                           human(sum(byte_totals.values()))))
        print("\nmovies: %d collected, %d skipped; experiments: %d"
              % (len(kept), len(skipped), len(experiments)))
        if collector.refused:
            print("refused (never-copy rule): %d paths" % len(collector.refused))
        return 0

    if not os.path.isdir(collector.dest):
        os.makedirs(collector.dest)
    index_dir = os.path.join(collector.dest, "index")
    if not os.path.isdir(index_dir):
        os.makedirs(index_dir)

    print("\ncopying %d files (%s) into %s ..."
          % (len(collector.plan), human(sum(byte_totals.values())), collector.dest), flush=True)
    collector.run()
    print("copied %d, already present %d" % (collector.copied, collector.reused), flush=True)

    print("reading metadata and writing tables ...", flush=True)
    rows_movies, rows_selection, rows_members, rows_subsets = [], [], [], []
    unnamed = []
    for i, movie in enumerate(kept, 1):
        movie.read_metadata()
        if any(m.startswith("model_") for m in movie.member_list()):
            unnamed.append(movie)
        rows_movies.append(movies_index_row(movie))
        rows_selection.extend(selection_rows(movie))
        rows_members.extend(member_scores_rows(movie))
        rows_subsets.extend(subset_size_rows(movie))
        if i % 25 == 0 or i == len(kept):
            print("  %d/%d movies read" % (i, len(kept)), flush=True)

    rows_training = []
    for run_dir in run_dirs:
        row, _ = training_row(run_dir, members_by_source)
        rows_training.append(row)

    rows_skipped = [[m.experiment, m.movie, m.path, m.skip_reason] for m in skipped]

    csv_paths = []

    def write_csv(name, header, rows):
        path = os.path.join(index_dir, name)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for row in rows:
                writer.writerow([fmt(cell) for cell in row])
        csv_paths.append(path)
        return path

    write_csv("movies_index.csv", MOVIES_INDEX_HEADER, rows_movies)
    write_csv("selection_by_group.csv", SELECTION_HEADER, rows_selection)
    write_csv("member_scores.csv", MEMBER_SCORES_HEADER, rows_members)
    write_csv("subset_sizes.csv", SUBSET_SIZES_HEADER, rows_subsets)
    write_csv("training_index.csv", TRAINING_HEADER, rows_training)
    write_csv("skipped.csv", SKIPPED_HEADER, rows_skipped)

    manifest = os.path.join(collector.dest, "manifest.csv")
    print("writing manifest%s ..." % (" with sha256" if args.do_hash else ""), flush=True)
    collector.write_manifest(manifest)

    provenance = os.path.join(collector.dest, "PROVENANCE.md")
    write_provenance(provenance, args, collector, counts, byte_totals,
                     len(kept), len(experiments), len(skipped), repo_dir, csv_paths,
                     unnamed)

    print("\n--- summary ---")
    print("movies collected: %d   skipped: %d   experiments: %d"
          % (len(kept), len(skipped), len(experiments)))
    print("files: %d   total size: %s" % (len(collector.plan), human(sum(byte_totals.values()))))
    for category in CATEGORIES:
        if counts.get(category):
            print("  %-12s %6d files  %12s" % (category, counts[category],
                                               human(byte_totals[category])))
    if collector.refused:
        print("refused by the never-copy rule: %d paths" % len(collector.refused))
    if unnamed:
        print("WARNING: %d movies have unnamed members (legend fell back to "
              "model_0..model_N because extra *_01 member folders outnumber the "
              "models in the selection array); see PROVENANCE.md:" % len(unnamed))
        for movie in unnamed:
            print("    %s/%s" % (movie.experiment, movie.movie))
    print("tables:")
    for path in csv_paths:
        print("  %s" % path)
    print("  %s" % manifest)
    print("  %s" % provenance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
