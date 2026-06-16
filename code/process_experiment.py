"""
process_experiment.py
=====================

End-to-end wrapper around the pose-estimation data-prep pipeline. Drives:

    1. CLEAN — delete extraneous files left over from Roni's hull-reconstruction
       project: *.csv, desktop.ini, hull_op/ and Segmentation/ folders.
    2. PRESCAN — scan source *_sparse.mat files; flag movies where the fly is
       tracked by all 4 cams for fewer than --prescan-min-intersection frames
       (default 500). Flagged movies are dropped from FLIP/BUILD/VERIFY.
    3. FLIP  — vertically flip the mirror camera's sparse .mat (e.g. cam1 in
       2023, cam5 in 2022) using `code/flip_sparse_cam_mat.py`'s logic.
    4. BUILD — invoke MATLAB to produce one h5 per movie plus one shared
       calibration.h5 (same logic as `code/build_experiment.sh`).
    5. VERIFY (optional) — per-movie reprojection-error sanity check using
       `code/verify_calibration.py`'s machinery.

After BUILD (multi-movie mode), a manifest is written to
`manifests/good_movies_<experiment>.txt` containing one movie directory per
OK movie, ready for `sbatch --array=0-N% predict_array.sh <manifest> <config>`.

A full transcript of each run is appended to `<input_dir>/process_report.txt`
(timestamped) so you can review prescan + verify output later.

Input modes (auto-detected from the path's contents):
    single-movie:    <input_dir>/movN_camK_sparse.mat × 4
    experiment-dir:  <input_dir>/mov*/movX_camK_sparse.mat × 4

USAGE
-----
    .env/bin/python code/process_experiment.py <input_dir> [options]

Common options:
    --easywand <path>       path to easyWand .mat (REQUIRED for the build step;
                            for multi-day Roni experiments use the END-of-
                            experiment .mat)
    --cam <name>            cam-name substring of the mirror cam to flip
                            (e.g. 'cam1' for 2023 setups, 'cam5' for 2022).
                            REQUIRED for the flip step.
    --max-frames N          cap each movie's h5 to the first N frames
                            (default: process every frame)
    --no-verify             skip the verification step (it runs by default)
    --verify-only           skip clean / prescan / flip / build; just verify
                            existing h5 + calibration files
    --skip-clean            skip the cleanup step
    --skip-prescan          skip the source-mat fly-visibility prescan
    --prescan-only          run only the prescan, then exit
    --prescan-min-intersection N
                            flag a movie BAD if the all-cams fly-visible
                            intersection has fewer than N frames (default: 500)
    --prescan-pixel-threshold N
                            non-zero pixel count per frame above which a cam
                            is considered to "see the fly" (default: 50)
    --skip-flip             skip the flip step (e.g. if already flipped)
    --skip-build            skip the h5 build step
    --verify-threshold P    LOO median (px) above which a movie is marked FAIL
                            (default: 15). The known-broken cases give 100-400
                            px; the known-working cases give 2-8 px.
    --dry-run               print what each step would do, do nothing
    -h, --help              show this message

Examples
--------
    # End-to-end on the 2023 experiment, flipping cam1 (verify runs by default):
    .env/bin/python code/process_experiment.py \\
        inference_datasets/new_roni_experiments/2023_08_09_60ms/movies \\
        --easywand inference_datasets/new_roni_experiments/2023_08_09_60ms/10_8_23/10_8_23_allmovs_easyWandData.mat \\
        --cam cam1

    # Single movie, 500-frame test build:
    .env/bin/python code/process_experiment.py \\
        inference_datasets/.../2023_08_09_60ms/movies/mov5 \\
        --easywand inference_datasets/.../2023_08_09_60ms/10_8_23/10_8_23_allmovs_easyWandData.mat \\
        --cam cam1 --max-frames 500

    # Verify-only after a previous build:
    .env/bin/python code/process_experiment.py \\
        inference_datasets/.../2023_08_09_60ms/movies --verify-only

Notes
-----
- The flip step is NOT idempotent. Running this script with --cam twice on the
  same data un-does the flip. Use --skip-flip on follow-up runs.
- The build step requires MATLAB on PATH (override with MATLAB_BIN env var).
- The verify step uses blob-silhouette centroids as the 2D measurement source.
  Typical "working" LOO medians are 2-8 px; "broken" calibrations give 100+.
"""

import argparse
import contextlib
import datetime
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

# Local sibling imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flip_sparse_cam_mat import flip_sparse_cam_mat
from pipeline_timing import record as record_timing
from scan_sparse_movies import scan_experiment
from verify_calibration import (
    load_calibration,
    collect_measurements,
    per_cam_errors,
)


def _timings_path(input_dir):
    """Per-experiment timing ledger location (shared with the predict array)."""
    return os.path.join(input_dir, "pipeline_timings.csv")

# ---------------------------------------------------------------------------
# Paths to the MATLAB build scripts, mirroring build_experiment.sh.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SCRIPT = os.path.join(REPO_ROOT, "matlab", "CreateDatasetHDF5_from_list_fixed.m")
CALIB_SCRIPT = os.path.join(REPO_ROOT, "matlab", "run after experiment",
                            "create_camera_calibration_h5.m")
MATLAB_BIN = os.environ.get("MATLAB_BIN", "matlab")


# ---------------------------------------------------------------------------
# Stdout tee — mirrors all `print()` output to a buffer while still showing
# it to the user, so we can write a full report file at the end.
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------
def count_sparse_mats(d: str) -> int:
    """Count *_sparse.mat files directly inside d (non-recursive)."""
    return len(glob.glob(os.path.join(d, "*_sparse.mat")))


def parse_movie_num(d: str) -> "int | None":
    """Parse N from a 'movN' basename. Returns None on no match."""
    m = re.match(r"^mov(\d+)$", os.path.basename(d))
    return int(m.group(1)) if m else None


def detect_mode(input_dir: str) -> tuple:
    """Returns ('single', [(movie_dir, movie_num)]) or
       ('multi', [(movie_dir, movie_num), ...]) or sys.exits on error.
    """
    n_direct = count_sparse_mats(input_dir)
    if n_direct == 4:
        mn = parse_movie_num(input_dir)
        if mn is None:
            sys.exit(f"Single-movie mode requires a 'mov<N>' basename; got "
                     f"'{os.path.basename(input_dir)}'")
        return "single", [(input_dir, mn)]
    if n_direct != 0:
        sys.exit(f"Ambiguous: {input_dir} contains {n_direct} *_sparse.mat "
                 f"(expected 0 or 4)")
    # Multi-movie mode.
    movies = []
    incomplete = []   # (name, n_sparse) for mov dirs without exactly 4 mats
    for sub in sorted(glob.glob(os.path.join(input_dir, "mov*"))):
        if not os.path.isdir(sub):
            continue
        mn = parse_movie_num(sub)
        if mn is None:
            continue
        n = count_sparse_mats(sub)
        if n == 4:
            movies.append((sub, mn))
        else:
            print(f"  (skipping {sub}: {n} *_sparse.mat, expected 4)")
            incomplete.append((os.path.basename(sub), n))
    if incomplete:
        print(f"Skipped {len(incomplete)} incomplete movie(s) "
              f"(missing/extra *_sparse.mat — dropped from the whole pipeline): "
              + ", ".join(f"{name}({n}/4)" for name, n in incomplete))
    if not movies:
        sys.exit(f"No 'mov<N>/' subdirs with 4 *_sparse.mat in {input_dir}")
    return "multi", movies


# ---------------------------------------------------------------------------
# Clean step
# ---------------------------------------------------------------------------
CLEAN_FILE_GLOBS = ("*.csv", "desktop.ini")
CLEAN_DIR_NAMES = ("hull_op", "Segmentation")


def clean_one_dir(d: str, dry_run: bool) -> tuple:
    """Remove extraneous files/dirs directly inside d. Returns (n_files, n_dirs)."""
    n_files = 0
    n_dirs = 0
    for pat in CLEAN_FILE_GLOBS:
        for f in glob.glob(os.path.join(d, pat)):
            if os.path.isfile(f):
                print(f"  {'would remove' if dry_run else 'remove'} file:  {f}")
                if not dry_run:
                    os.remove(f)
                n_files += 1
    for name in CLEAN_DIR_NAMES:
        sub = os.path.join(d, name)
        if os.path.isdir(sub):
            print(f"  {'would remove' if dry_run else 'remove'} dir:   {sub}")
            if not dry_run:
                shutil.rmtree(sub)
            n_dirs += 1
    return n_files, n_dirs


def run_clean(input_dir: str, mode: str, movies: list, dry_run: bool) -> None:
    print("\n===== CLEAN =====")
    total_files, total_dirs = 0, 0
    # Always clean the experiment-level dir (multi) or the movie dir itself (single).
    f, d = clean_one_dir(input_dir, dry_run)
    total_files += f
    total_dirs += d
    if mode == "multi":
        for movie_dir, _ in movies:
            f, d = clean_one_dir(movie_dir, dry_run)
            total_files += f
            total_dirs += d
    print(f"Clean done: {total_files} files, {total_dirs} dirs removed.")


# ---------------------------------------------------------------------------
# Flip step
# ---------------------------------------------------------------------------
def find_flip_target(movie_dir: str, cam: str) -> "str | None":
    matches = sorted(glob.glob(os.path.join(movie_dir, f"*{cam}_sparse.mat")))
    if len(matches) == 0:
        print(f"  skip {os.path.basename(movie_dir)}: no '*{cam}_sparse.mat'")
        return None
    if len(matches) > 1:
        print(f"  skip {os.path.basename(movie_dir)}: multiple '*{cam}_sparse.mat'")
        return None
    return matches[0]


def run_flip(movies: list, cam: str, dry_run: bool,
             timings_path: "str | None" = None) -> None:
    print(f"\n===== FLIP (cam='{cam}') =====")
    n_ok = n_fail = n_skip = 0
    failed = []
    for movie_dir, _ in movies:
        target = find_flip_target(movie_dir, cam)
        if target is None:
            n_skip += 1
            continue
        name = os.path.basename(movie_dir)
        if dry_run:
            print(f"  would flip [{name}]: {os.path.basename(target)}")
            n_ok += 1
            continue
        print(f"  flip [{name}]: {os.path.basename(target)}")
        t0 = time.time()
        ok = flip_sparse_cam_mat(target)
        t1 = time.time()
        record_timing(timings_path, name, "flip", t0, t1)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failed.append(name)
    print(f"Flip done: {n_ok} ok, {n_fail} failed, {n_skip} skipped.")
    if failed:
        print(f"Failed: {', '.join(failed)}")


# ---------------------------------------------------------------------------
# Build step (h5 per movie + one shared calibration.h5)
# ---------------------------------------------------------------------------
def build_dataset_matlab_cmd(sparse_folder_path: str, save_path: str,
                             movie_num: int, max_frames: "int | None",
                             start_ind: "int | None" = None,
                             end_ind: "int | None" = None) -> str:
    cmd = (f"sparse_folder_path='{sparse_folder_path}'; "
           f"save_path='{save_path}'; "
           f"movie_num={movie_num};")
    if start_ind is not None:
        cmd += f" start_ind={start_ind};"
    if end_ind is not None:
        cmd += f" end_ind={end_ind};"
    if max_frames is not None:
        cmd += f" max_frames={max_frames};"
    cmd += f" run('{DATASET_SCRIPT}')"
    return cmd


def build_calibration_matlab_cmd(easywand_path: str, save_path: str) -> str:
    return (f"easy_wand_path='{easywand_path}'; "
            f"savePath='{save_path}'; "
            f"run('{CALIB_SCRIPT}')")


def matlab_batch(cmd: str, dry_run: bool) -> int:
    if dry_run:
        print(f"  would run MATLAB: {cmd}")
        return 0
    print(f"  MATLAB: {cmd}")
    result = subprocess.run([MATLAB_BIN, "-batch", cmd])
    return result.returncode


def run_build(input_dir: str, mode: str, movies: list,
              easywand: str, max_frames: "int | None", dry_run: bool,
              movie_ranges: "dict | None" = None,
              timings_path: "str | None" = None) -> None:
    print("\n===== BUILD =====")
    sparse_folder_path = input_dir if mode == "multi" else os.path.dirname(input_dir)
    n_ok = n_fail = 0
    failed = []
    for movie_dir, mn in movies:
        start_ind = end_ind = None
        if movie_ranges and movie_dir in movie_ranges:
            start_ind, end_ind = movie_ranges[movie_dir]
        print(f"\n-- mov{mn}")
        print(f"   sparse_folder_path={sparse_folder_path}")
        print(f"   save_path={movie_dir}")
        if start_ind is not None:
            print(f"   build range: start_ind={start_ind}, end_ind={end_ind} "
                  f"({end_ind - start_ind + 1} frames)")
        cmd = build_dataset_matlab_cmd(sparse_folder_path, movie_dir, mn,
                                       max_frames, start_ind, end_ind)
        t0 = time.time()
        rc = matlab_batch(cmd, dry_run)
        t1 = time.time()
        # On success, read the built h5 to capture the post-intersection
        # frame count for the timing ledger.
        n_built = None
        if rc == 0 and not dry_run:
            built = find_movie_h5(movie_dir)
            if built is not None:
                try:
                    with h5py.File(built, "r") as f:
                        n_built = int(f["cropzone"].shape[0])
                except Exception:
                    pass
        record_timing(timings_path, f"mov{mn}", "build", t0, t1,
                      n_frames=n_built)
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
            failed.append(f"mov{mn}")
    print(f"\nBuild dataset: {n_ok} ok, {n_fail} failed.")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    # Build the calibration.h5 inside input_dir (matches build_experiment.sh).
    calib_out = os.path.join(input_dir, "calibration.h5")
    print(f"\n-- calibration.h5 → {calib_out}")
    if os.path.isfile(calib_out) and not dry_run:
        print(f"   (removing pre-existing {calib_out} to avoid h5create collision)")
        os.remove(calib_out)
    cmd = build_calibration_matlab_cmd(easywand, calib_out)
    matlab_batch(cmd, dry_run)


# ---------------------------------------------------------------------------
# Prescan + manifest helpers
# ---------------------------------------------------------------------------
# The MATLAB builder needs `time_jump` frames of padding before start_ind and
# after end_ind for its time-channel windows; keep this in sync with the
# script's `time_jump` (=7).
MATLAB_TIME_JUMP_MARGIN = 7


def run_prescan(input_dir: str, movies: list,
                min_intersection: int, pixel_threshold: int,
                blob_ratio: float, blob_distance: float,
                dry_run: bool) -> tuple:
    """Returns (filtered_movies, movie_ranges):
      - filtered_movies: list of (movie_dir, movie_num) — only OK movies
      - movie_ranges:    dict {movie_dir: (start_ind, end_ind)} in MATLAB's
                         1-based inclusive convention, clamped to leave
                         `MATLAB_TIME_JUMP_MARGIN` frames of padding so the
                         builder's time-channel windows stay in range.
    BAD movies are reported to stdout and dropped from subsequent steps."""
    print(f"\n===== PRESCAN =====")
    results = scan_experiment(input_dir, min_intersection,
                              pixel_threshold, blob_ratio, blob_distance,
                              print_results=True)
    ok_dirs = {r["movie_dir"] for r in results if r["verdict"] == "OK"}
    n_before = len(movies)
    filtered = [(d, n) for d, n in movies if d in ok_dirs]
    n_dropped = n_before - len(filtered)
    if n_dropped:
        action = "would drop" if dry_run else "dropping"
        print(f"\nPrescan: {action} {n_dropped} BAD movie(s) "
              f"from FLIP / BUILD / VERIFY steps")
    movie_ranges = {}
    for r in results:
        if r["verdict"] != "OK":
            continue
        # 0-based [good_start, good_end) -> 1-based inclusive [start, end].
        # Clamp to leave time-jump padding both ends (MATLAB indexes
        # `(start_ind - time_jump):(end_ind + time_jump)` from the raw mat).
        start_ind = max(r["good_start"] + 1, MATLAB_TIME_JUMP_MARGIN + 1)
        end_ind = min(r["good_end"], r["n_frames"] - MATLAB_TIME_JUMP_MARGIN)
        if start_ind <= end_ind:
            movie_ranges[r["movie_dir"]] = (start_ind, end_ind)
    return filtered, movie_ranges


def write_good_movies_manifest(input_dir: str, movies: list,
                               dry_run: bool) -> "str | None":
    """Write manifests/good_movies_<experiment>.txt — one movie directory per
    line, ready for `sbatch --array=...% predict_array.sh <manifest>`.
    Skips directories that don't yet have a built h5 (so BAD movies that
    were filtered out don't end up in the manifest)."""
    if dry_run:
        print("\n(dry-run: would write manifest of good movies)")
        return None
    manifest_dir = os.path.join(REPO_ROOT, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    exp_name = os.path.basename(input_dir.rstrip(os.sep))
    manifest_path = os.path.join(manifest_dir, f"good_movies_{exp_name}.txt")
    kept_dirs = []
    missing = []
    for movie_dir, mn in movies:
        if find_movie_h5(movie_dir) is None:
            missing.append(f"mov{mn}")
        else:
            kept_dirs.append(movie_dir)
    with open(manifest_path, "w") as f:
        for d in kept_dirs:
            f.write(d + "\n")
    print(f"\nWrote manifest: {manifest_path}  ({len(kept_dirs)} movies)")
    if missing:
        print(f"  (skipped {len(missing)} movie(s) with no built h5: "
              f"{', '.join(missing)})")
    return manifest_path


# ---------------------------------------------------------------------------
# Verify step
# ---------------------------------------------------------------------------
def find_movie_h5(movie_dir: str) -> "str | None":
    """Find the dataset h5 produced by the build step (one per movie dir).
    The build writes filenames like `mov_<N>_<start>_<end>_ds_<tc>tc_<tj>tj.h5`."""
    matches = sorted(glob.glob(os.path.join(movie_dir, "mov_*_ds_*tc_*tj.h5")))
    return matches[0] if matches else None


def find_calibration_h5(input_dir: str, mode: str, movies: list) -> "str | None":
    """Find the calibration.h5 for a verify run. Looks in plausible locations
    in priority order, since builds may have placed it either inside the
    movie dir (single-mode build) or at the experiment level (multi-mode)."""
    candidates = []
    if mode == "single":
        movie_dir = movies[0][0]
        candidates.append(os.path.join(movie_dir, "calibration.h5"))
        candidates.append(os.path.join(os.path.dirname(movie_dir), "calibration.h5"))
    else:
        candidates.append(os.path.join(input_dir, "calibration.h5"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def verify_one_movie(h5_path: str, calib_path: str, threshold: float,
                     image_height: int = 800, num_cams: int = 4,
                     subsample_frames: int = 500,
                     min_intersection: int = 30) -> tuple:
    """Verify a movie's calibration by reprojection-error.

    Samples from the intersection of frames where ALL cameras have a tracked
    blob (cropzone != [1,1]). Without this, the verify can sample frames where
    a camera didn't capture the fly yet, producing NaN / artificially huge
    errors that look like calibration failures but are actually just data gaps.

    Returns (status, medians, info) where:
      - status: 'PASS' | 'FAIL' | 'BAD_DATA' | 'ERR'
      - medians: per-cam LOO medians (list[float]) for PASS/FAIL,
                 None for BAD_DATA, error message list for ERR.
      - info: dict with diagnostic counts, or None.
    """
    try:
        M = load_calibration(calib_path)
    except Exception as e:
        return "ERR", [f"calib load err: {e}"], None
    try:
        with h5py.File(h5_path, "r") as f:
            cropzone_full = f["cropzone"][:]
            n_total = int(cropzone_full.shape[0])
            # All-cams intersection: frames where every cam has a real blob
            # (the MATLAB builder writes cropzone=[1,1] when blob detection failed).
            bad_per_cam = ((cropzone_full[:, :, 0] == 1) &
                           (cropzone_full[:, :, 1] == 1))
            all_valid = ~bad_per_cam.any(axis=1)
            valid_idx = np.where(all_valid)[0]
            n_intersection = int(len(valid_idx))
            info = {"n_intersection": n_intersection, "n_total": n_total}
            if n_intersection < min_intersection:
                return "BAD_DATA", None, info
            # Sample subsample_frames evenly across the intersection so we get
            # coverage of the whole tracked segment, not just one window.
            if n_intersection > subsample_frames:
                sample_idx = valid_idx[np.linspace(0, n_intersection - 1,
                                                   subsample_frames,
                                                   dtype=int)]
            else:
                sample_idx = valid_idx
            box = f["box"][sample_idx]
            cropzone = cropzone_full[sample_idx]
    except Exception as e:
        return "ERR", [f"h5 load err: {e}"], None

    meas, valid = collect_measurements(box, cropzone, image_height, num_cams)
    errs = per_cam_errors(meas, valid, M, mode="loo")
    medians = [float(np.nanmedian(errs[:, c])) for c in range(num_cams)]
    passed = all(m < threshold for m in medians if not np.isnan(m))
    return ("PASS" if passed else "FAIL"), medians, info


def run_verify(input_dir: str, mode: str, movies: list,
               threshold: float, dry_run: bool,
               timings_path: "str | None" = None) -> "list | None":
    """Returns the list of (movie_dir, movie_num) that PASSED, or None when
    no filtering happened (calibration missing, or dry-run) so the caller
    falls back to the input list."""
    print(f"\n===== VERIFY (threshold: {threshold:.1f} px per-cam LOO median) =====")
    calib_path = find_calibration_h5(input_dir, mode, movies)
    if calib_path is None:
        print(f"  ERROR: calibration.h5 not found near {input_dir}; skipping verify.")
        return None
    print(f"  using calibration: {calib_path}")
    if dry_run:
        print(f"  would verify {len(movies)} movies against {calib_path}")
        return None

    passed_movies = []
    n_pass = n_fail = n_bad = n_err = n_missing = 0
    failed_lines = []
    bad_lines = []
    for movie_dir, mn in movies:
        h5 = find_movie_h5(movie_dir)
        if h5 is None:
            print(f"  [mov{mn}] no dataset h5 found; skipping")
            n_missing += 1
            continue
        t0 = time.time()
        status, medians, info = verify_one_movie(h5, calib_path, threshold)
        t1 = time.time()
        n_inter = info["n_intersection"] if (info and "n_intersection" in info) else None
        record_timing(timings_path, f"mov{mn}", "verify", t0, t1,
                      n_frames=n_inter)
        if status == "ERR":
            n_err += 1
            print(f"  [mov{mn}] ERR  {medians}")
            failed_lines.append(f"mov{mn} load error: {medians}")
            continue
        if status == "BAD_DATA":
            n_bad += 1
            inter = info["n_intersection"]
            total = info["n_total"]
            print(f"  [mov{mn}] BAD_DATA — only {inter}/{total} frames have "
                  f"all 4 cams tracking simultaneously; skipping")
            bad_lines.append(f"mov{mn}  intersection={inter}/{total}")
            continue
        # PASS or FAIL
        if isinstance(medians, list) and all(isinstance(m, (int, float))
                                             and not np.isnan(m) for m in medians):
            meds_str = "[" + ", ".join(f"{m:.1f}" for m in medians) + "]"
        else:
            meds_str = str(medians)
        inter = info["n_intersection"]
        total = info["n_total"]
        print(f"  [mov{mn}] {status}  LOO medians: {meds_str}  "
              f"(intersection={inter}/{total})")
        if status == "PASS":
            n_pass += 1
            passed_movies.append((movie_dir, mn))
        else:
            n_fail += 1
            failed_lines.append(f"mov{mn} {meds_str}")

    print(f"\nVerify summary: {n_pass} PASSED, {n_fail} FAILED, "
          f"{n_bad} BAD_DATA, {n_err} ERR, "
          f"{n_missing} missing (of {len(movies)} total).")
    if failed_lines:
        print("Failed details:")
        for line in failed_lines:
            print(f"  {line}")
    if bad_lines:
        print("BAD_DATA details (recording problem — fly not tracked by all 4 cams):")
        for line in bad_lines:
            print(f"  {line}")
    dropped = len(movies) - len(passed_movies)
    if dropped:
        print(f"Dropping {dropped} non-PASS movie(s) from the manifest.")
    return passed_movies


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_dir",
                   help="movie folder (with 4 *_sparse.mat) OR experiment "
                        "folder (with mov*/ subdirs)")
    p.add_argument("--easywand", default=None,
                   help="path to easyWand .mat (required for build step)")
    p.add_argument("--cam", default=None,
                   help="cam-name substring of the mirror cam to flip "
                        "(required for flip step, e.g. 'cam1' or 'cam5')")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap each movie's h5 at the first N frames (default: all)")
    p.add_argument("--skip-clean", action="store_true")
    p.add_argument("--skip-prescan", action="store_true",
                   help="skip the pre-build sparse-mat scan that filters out "
                        "movies where the fly is rarely tracked by all 4 cams")
    p.add_argument("--prescan-only", action="store_true",
                   help="run only the prescan; do not flip / build / verify")
    p.add_argument("--prescan-min-intersection", type=int, default=500,
                   help="movies with fewer than N frames where all 4 cams "
                        "see the fly are flagged BAD and skipped (default: 500)")
    p.add_argument("--prescan-pixel-threshold", type=int, default=50,
                   help="per-frame non-zero pixel count above which a cam is "
                        "considered to 'see the fly' (default: 50)")
    p.add_argument("--prescan-blob-ratio", type=float, default=0.30,
                   help="prescan: a 2nd connected blob counts as a separate "
                        "fly when its size >= ratio * largest blob (default: "
                        "0.30)")
    p.add_argument("--prescan-blob-distance", type=float, default=100.0,
                   help="prescan: a 2nd blob counts as a separate fly only if "
                        "its centroid is at least this many px from the "
                        "largest blob's centroid (default: 100)")
    p.add_argument("--skip-flip", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the verification step (it runs by default)")
    p.add_argument("--verify-only", action="store_true",
                   help="skip clean / prescan / flip / build; just verify "
                        "existing files")
    p.add_argument("--verify-threshold", type=float, default=15.0,
                   help="per-cam LOO median above which a movie is marked FAIL "
                        "(default: 15.0)")
    p.add_argument("--dry-run", action="store_true",
                   help="print what each step would do, do nothing")
    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"input_dir is not a directory: {args.input_dir}")
    input_dir = os.path.abspath(args.input_dir)

    # Capture everything printed during the run so it can be saved as a
    # report file next to the experiment for later review.
    report_buf = io.StringIO()
    tee = _Tee(sys.stdout, report_buf)
    started_at = datetime.datetime.now().isoformat(timespec="seconds")

    with contextlib.redirect_stdout(tee):
        print(f"================================================================")
        print(f"process_experiment.py run at {started_at}")
        print(f"  input_dir: {input_dir}")
        print(f"  argv:      {' '.join(sys.argv)}")
        print(f"================================================================")

        mode, movies = detect_mode(input_dir)
        print(f"Mode: {mode}; movies queued: "
              f"{', '.join(f'mov{n}' for _, n in movies)}")

        timings_path = _timings_path(input_dir)
        print(f"timings ledger: {timings_path}")

        if args.verify_only:
            run_verify(input_dir, mode, movies, args.verify_threshold,
                       args.dry_run, timings_path=timings_path)
        else:
            if not args.skip_clean:
                run_clean(input_dir, mode, movies, args.dry_run)

            # Prescan runs before FLIP so we don't waste time flipping cams
            # of movies we won't process. It also computes the per-movie build
            # range so BUILD emits only the contiguous all-cams-visible window.
            movie_ranges = {}
            if not args.skip_prescan:
                movies, movie_ranges = run_prescan(
                    input_dir, movies,
                    args.prescan_min_intersection,
                    args.prescan_pixel_threshold,
                    args.prescan_blob_ratio,
                    args.prescan_blob_distance,
                    args.dry_run,
                )
                if not movies:
                    print("\nAll movies flagged BAD by prescan; nothing to do.")
                    _write_report(input_dir, report_buf, args.dry_run)
                    return

            if args.prescan_only:
                _write_report(input_dir, report_buf, args.dry_run)
                return

            if not args.skip_flip:
                if not args.cam:
                    print("\n===== FLIP =====\n  skipped (no --cam given)")
                else:
                    run_flip(movies, args.cam, args.dry_run,
                             timings_path=timings_path)

            if not args.skip_build:
                if not args.easywand:
                    print("\n--easywand is required for the build step "
                          "(or pass --skip-build)")
                    _write_report(input_dir, report_buf, args.dry_run)
                    sys.exit(1)
                if not os.path.isfile(args.easywand):
                    print(f"easyWand .mat not found: {args.easywand}")
                    _write_report(input_dir, report_buf, args.dry_run)
                    sys.exit(1)
                run_build(input_dir, mode, movies,
                          os.path.abspath(args.easywand), args.max_frames,
                          args.dry_run, movie_ranges=movie_ranges,
                          timings_path=timings_path)

            # Verification is on by default; --no-verify skips it. When
            # verify ran and produced a PASS list, restrict the manifest to it
            # so failed-calibration movies don't get queued for prediction.
            if not args.no_verify:
                verified = run_verify(input_dir, mode, movies,
                                      args.verify_threshold, args.dry_run,
                                      timings_path=timings_path)
                if verified is not None:
                    movies = verified

            # Write a manifest of good (post-prescan, post-build, post-verify)
            # movies for `predict_array.sh`.
            if mode == "multi" and not args.skip_build:
                write_good_movies_manifest(input_dir, movies, args.dry_run)

    _write_report(input_dir, report_buf, args.dry_run)


def _write_report(input_dir: str, report_buf: io.StringIO,
                  dry_run: bool) -> None:
    """Save the captured stdout to <input_dir>/process_report.txt. Each new
    run appends with a timestamped header so prior runs aren't lost."""
    if dry_run:
        print("\n(dry-run: would write process_report.txt)")
        return
    report_path = os.path.join(input_dir, "process_report.txt")
    with open(report_path, "a") as f:
        f.write(report_buf.getvalue())
        f.write("\n")
    print(f"\nReport appended to: {report_path}")


if __name__ == "__main__":
    main()
