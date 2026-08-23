"""
process_experiment.py
=====================

End-to-end wrapper around the pose-estimation data-prep pipeline. Drives:

    1. CLEAN — delete extraneous files left over from Roni's hull-reconstruction
       project: *.csv, desktop.ini, hull_op/ and Segmentation/ folders.
    2. PRESCAN — scan source *_sparse.mat files; flag movies where the fly is
       tracked for fewer than --prescan-min-intersection frames (default 500).
       Flagged movies are dropped from FLIP/BUILD/VERIFY. A frame counts as
       tracked only if EVERY cam sees a single fly and at least
       --prescan-min-cams-in-frame of them see it WHOLLY inside their field of
       view (--prescan-min-edge-margin), so the build range stops before the
       fly flies off the edge of too many cameras at once. The prescan also
       writes each movie a `prescan_cam_validity.npz` naming which cams saw the
       whole fly per built frame; prediction uses it to drop the camera pairs
       that include a cut cam.
    3. MIRROR CHECK — test every flip hypothesis against the calibration on
       the raw mats and verify it against what --cam asked for. This is also
       the idempotency check the flip has never had: flipping is self-inverse
       with no marker in the data, so only the calibration can tell "already
       flipped" from "needs flipping". Disagreements abort. --cam auto hands
       the decision to it outright; --no-mirror-check opts out.
    4. FLIP  — vertically flip the mirror camera's sparse .mat (e.g. cam1 in
       2023, cam5 in 2022) using `code/flip_sparse_cam_mat.py`'s logic.
    5. BUILD — invoke MATLAB to produce one h5 per movie plus one shared
       calibration.h5 (same logic as `code/build_experiment.sh`). MATLAB
       writes into a per-movie `.build_tmp/` staging dir and the h5 is moved
       into place only once MATLAB exits 0, so a build that dies partway
       leaves no dataset rather than a silently truncated one. MATLAB's own
       output is captured to `<movie_dir>/build_mov<N>.log`.
    6. VERIFY (optional) — per-movie reprojection-error sanity check using
       `code/verify_calibration.py`'s machinery. Also rejects movies whose
       build never finished (INCOMPLETE) or whose all-cams intersection is
       under --verify-min-intersection frames (BAD_DATA).

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
                            (e.g. 'cam1' for 2023 setups, 'cam5' for 2022), or
                            'auto' to let the MIRROR CHECK decide. Whatever is
                            passed is verified against the calibration first.
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
    --prescan-min-edge-margin N
                            px of clearance the fly's blob must keep from
                            every image border (default: 5, 0 disables). A
                            fly leaving the field of view keeps far more than
                            --prescan-pixel-threshold pixels on its way out,
                            so without this the build range runs on into
                            frames holding half a fly.
    --prescan-min-cams-in-frame N
                            how many cams must see the WHOLE fly for a frame
                            to count (default: 3, 0 = every cam). A cam that
                            sees the fly CUT is tolerated as long as N others
                            see it whole, and the majority may be different
                            cams each frame. Which cams were whole is saved to
                            each movie's prescan_cam_validity.npz so predict
                            can drop the camera pairs a cut cam is in.
    --skip-flip             skip the flip step (e.g. if already flipped)
    --skip-build            skip the h5 build step
    --verify-threshold P    LOO median (px) above which a movie is marked FAIL
                            (default: 15). The known-broken cases give 100-400
                            px; the known-working cases give 2-8 px.
    --verify-min-intersection N
                            mark a BUILT movie BAD_DATA if fewer than N of its
                            frames have all 4 cams tracking (default: 500).
                            The prescan applies the same floor to the raw mats;
                            this re-applies it to what the build actually
                            produced, so a build that died early is caught.
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
- The flip step is NOT idempotent: it is self-inverse, so --cam twice on the
  same data un-does it. The MIRROR CHECK is what now catches that -- a second
  run sees the data already agreeing with the calibration and refuses. Passing
  --no-mirror-check removes that protection.
- The build step requires MATLAB on PATH (override with MATLAB_BIN env var).
- The verify step uses blob-silhouette centroids as the 2D measurement source.
  Typical "working" LOO medians are 2-8 px; "broken" calibrations give 100+.
"""

import argparse
import contextlib
import datetime
import glob
import io
import json
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
from scan_sparse_movies import (DEFAULT_MIN_CAMS_IN_FRAME,
                                DEFAULT_MIN_EDGE_MARGIN,
                                MIN_USABLE_CAMS_IN_FRAME,
                                SUPPORTED_CAM_COUNTS,
                                resolve_min_cams_in_frame,
                                scan_experiment)
from verify_calibration import (
    load_calibration,
    collect_measurements,
    per_cam_errors,
)
from utils import (PERTURBATION_FILE, get_trigger_frame_info, load_perturbation)
from find_mirror_cam import detect_mirror_cam, print_hypothesis_table


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

# Returned by matlab_batch when MATLAB had to be killed for not exiting. 124 is
# the conventional exit code for a timeout (coreutils `timeout` uses it) and is
# distinct from anything MATLAB itself returns.
MATLAB_TIMEOUT_RC = 124
# Budget for one movie's build. Observed throughput is ~10 frames/s, so the
# slack below is ~40x the expected time -- generous enough that a merely slow
# node is never killed, tight enough that a hang does not eat the whole
# walltime. The floor covers short movies where the fixed startup dominates.
BUILD_SECONDS_PER_FRAME = 0.1
BUILD_TIMEOUT_SLACK = 4.0
BUILD_TIMEOUT_FLOOR = 1200          # 20 min
BUILD_TIMEOUT_UNKNOWN = 4 * 3600    # when the frame count is not known upfront

# --cam value that hands the choice to the mirror check rather than naming a
# camera. Spelled like the predict config's "auto" fields for consistency.
AUTO_CAM = "auto"


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
    """Parse N from a 'movN' basename, case-insensitively and ignoring leading
    zeros (mov3, Mov003, MOV03 all -> 3). Returns None on no match."""
    m = re.match(r"^mov(\d+)$", os.path.basename(d), re.IGNORECASE)
    return int(m.group(1)) if m else None


def canonical_movie_dir(movie_dir: str, movie_num: int,
                        dry_run: bool = False) -> str:
    """MATLAB's builder reconstructs the movie subdir as ['mov', int2str(movie_num)]
    — lowercase 'mov', no leading zeros. Raw Windows-exported dirs are named like
    'Mov001', which MATLAB (and the mov* discovery here) won't find. Rename such a
    dir to the canonical <parent>/mov<N> and return the path to use. On a name
    collision, leaves the dir untouched and returns it unchanged."""
    parent = os.path.dirname(movie_dir)
    canon = os.path.join(parent, f"mov{movie_num}")
    if os.path.abspath(movie_dir) == os.path.abspath(canon):
        return movie_dir
    if os.path.exists(canon):
        print(f"  WARNING: cannot rename {os.path.basename(movie_dir)} -> "
              f"mov{movie_num} (target already exists); leaving as-is")
        return movie_dir
    print(f"  {'would rename' if dry_run else 'renaming'} "
          f"{os.path.basename(movie_dir)} -> mov{movie_num} (pipeline convention)")
    if not dry_run:
        os.rename(movie_dir, canon)
        return canon
    return movie_dir


def detect_mode(input_dir: str, dry_run: bool = False) -> tuple:
    """Returns ('single', [(movie_dir, movie_num)]) or
       ('multi', [(movie_dir, movie_num), ...]) or sys.exits on error.
    Non-canonical movie dirs (e.g. 'Mov001') are renamed to 'mov<N>' first, since
    the whole downstream pipeline (MATLAB build included) expects that convention.
    """
    n_direct = count_sparse_mats(input_dir)
    if n_direct in SUPPORTED_CAM_COUNTS:
        mn = parse_movie_num(input_dir)
        if mn is None:
            sys.exit(f"Single-movie mode requires a 'mov<N>' basename; got "
                     f"'{os.path.basename(input_dir)}'")
        input_dir = canonical_movie_dir(input_dir, mn, dry_run)
        return "single", [(input_dir, mn)]
    if n_direct != 0:
        sys.exit(f"Ambiguous: {input_dir} contains {n_direct} *_sparse.mat "
                 f"(expected 0, "
                 f"{' or '.join(map(str, SUPPORTED_CAM_COUNTS))})")
    # Multi-movie mode. Use os.listdir (not a case-sensitive glob) so 'Mov001'
    # from a Windows export is discovered alongside canonical 'mov1'.
    movies = []
    incomplete = []   # (name, n_sparse) for mov dirs without exactly 4 mats
    for name in sorted(os.listdir(input_dir)):
        sub = os.path.join(input_dir, name)
        if not os.path.isdir(sub):
            continue
        mn = parse_movie_num(sub)
        if mn is None:
            continue
        n = count_sparse_mats(sub)
        if n in SUPPORTED_CAM_COUNTS:
            sub = canonical_movie_dir(sub, mn, dry_run)
            movies.append((sub, mn))
        else:
            print(f"  (skipping {sub}: {n} *_sparse.mat, expected "
                  f"{' or '.join(map(str, SUPPORTED_CAM_COUNTS))})")
            incomplete.append((os.path.basename(sub), n))
    movies.sort(key=lambda t: t[1])
    if incomplete:
        print(f"Skipped {len(incomplete)} incomplete movie(s) "
              f"(missing/extra *_sparse.mat — dropped from the whole pipeline): "
              + ", ".join(f"{name}({n})" for name, n in incomplete))
    if not movies:
        sys.exit(f"No 'mov<N>/' subdirs with "
                 f"{' or '.join(map(str, SUPPORTED_CAM_COUNTS))} "
                 f"*_sparse.mat in {input_dir}")
    return "multi", movies


def detect_num_cams(movies: list, override: "int | None" = None) -> int:
    """How many cameras this experiment was recorded with.

    Taken from the number of *_sparse.mat per movie dir, and required to be
    the SAME for every movie: a mismatch means some movie is missing a
    camera's export, which would silently produce a box with a blank camera
    and a calibration that no longer lines up. That is a data error worth
    aborting on, not something to paper over per movie."""
    counts = {}
    for movie_dir, mn in movies:
        counts.setdefault(count_sparse_mats(movie_dir), []).append(f"mov{mn}")
    if len(counts) > 1:
        detail = "; ".join(f"{n} cams: {', '.join(ms[:8])}"
                           + (" ..." if len(ms) > 8 else "")
                           for n, ms in sorted(counts.items()))
        sys.exit(f"Movies disagree on camera count ({detail}). Every movie in "
                 f"an experiment must have the same cameras.")
    detected = next(iter(counts))
    if override is not None and override != detected:
        print(f"  WARNING: --num-cams {override} overrides the {detected} "
              f"*_sparse.mat found per movie dir")
        return override
    return detected


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
MIRROR_CHECK_MOVIES = 3     # enough to average out one odd movie; ~5-8 s each


def run_mirror_check(movies: list, easywand: "str | None", requested_cam,
                     dry_run: bool) -> "str | None":
    """Decide, or verify, which camera needs the vertical flip.

    Runs BEFORE the flip, on the raw mats, so its answer is valid whatever the
    caller asked for. It is also the idempotency check the flip step has never
    had: flipping is a self-inverse operation with no marker in the data, so
    the only way to tell "already flipped" from "needs flipping" is to ask the
    calibration -- which is exactly what this does. `--cam` twice on the same
    experiment used to silently un-flip it; now it stops.

    Returns the cam name to flip, or None for "flip nothing". Exits when the
    detection unambiguously contradicts `requested_cam`, since every such case
    corrupts data: flipping an already-correct camera, flipping the wrong one,
    or skipping a flip the data needs.
    """
    print("\n===== MIRROR CHECK =====")
    if not easywand:
        print("  skipped: no --easywand to check the cameras against")
        return requested_cam if requested_cam != AUTO_CAM else None
    sample = movies[::max(1, len(movies) // MIRROR_CHECK_MOVIES)][:MIRROR_CHECK_MOVIES]
    verdict = detect_mirror_cam([d for d, _ in sample], easywand=easywand)
    if verdict is None:
        print("  skipped: no usable measurements (could not read the mats?)")
        return requested_cam if requested_cam != AUTO_CAM else None
    print_hypothesis_table(verdict)

    asked = None if requested_cam in (None, AUTO_CAM) else requested_cam
    is_auto = requested_cam == AUTO_CAM

    if not verdict["conclusive"]:
        # Inconclusive means the evidence cannot settle it -- usually a
        # calibration problem, which no flip repairs. Aborting would help
        # nobody, so say so loudly and do as asked.
        print(f"  INCONCLUSIVE -- {verdict['reason']}")
        if is_auto:
            sys.exit("--cam auto cannot decide; pass an explicit --cam or "
                     "--skip-flip after resolving the above")
        print(f"  proceeding as asked ({'flip ' + asked if asked else 'no flip'})")
        return asked

    detected = verdict["flip"][0] if verdict["flip"] else None
    print(f"  detected: {'flip ' + detected if detected else 'NO flip needed'} "
          f"({verdict['worst']:.2f} px vs {verdict['runner_up']:.2f} px for the "
          f"next hypothesis)")
    if is_auto:
        print(f"  --cam auto -> {'flipping ' + detected if detected else 'skipping the flip'}")
        return detected
    if asked == detected:
        print("  agrees with what you asked for.")
        return asked
    if asked and detected is None:
        sys.exit(f"\nREFUSING TO FLIP {asked}: its data already agrees with the "
                 f"calibration.\nFlipping it now would BREAK it. This is what a "
                 f"second --cam run on already-flipped data looks like.\n"
                 f"Use --skip-flip, or --no-mirror-check to override.")
    if asked and detected:
        sys.exit(f"\nREFUSING TO FLIP {asked}: the evidence says {detected} is "
                 f"the mirror camera.\nUse --cam {detected}, --cam auto, or "
                 f"--no-mirror-check to override.")
    sys.exit(f"\nREFUSING TO SKIP THE FLIP: {detected} disagrees with the "
             f"calibration and needs flipping.\nUse --cam {detected}, --cam "
             f"auto, or --no-mirror-check to override.")


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
                             end_ind: "int | None" = None,
                             num_cams: "int | None" = None) -> str:
    # Absolute paths are mandatory, not cosmetic: the command ends in
    # `run('<abs>/CreateDatasetHDF5_from_list_fixed.m')`, and MATLAB's `run`
    # changes the working directory to the script's own folder for the
    # duration of the script. A relative sparse_folder_path/save_path would
    # therefore be resolved against matlab/ and fail with a confusing
    # "File or folder not found" from h5create.
    sparse_folder_path = os.path.abspath(sparse_folder_path)
    save_path = os.path.abspath(save_path)
    cmd = (f"sparse_folder_path='{sparse_folder_path}'; "
           f"save_path='{save_path}'; "
           f"movie_num={movie_num};")
    # The builder defaults num_cams to 4; a 3-camera rig must say so, or it
    # indexes a 4th camera that has no *_sparse.mat.
    if num_cams is not None:
        cmd += f" num_cams={num_cams};"
    if start_ind is not None:
        cmd += f" start_ind={start_ind};"
    if end_ind is not None:
        cmd += f" end_ind={end_ind};"
    if max_frames is not None:
        cmd += f" max_frames={max_frames};"
    cmd += f" run('{DATASET_SCRIPT}')"
    return cmd


def build_calibration_matlab_cmd(easywand_path: str, save_path: str) -> str:
    # Absolute for the same reason as build_dataset_matlab_cmd: `run` cds into
    # the script's folder before the script sees these variables.
    return (f"easy_wand_path='{os.path.abspath(easywand_path)}'; "
            f"savePath='{os.path.abspath(save_path)}'; "
            f"run('{CALIB_SCRIPT}')")


def _as_text(stream) -> str:
    """TimeoutExpired can carry bytes even when the run was text=True."""
    if stream is None:
        return ""
    return stream.decode(errors="replace") if isinstance(stream, bytes) else stream


def matlab_batch(cmd: str, dry_run: bool, log_path: "str | None" = None,
                 tail_on_error: int = 40,
                 timeout: "float | None" = None) -> int:
    """Run `matlab -batch <cmd>`, capturing its output instead of losing it.

    subprocess hands the child the inherited OS fd 1, while the report only
    captures Python's `sys.stdout` object -- so MATLAB's own stdout/stderr used
    to sail straight past process_report.txt. Every build failure in the
    archive was recorded as a bare "Failed: mov<N>" with not one line of
    MATLAB's output to explain it. Keep the full stream in `log_path` and echo
    its tail into the report on failure, so the report alone is diagnostic.

    `timeout` guards against a MATLAB that never exits. `-batch` is supposed to
    quit when the command returns, but it has been observed finishing its work
    and then sitting idle forever (see build_one_movie). With no timeout the
    parent blocks on it for the whole SLURM walltime and every remaining movie
    in the run is silently starved. Returns MATLAB_TIMEOUT_RC in that case.
    """
    if dry_run:
        print(f"  would run MATLAB: {cmd}")
        return 0
    print(f"  MATLAB: {cmd}", flush=True)
    try:
        result = subprocess.run([MATLAB_BIN, "-batch", cmd],
                                capture_output=True, text=True, timeout=timeout)
        returncode, output = result.returncode, (_as_text(result.stdout)
                                                 + _as_text(result.stderr))
    except subprocess.TimeoutExpired as e:
        # subprocess.run has already killed the child by the time this lands.
        returncode = MATLAB_TIMEOUT_RC
        output = _as_text(e.stdout) + _as_text(e.stderr)
        print(f"   MATLAB still alive after {timeout:.0f}s and was killed; "
              f"treating as a hang, not a build error")
    if log_path:
        try:
            with open(log_path, "w") as f:
                f.write(output)
            print(f"   MATLAB log: {log_path}")
        except OSError as e:
            print(f"   (could not write MATLAB log {log_path}: {e})")
    if returncode != 0:
        lines = output.splitlines()
        if returncode == MATLAB_TIMEOUT_RC:
            note = f"   MATLAB timed out after {timeout:.0f}s"
        else:
            # A negative returncode means a signal (e.g. -9 = SIGKILL, the
            # signature of an OOM kill or an external teardown).
            signal_note = (f" (killed by signal {-returncode})"
                           if returncode < 0 else "")
            note = f"   MATLAB exited {returncode}{signal_note}"
        print(note)
        if lines:
            print(f"   last {min(tail_on_error, len(lines))} line(s) of its output:")
            for line in lines[-tail_on_error:]:
                print(f"     | {line}")
        else:
            print("     | (no output at all — died before writing anything)")
    return returncode


def build_one_movie(sparse_folder_path: str, movie_dir: str, movie_num: int,
                    max_frames: "int | None" = None,
                    start_ind: "int | None" = None,
                    end_ind: "int | None" = None,
                    dry_run: bool = False,
                    num_cams: "int | None" = None) -> "tuple[int, int | None]":
    """Build one movie's dataset h5, staged so a partial build never lands.

    MATLAB writes into a private `.build_tmp/` inside the movie dir; the h5 is
    moved into place with os.replace only once MATLAB exits 0. The builder
    appends to extendable datasets with no atomic commit, so a build that dies
    partway leaves a well-formed but truncated h5 that nothing downstream can
    distinguish from a complete one. Staging confines that debris to a
    directory nobody looks in (find_movie_h5 globs the movie dir itself, not
    sub-dirs), turning "silently truncated dataset" into "no dataset" -- which
    the pipeline already handles correctly. It also sidesteps the h5create
    collision that makes a rebuild over an existing h5 fail outright, since
    MATLAB always writes into a fresh empty directory.

    Returns (returncode, n_frames_built). Split out of run_build so a targeted
    rebuild of individual movies runs through exactly this code path.
    """
    build_dir = os.path.join(movie_dir, ".build_tmp")
    log_path = os.path.join(movie_dir, f"build_mov{movie_num}.log")
    if not dry_run:
        shutil.rmtree(build_dir, ignore_errors=True)
        os.makedirs(build_dir, exist_ok=True)
    cmd = build_dataset_matlab_cmd(sparse_folder_path, build_dir, movie_num,
                                   max_frames, start_ind, end_ind, num_cams)
    if start_ind is not None and end_ind is not None:
        n_planned = end_ind - start_ind + 1
    else:
        n_planned = max_frames
    timeout = (max(BUILD_TIMEOUT_FLOOR,
                   n_planned * BUILD_SECONDS_PER_FRAME * BUILD_TIMEOUT_SLACK)
               if n_planned else BUILD_TIMEOUT_UNKNOWN)
    rc = matlab_batch(cmd, dry_run, log_path=None if dry_run else log_path,
                      timeout=timeout)
    if rc == MATLAB_TIMEOUT_RC and not dry_run:
        # A hang is not the same as a failed build. MATLAB has been seen to
        # write a complete, verifiable dataset and only then fail to exit
        # (100ms mov26: h5 finished at 13:06, the process still alive and
        # burning no CPU 1h45m later). Throwing that away would waste a good
        # build and, worse, leave the movie with no h5 at all now that the old
        # one is archived. Commit it if -- and only if -- it is complete.
        staged = sorted(glob.glob(os.path.join(build_dir,
                                               "mov_*_ds_*tc_*tj.h5")))
        if not staged:
            print("   hung before writing any h5; treating as failed")
        else:
            try:
                with h5py.File(staged[0], "r") as f:
                    partial = check_build_complete(f)
            except Exception as e:                      # unreadable == unusable
                partial = {"error": str(e)}
            if partial is None:
                print("   hung AFTER writing a complete dataset; committing it")
                rc = 0
            else:
                print(f"   hung with an incomplete dataset ({partial}); "
                      f"discarding")
    if rc == 0 and not dry_run:
        staged = sorted(glob.glob(os.path.join(build_dir,
                                               "mov_*_ds_*tc_*tj.h5")))
        if staged:
            # os.replace is atomic within one filesystem, and build_dir lives
            # inside movie_dir, so there is no window in which a half-copied
            # file is visible under the committed name.
            final = os.path.join(movie_dir, os.path.basename(staged[0]))
            os.replace(staged[0], final)
            print(f"   committed: {final}")
        else:
            print("   MATLAB exited 0 but produced no h5; treating as failed")
            rc = 1
    if not dry_run:
        # The staged remains of a failed build are pure waste (hundreds of MB);
        # the frame counts in the MATLAB log say how far it got.
        shutil.rmtree(build_dir, ignore_errors=True)
    n_built = None
    if rc == 0 and not dry_run:
        built = find_movie_h5(movie_dir)
        if built is not None:
            try:
                with h5py.File(built, "r") as f:
                    n_built = int(f["cropzone"].shape[0])
            except Exception:
                pass
    return rc, n_built


def run_build(input_dir: str, mode: str, movies: list,
              easywand: str, max_frames: "int | None", dry_run: bool,
              movie_ranges: "dict | None" = None,
              timings_path: "str | None" = None,
              num_cams: "int | None" = None) -> list:
    """Runs the MATLAB builder per movie. Returns the list of (movie_dir,
    movie_num) whose build exited non-zero, so the caller can keep them out of
    VERIFY and the manifest -- a failed build still leaves whatever frames it
    managed to write on disk, and that partial h5 is indistinguishable from a
    complete one unless we remember that MATLAB died."""
    print("\n===== BUILD =====")
    sparse_folder_path = input_dir if mode == "multi" else os.path.dirname(input_dir)
    n_ok = n_fail = 0
    failed = []
    failed_movies = []
    for movie_dir, mn in movies:
        start_ind = end_ind = None
        if movie_ranges and movie_dir in movie_ranges:
            start_ind, end_ind = movie_ranges[movie_dir]
        print(f"\n-- mov{mn}")
        print(f"   sparse_folder_path={sparse_folder_path}")
        print(f"   save_path={movie_dir}")
        # The builder indexes cameras by the ALPHABETICAL order of these
        # files, so this order is what camera index 0..N-1 means downstream
        # and must match the easyWand's camera order. Print it: a rig whose
        # mat names sort differently is otherwise invisible until the
        # reprojection errors come out nonsensical.
        print(f"   cameras (index order): "
              + ", ".join(os.path.basename(m) for m in
                          sorted(glob.glob(os.path.join(movie_dir,
                                                        "*_sparse.mat")))))
        if start_ind is not None:
            print(f"   build range: start_ind={start_ind}, end_ind={end_ind} "
                  f"({end_ind - start_ind + 1} frames)")
        t0 = time.time()
        rc, n_built = build_one_movie(sparse_folder_path, movie_dir, mn,
                                      max_frames, start_ind, end_ind, dry_run,
                                      num_cams=num_cams)
        t1 = time.time()
        record_timing(timings_path, f"mov{mn}", "build", t0, t1,
                      n_frames=n_built)
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
            failed.append(f"mov{mn}")
            failed_movies.append((movie_dir, mn))
    print(f"\nBuild dataset: {n_ok} ok, {n_fail} failed.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        print("  (nothing was committed for these — whatever the builder had "
              "written was discarded with its staging dir, so they have no "
              "dataset h5 and are excluded from VERIFY and the manifest)")
        print("  (see build_mov<N>.log in each movie dir for MATLAB's output)")

    # Build the calibration.h5 inside input_dir (matches build_experiment.sh).
    calib_out = os.path.join(input_dir, "calibration.h5")
    print(f"\n-- calibration.h5 → {calib_out}")
    if os.path.isfile(calib_out) and not dry_run:
        print(f"   (removing pre-existing {calib_out} to avoid h5create collision)")
        os.remove(calib_out)
    cmd = build_calibration_matlab_cmd(easywand, calib_out)
    matlab_batch(cmd, dry_run,
                 log_path=None if dry_run else os.path.join(
                     input_dir, "build_calibration.log"))

    return failed_movies


# ---------------------------------------------------------------------------
# Prescan + manifest helpers
# ---------------------------------------------------------------------------
# The MATLAB builder needs `time_jump` frames of padding before start_ind and
# after end_ind for its time-channel windows; keep this in sync with the
# script's `time_jump` (=7).
MATLAB_TIME_JUMP_MARGIN = 7

# Fewest frames with all 4 cams tracking the fly for a movie to be worth
# processing. Applied twice: by the prescan to the raw sparse mats (so we never
# build a hopeless movie) and by verify to the h5 the build actually produced
# (so a build that died partway is caught rather than passed on with a fraction
# of its frames).
DEFAULT_MIN_INTERSECTION = 500


def run_prescan(input_dir: str, movies: list,
                min_intersection: int, pixel_threshold: int,
                blob_ratio: float, blob_distance: float,
                dry_run: bool,
                min_edge_margin: float = DEFAULT_MIN_EDGE_MARGIN,
                min_cams_in_frame=DEFAULT_MIN_CAMS_IN_FRAME) -> tuple:
    """Returns (filtered_movies, movie_ranges, scan_results):
      - filtered_movies: list of (movie_dir, movie_num) — only OK movies
      - movie_ranges:    dict {movie_dir: (start_ind, end_ind)} in MATLAB's
                         1-based inclusive convention, clamped to leave
                         `MATLAB_TIME_JUMP_MARGIN` frames of padding so the
                         builder's time-channel windows stay in range.
      - scan_results:    dict {movie_dir: scan result}, carrying the per-frame
                         per-cam masks that write_cam_validity_sidecar slices
                         into the movie's npz once BUILD has committed an h5.
    BAD movies are reported to stdout and dropped from subsequent steps.

    The range is also what trims a movie before the fly flies out of the
    field of view: frames where fewer than `min_cams_in_frame` cams see the
    whole fly fail the scan's in-frame test, so the longest contiguous run
    ends there and BUILD never reaches the truncated-fly tail."""
    print(f"\n===== PRESCAN =====")
    results = scan_experiment(input_dir, min_intersection,
                              pixel_threshold, blob_ratio, blob_distance,
                              print_results=True,
                              min_edge_margin=min_edge_margin,
                              min_cams_in_frame=min_cams_in_frame)
    ok_dirs = {r["movie_dir"] for r in results if r["verdict"] == "OK"}
    n_before = len(movies)
    filtered = [(d, n) for d, n in movies if d in ok_dirs]
    n_dropped = n_before - len(filtered)
    if n_dropped:
        action = "would drop" if dry_run else "dropping"
        print(f"\nPrescan: {action} {n_dropped} BAD movie(s) "
              f"from FLIP / BUILD / VERIFY steps")
    movie_ranges = {}
    scan_results = {}
    for r in results:
        if r["verdict"] != "OK":
            continue
        scan_results[r["movie_dir"]] = r
        # 0-based [good_start, good_end) -> 1-based inclusive [start, end].
        # Clamp to leave time-jump padding both ends (MATLAB indexes
        # `(start_ind - time_jump):(end_ind + time_jump)` from the raw mat).
        start_ind = max(r["good_start"] + 1, MATLAB_TIME_JUMP_MARGIN + 1)
        end_ind = min(r["good_end"], r["n_frames"] - MATLAB_TIME_JUMP_MARGIN)
        if start_ind <= end_ind:
            movie_ranges[r["movie_dir"]] = (start_ind, end_ind)
    return filtered, movie_ranges, scan_results


# Per-movie record of which cams saw the WHOLE fly at each BUILT frame.
# Prediction reads it to drop the camera pairs that include a cut cam; without
# it the relaxed --prescan-min-cams-in-frame rule would simply admit garbage.
CAM_VALIDITY_SIDECAR = "prescan_cam_validity.npz"


def parse_h5_range(h5_path: str) -> "tuple | None":
    """(start_ind, end_ind) as encoded in a built h5's filename
    (`mov_<n>_<start>_<end>_ds_..`), or None if it doesn't match."""
    m = re.match(r"mov_(\d+)_(\d+)_(\d+)_ds_", os.path.basename(h5_path))
    return (int(m.group(2)), int(m.group(3))) if m else None


def write_cam_validity_sidecar(movie_dir: str, scan_result: dict,
                               dry_run: bool = False,
                               params: "dict | None" = None) -> "str | None":
    """Slice the prescan's per-frame per-cam masks to the frames the build
    actually committed, and save them next to the movie.

    Frame alignment: the prescan masks are indexed by raw 0-based frame, while
    the box holds the raw 1-based inclusive window [start_ind, end_ind]. The
    MATLAB builder loads `frames((start_ind-time_jump):(end_ind+time_jump))`
    and starts its loop at `1+time_jump`, so box frame k (0-based) is raw
    1-based frame `start_ind + k` -- the same convention
    utils.get_trigger_frame_info documents. Hence the slice starts at
    `start_ind - 1`.

    The committed range is read back from the h5 rather than taken from the
    planned range, because --max-frames and build_one_movie's hang-recovery
    path can both change what actually landed. Returns the path written, or
    None when there is nothing to write."""
    if dry_run:
        return None
    h5 = find_movie_h5(movie_dir)
    if h5 is None:
        return None
    rng = parse_h5_range(h5)
    if rng is None:
        print(f"   (no cam-validity sidecar: cannot parse range from "
              f"{os.path.basename(h5)})")
        return None
    start_ind, _ = rng
    try:
        with h5py.File(h5, "r") as f:
            n_box = int(f["cropzone"].shape[0])
    except Exception as e:
        print(f"   (no cam-validity sidecar: cannot read {h5}: {e})")
        return None
    lo = start_ind - 1
    hi = lo + n_box
    masks = {}
    for key, name in (("in_frame_mask", "in_frame"),
                      ("visible_mask", "visible"),
                      ("single_mask", "single")):
        m = scan_result.get(key)
        if m is None:
            print("   (no cam-validity sidecar: prescan returned no masks)")
            return None
        masks[name] = m[lo:hi]
    n_got = masks["in_frame"].shape[0]
    if n_got != n_box:
        # Would silently mis-attribute validity to the wrong frames; a wrong
        # mask is worse than no mask, so refuse rather than pad.
        print(f"   (no cam-validity sidecar: prescan covers {n_got} of the "
              f"{n_box} built frames from start_ind={start_ind})")
        return None
    out = os.path.join(movie_dir, CAM_VALIDITY_SIDECAR)
    meta = dict(params or {})
    meta.update({"start_ind": start_ind, "n_box_frames": n_box,
                 "min_cams_in_frame": scan_result.get("min_cams_in_frame"),
                 "n_cams": scan_result.get("n_cams")})
    np.savez_compressed(
        out,
        mat_names=np.array([os.path.basename(m)
                            for m in scan_result.get("mats", [])]),
        params=json.dumps(meta),
        **masks)
    return out


def run_cam_validity_sidecars(movies: list, scan_results: dict,
                              dry_run: bool, params: "dict | None" = None) -> None:
    """Write a cam-validity sidecar for every movie that has both a prescan
    result and a committed h5."""
    print("\n===== CAM VALIDITY =====")
    if dry_run:
        print("  (dry-run: would write per-movie "
              f"{CAM_VALIDITY_SIDECAR})")
        return
    if not scan_results:
        print("  skipped (no prescan results — run without --skip-prescan "
              "to record which cams saw the whole fly)")
        return
    n_ok = n_skip = 0
    for movie_dir, mn in movies:
        r = scan_results.get(movie_dir)
        if r is None:
            n_skip += 1
            continue
        print(f"-- mov{mn}")
        if write_cam_validity_sidecar(movie_dir, r, dry_run, params):
            n_ok += 1
            hist = r.get("in_frame_cam_histogram") or []
            print(f"   {CAM_VALIDITY_SIDECAR}  "
                  f"(whole-fly cams per frame over the whole movie: "
                  + ", ".join(f"{j}={hist[j]}"
                              for j in range(len(hist) - 1, -1, -1)) + ")")
        else:
            n_skip += 1
    print(f"Cam validity: {n_ok} written, {n_skip} skipped.")


# ---------------------------------------------------------------------------
# Perturbation declaration
# ---------------------------------------------------------------------------
def run_perturbation_declaration(input_dir: str, args, dry_run: bool) -> "dict | None":
    """Write (or validate) the experiment's perturbation.json.

    Placed beside calibration.h5 so the declaration travels with the data, and
    so its mere presence is what marks the experiment -- predict needs no flag
    of its own. An existing file is KEPT unless --perturbation-force: the CLI
    can only express one window for the whole experiment, while a hand-authored
    file can carry per-movie windows and provenance, and silently replacing
    that with the poorer version would be a data loss.
    """
    print("\n===== PERTURBATION =====")
    path = os.path.join(input_dir, PERTURBATION_FILE)
    if os.path.isfile(path) and not args.perturbation_force:
        try:
            with open(path) as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  {PERTURBATION_FILE} exists but could not be read: {e}")
            print(f"  fix it or pass --perturbation-force to replace it")
            return None
        pert = (doc.get("perturbation") or {})
        dur = pert.get("duration_ms")
        print(f"  keeping the existing declaration: {path}")
        print(f"    type    : {pert.get('type', 'unspecified')}")
        print(f"    onset   : trigger frame "
              f"{pert.get('onset_trigger_frame', pert.get('onset_frame', 0))}")
        print(f"    duration: {f'{dur:g} ms' if dur is not None else 'NOT RECORDED'}"
              + ("" if dur is not None else " -> frames from the onset on will "
                                            "be labelled 'unknown'"))
        print(f"  (pass --perturbation-force to replace it with the CLI values)")
        return doc

    doc = {
        "experiment": os.path.basename(input_dir.rstrip(os.sep)),
        "source": "declared by process_experiment.py --perturbation",
        "perturbation": {
            "type": args.perturbation_type,
            "onset_trigger_frame": args.perturbation_onset_frame,
            "onset_status": ("Trigger-relative: frame 0 is the hardware trigger. "
                             "Exact for every movie -- the build range is "
                             "reconciled against it downstream via frame_index."),
            "duration_ms": args.perturbation_duration_ms,
            "duration_status": ("recorded" if args.perturbation_duration_ms is not None
                                else "NOT RECORDED -- frames from the onset onward "
                                     "are labelled 'unknown' rather than guessed"),
        },
    }
    if dry_run:
        print(f"  would write {path}")
        print(f"    {json.dumps(doc['perturbation'])}")
        return doc
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  wrote {path}")
    dur = args.perturbation_duration_ms
    print(f"    type={args.perturbation_type}  "
          f"onset=trigger frame {args.perturbation_onset_frame}  "
          f"duration={f'{dur:g} ms' if dur is not None else 'NOT RECORDED'}")
    return doc


def report_perturbation_coverage(movies: list, dry_run: bool) -> None:
    """Say, per movie, whether the built range actually contains the onset.

    The prescan picks its range from fly visibility and knows nothing about the
    perturbation, so a movie can legitimately start after the onset -- and then
    no frame in it is labelled 'before'. Surfacing the count here means it is
    visible before the GPU array runs, instead of being discovered as a hole in
    the analysis much later."""
    if dry_run:
        print("\n(dry-run: would report perturbation coverage)")
        return
    print("\n===== PERTURBATION COVERAGE =====")
    n_have = n_unknown = 0
    late, early = [], []
    for movie_dir, mn in movies:
        h5 = find_movie_h5(movie_dir)
        if h5 is None:
            continue
        trig_off, frame_rate = get_trigger_frame_info(h5)
        pert = load_perturbation(h5, frame_rate)
        if pert is None or trig_off is None:
            n_unknown += 1
            continue
        try:
            with h5py.File(h5, "r") as f:
                n_frames = int(f["cropzone"].shape[0])
        except OSError:
            n_unknown += 1
            continue
        last = trig_off + n_frames - 1
        onset = pert["onset_frame"]
        if trig_off <= onset <= last:
            n_have += 1
        elif trig_off > onset:
            late.append(f"mov{mn}({trig_off}..{last})")
        else:
            early.append(f"mov{mn}({trig_off}..{last})")
    print(f"  onset (trigger frame {pert['onset_frame']}) inside the built "
          f"range: {n_have} movie(s)")
    # The two ways of missing the onset are opposite problems and only one of
    # them costs you a baseline, so they are worth separating.
    if late:
        print(f"  starts AFTER the onset: {len(late)} movie(s) — no "
              f"pre-perturbation frames, so no within-movie baseline:")
        print("    " + ", ".join(late))
    if early:
        print(f"  ends BEFORE the onset: {len(early)} movie(s) — entirely "
              f"pre-perturbation:")
        print("    " + ", ".join(early))
    if n_unknown:
        print(f"  could not determine: {n_unknown} movie(s)")


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


def check_build_complete(f) -> "dict | None":
    """Detect a movie h5 the MATLAB builder never finished writing.

    The builder creates `best_frames_mov_idx` up front at the full requested
    size, then appends to the extendable `box` / `cropzone` / `frameInds`
    datasets one frame at a time, in that order, with no atomic commit
    (CreateDatasetHDF5_from_list_fixed.m). A build that dies partway therefore
    leaves a perfectly well-formed HDF5 file holding only the frames it got to
    — nothing in it says "incomplete". Two signatures give it away:

      - the three per-frame datasets disagree in length: killed mid-frame,
        between two of the three h5write calls;
      - they agree but fall short of `best_frames_mov_idx`: killed cleanly
        between frames.

    Returns None when the build looks complete, else a dict of the counts.
    """
    n_box = int(f["box"].shape[0])
    n_crop = int(f["cropzone"].shape[0])
    n_find = int(f["frameInds"].shape[0])
    n_expected = (int(f["best_frames_mov_idx"].shape[-1])
                  if "best_frames_mov_idx" in f else None)
    ragged = not (n_box == n_crop == n_find)
    short = n_expected is not None and n_crop < n_expected
    if not (ragged or short):
        return None
    return {"n_box": n_box, "n_cropzone": n_crop, "n_frameinds": n_find,
            "n_expected": n_expected, "ragged": ragged}


def verify_one_movie(h5_path: str, calib_path: str, threshold: float,
                     image_height: int = 800, num_cams: "int | None" = None,
                     subsample_frames: int = 500,
                     min_intersection: int = DEFAULT_MIN_INTERSECTION) -> tuple:
    """Verify a movie's calibration by reprojection-error.

    Samples from the intersection of frames where ALL cameras have a tracked
    blob (cropzone != [1,1]). Without this, the verify can sample frames where
    a camera didn't capture the fly yet, producing NaN / artificially huge
    errors that look like calibration failures but are actually just data gaps.

    Returns (status, medians, info) where:
      - status: 'PASS' | 'FAIL' | 'INCOMPLETE' | 'BAD_DATA' | 'ERR'
      - medians: per-cam LOO medians (list[float]) for PASS/FAIL,
                 None for INCOMPLETE/BAD_DATA, error message list for ERR.
      - info: dict with diagnostic counts, or None.

    INCOMPLETE outranks the calibration checks on purpose: a truncated build
    reprojects perfectly well over the frames it did write (mov40 scored
    2.3-4.3 px on 143 of 3085 frames), so geometry alone can never catch it.
    """
    try:
        M = load_calibration(calib_path)
    except Exception as e:
        return "ERR", [f"calib load err: {e}"], None
    try:
        with h5py.File(h5_path, "r") as f:
            partial = check_build_complete(f)
            if partial is not None:
                return "INCOMPLETE", None, partial
            cropzone_full = f["cropzone"][:]
            # The box knows its own camera count; trusting a default of 4 here
            # silently mis-reads a 3-camera movie.
            if num_cams is None:
                num_cams = int(cropzone_full.shape[1])
            n_total = int(cropzone_full.shape[0])
            # All-cams intersection: frames where every cam has a real blob
            # (the MATLAB builder writes cropzone=[1,1] when blob detection failed).
            bad_per_cam = ((cropzone_full[:, :, 0] == 1) &
                           (cropzone_full[:, :, 1] == 1))[:, :num_cams]
            all_valid = ~bad_per_cam.any(axis=1)
            valid_idx = np.where(all_valid)[0]
            n_intersection = int(len(valid_idx))
            info = {"n_intersection": n_intersection, "n_total": n_total,
                    "num_cams": num_cams}
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

    if M.shape[0] < num_cams:
        return "ERR", [f"calibration has {M.shape[0]} cams, movie has "
                       f"{num_cams}"], None
    meas, valid = collect_measurements(box, cropzone, image_height, num_cams)
    errs = per_cam_errors(meas, valid, M, mode="loo")
    medians = [float(np.nanmedian(errs[:, c])) for c in range(num_cams)]
    passed = all(m < threshold for m in medians if not np.isnan(m))
    return ("PASS" if passed else "FAIL"), medians, info


def run_verify(input_dir: str, mode: str, movies: list,
               threshold: float, dry_run: bool,
               timings_path: "str | None" = None,
               min_intersection: int = DEFAULT_MIN_INTERSECTION,
               build_failed: "list | None" = None,
               num_cams: "int | None" = None) -> "list | None":
    """Returns the list of (movie_dir, movie_num) that PASSED, or None when
    no filtering happened (calibration missing, or dry-run) so the caller
    falls back to the input list.

    `build_failed` is run_build's list of movies whose MATLAB build exited
    non-zero. They are reported and dropped without being opened: their h5 (if
    any) holds only the frames the builder got to before it died."""
    print(f"\n===== VERIFY (threshold: {threshold:.1f} px per-cam LOO median, "
          f"min intersection: {min_intersection} frames) =====")
    calib_path = find_calibration_h5(input_dir, mode, movies)
    if calib_path is None:
        print(f"  ERROR: calibration.h5 not found near {input_dir}; skipping verify.")
        return None
    print(f"  using calibration: {calib_path}")
    if dry_run:
        print(f"  would verify {len(movies)} movies against {calib_path}")
        return None

    passed_movies = []
    n_pass = n_fail = n_bad = n_err = n_missing = n_incomplete = 0
    failed_lines = []
    bad_lines = []
    incomplete_lines = []
    build_failed_dirs = {d for d, _ in (build_failed or [])}
    for movie_dir, mn in movies:
        if movie_dir in build_failed_dirs:
            n_incomplete += 1
            print(f"  [mov{mn}] BUILD_FAILED — MATLAB exited non-zero during "
                  f"BUILD; any h5 on disk is partial; skipping")
            incomplete_lines.append(f"mov{mn}  build exited non-zero (rebuild)")
            continue
        h5 = find_movie_h5(movie_dir)
        if h5 is None:
            print(f"  [mov{mn}] no dataset h5 found; skipping")
            n_missing += 1
            continue
        t0 = time.time()
        status, medians, info = verify_one_movie(h5, calib_path, threshold,
                                                 num_cams=num_cams,
                                                 min_intersection=min_intersection)
        t1 = time.time()
        n_inter = info["n_intersection"] if (info and "n_intersection" in info) else None
        record_timing(timings_path, f"mov{mn}", "verify", t0, t1,
                      n_frames=n_inter)
        if status == "ERR":
            n_err += 1
            print(f"  [mov{mn}] ERR  {medians}")
            failed_lines.append(f"mov{mn} load error: {medians}")
            continue
        if status == "INCOMPLETE":
            n_incomplete += 1
            exp = info["n_expected"]
            got = info["n_cropzone"]
            pct = f"{100.0 * got / exp:.1f}%" if exp else "?"
            why = ("killed mid-frame" if info["ragged"]
                   else "killed between frames")
            print(f"  [mov{mn}] INCOMPLETE — build wrote {got}/{exp} frames "
                  f"({pct}); {why}: box/cropzone/frameInds = "
                  f"{info['n_box']}/{info['n_cropzone']}/{info['n_frameinds']}; "
                  f"skipping")
            incomplete_lines.append(
                f"mov{mn}  {got}/{exp} frames ({pct})  "
                f"box/cropzone/frameInds={info['n_box']}/"
                f"{info['n_cropzone']}/{info['n_frameinds']}")
            continue
        if status == "BAD_DATA":
            n_bad += 1
            inter = info["n_intersection"]
            total = info["n_total"]
            n_c = info.get("num_cams", "all")
            print(f"  [mov{mn}] BAD_DATA — only {inter}/{total} frames have "
                  f"all {n_c} cams tracking simultaneously "
                  f"(need {min_intersection}); skipping")
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
          f"{n_incomplete} INCOMPLETE, {n_bad} BAD_DATA, {n_err} ERR, "
          f"{n_missing} missing (of {len(movies)} total).")
    if failed_lines:
        print("Failed details:")
        for line in failed_lines:
            print(f"  {line}")
    if incomplete_lines:
        print("INCOMPLETE details (the build died partway — rebuild these movies):")
        for line in incomplete_lines:
            print(f"  {line}")
    if bad_lines:
        print("BAD_DATA details (recording problem — fly not tracked by every cam):")
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
                   help=f"cam-name substring of the mirror cam to flip "
                        f"(e.g. 'cam1' or 'cam5'), or '{AUTO_CAM}' to let the "
                        f"mirror check work it out from the calibration. "
                        f"Whatever you pass is verified against the "
                        f"calibration before anything is flipped.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap each movie's h5 at the first N frames (default: all)")
    p.add_argument("--num-cams", type=int, default=None,
                   help="override the camera count (normally detected from "
                        "the number of *_sparse.mat per movie dir; the old "
                        "lab rig had 3, the current one has 4)")
    p.add_argument("--skip-clean", action="store_true")
    p.add_argument("--skip-prescan", action="store_true",
                   help="skip the pre-build sparse-mat scan that filters out "
                        "movies where the fly is rarely tracked by all 4 cams")
    p.add_argument("--prescan-only", action="store_true",
                   help="run only the prescan; do not flip / build / verify")
    p.add_argument("--prescan-min-intersection", type=int,
                   default=DEFAULT_MIN_INTERSECTION,
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
    p.add_argument("--prescan-min-edge-margin", type=float,
                   default=DEFAULT_MIN_EDGE_MARGIN,
                   help="prescan: the fly's blob must stay this many px clear "
                        "of every image border. A frame where it doesn't is "
                        "one where the fly is partly outside the field of "
                        "view, so the build range is cut before it (default: "
                        f"{DEFAULT_MIN_EDGE_MARGIN}, 0 disables)")
    p.add_argument("--prescan-min-cams-in-frame", type=int,
                   default=DEFAULT_MIN_CAMS_IN_FRAME,
                   help="prescan: how many cams must see the WHOLE fly for a "
                        "frame to count; the rest may see it cut. Evaluated "
                        "per frame, so the majority need not be the same cams "
                        "throughout. Which cams were whole is recorded in each "
                        f"movie's {CAM_VALIDITY_SIDECAR} and used by predict "
                        "to drop the affected camera pairs (default: "
                        f"{DEFAULT_MIN_CAMS_IN_FRAME}; 0 = every cam; values "
                        f"below {MIN_USABLE_CAMS_IN_FRAME} are clamped)")
    p.add_argument("--skip-flip", action="store_true")
    p.add_argument("--no-mirror-check", action="store_true",
                   help="skip the pre-flip check that verifies --cam against "
                        "the calibration. The check is what stops a wrong or "
                        "repeated --cam from silently corrupting the mats, so "
                        "only use this when you know better than it does.")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the verification step (it runs by default)")
    p.add_argument("--verify-only", action="store_true",
                   help="skip clean / prescan / flip / build; just verify "
                        "existing files")
    p.add_argument("--verify-min-intersection", type=int,
                   default=DEFAULT_MIN_INTERSECTION,
                   help="a built movie with fewer than this many all-cams "
                        f"tracked frames is BAD_DATA (default: "
                        f"{DEFAULT_MIN_INTERSECTION})")
    p.add_argument("--verify-threshold", type=float, default=15.0,
                   help="per-cam LOO median above which a movie is marked FAIL "
                        "(default: 15.0)")
    p.add_argument("--perturbation", action="store_true",
                   help="declare this a perturbation experiment: write "
                        f"{PERTURBATION_FILE} beside calibration.h5. Predict "
                        "reads it and stamps every analysis h5 / CSV / mp4 "
                        "with the perturbation window.")
    p.add_argument("--perturbation-type", default="unspecified",
                   help="what the perturbation was, e.g. 'roll' or 'yaw'")
    p.add_argument("--perturbation-onset-frame", type=int, default=0,
                   help="trigger-relative frame at which the perturbation "
                        "starts (default: 0, i.e. it fires on the trigger)")
    p.add_argument("--perturbation-duration-ms", type=float, default=None,
                   help="how long the perturbation lasted, in ms. OMIT when "
                        "the log never recorded it: frames from the onset on "
                        "are then labelled 'unknown' rather than guessed.")
    p.add_argument("--perturbation-force", action="store_true",
                   help=f"overwrite an existing {PERTURBATION_FILE}. Without "
                        "this an existing file is validated and kept, so a "
                        "hand-authored declaration (per-movie windows, "
                        "provenance) is never clobbered by the CLI's simpler "
                        "one.")
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

        mode, movies = detect_mode(input_dir, args.dry_run)
        # Single-movie mode may have renamed the input dir to canonical 'mov<N>';
        # follow it so clean / verify / report all target the right path.
        if mode == "single":
            input_dir = movies[0][0]
        print(f"Mode: {mode}; movies queued: "
              f"{', '.join(f'mov{n}' for _, n in movies)}")
        num_cams = detect_num_cams(movies, args.num_cams)
        print(f"cameras: {num_cams}")

        timings_path = _timings_path(input_dir)
        print(f"timings ledger: {timings_path}")

        if args.verify_only:
            run_verify(input_dir, mode, movies, args.verify_threshold,
                       args.dry_run, timings_path=timings_path,
                       min_intersection=args.verify_min_intersection,
                       num_cams=num_cams)
        else:
            if not args.skip_clean:
                run_clean(input_dir, mode, movies, args.dry_run)

            # Prescan runs before FLIP so we don't waste time flipping cams
            # of movies we won't process. It also computes the per-movie build
            # range so BUILD emits only the contiguous all-cams-visible window.
            movie_ranges = {}
            scan_results = {}
            # Movies whose MATLAB build exited non-zero. Stays empty under
            # --skip-build (re-verifying an already-built experiment), where
            # the completeness check in verify is the only guard available.
            build_failed = []
            if not args.skip_prescan:
                movies, movie_ranges, scan_results = run_prescan(
                    input_dir, movies,
                    args.prescan_min_intersection,
                    args.prescan_pixel_threshold,
                    args.prescan_blob_ratio,
                    args.prescan_blob_distance,
                    args.dry_run,
                    min_edge_margin=args.prescan_min_edge_margin,
                    min_cams_in_frame=args.prescan_min_cams_in_frame,
                )
                if not movies:
                    print("\nAll movies flagged BAD by prescan; nothing to do.")
                    _write_report(input_dir, report_buf, args.dry_run)
                    return

            if args.prescan_only:
                _write_report(input_dir, report_buf, args.dry_run)
                return

            # Verify the flip decision against the calibration before acting on
            # it. Runs for --skip-flip too: "this experiment needs no flip" and
            # "someone forgot --cam" look identical from the command line, and
            # only the data can tell them apart.
            cam_to_flip = args.cam
            if not args.no_mirror_check:
                cam_to_flip = run_mirror_check(
                    movies, args.easywand,
                    AUTO_CAM if args.cam == AUTO_CAM else
                    (None if args.skip_flip else args.cam),
                    args.dry_run)
            elif args.cam == AUTO_CAM:
                sys.exit("--cam auto needs the mirror check; drop "
                         "--no-mirror-check or name a camera explicitly")

            if args.skip_flip:
                print("\n===== FLIP =====\n  skipped (--skip-flip)")
            elif not cam_to_flip:
                print("\n===== FLIP =====\n  skipped (no camera needs flipping)")
            else:
                run_flip(movies, cam_to_flip, args.dry_run,
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
                build_failed = run_build(input_dir, mode, movies,
                                         os.path.abspath(args.easywand),
                                         args.max_frames, args.dry_run,
                                         movie_ranges=movie_ranges,
                                         timings_path=timings_path,
                                         num_cams=num_cams)

            # Record which cams saw the whole fly per BUILT frame. Runs after
            # BUILD because the slice depends on the range the build actually
            # committed, and before VERIFY so a movie dropped by verify still
            # has its sidecar (it may be re-verified later with a different
            # threshold).
            run_cam_validity_sidecars(
                movies, scan_results, args.dry_run,
                params={
                    "min_edge_margin": args.prescan_min_edge_margin,
                    "pixel_threshold": args.prescan_pixel_threshold,
                    "blob_ratio": args.prescan_blob_ratio,
                    "blob_distance": args.prescan_blob_distance,
                })

            # Verification is on by default; --no-verify skips it. When
            # verify ran and produced a PASS list, restrict the manifest to it
            # so failed-calibration movies don't get queued for prediction.
            if not args.no_verify:
                verified = run_verify(input_dir, mode, movies,
                                      args.verify_threshold, args.dry_run,
                                      timings_path=timings_path,
                                      min_intersection=args.verify_min_intersection,
                                      build_failed=build_failed,
                                      num_cams=num_cams)
                if verified is not None:
                    movies = verified

            # Declare the perturbation window (after BUILD, so the coverage
            # report below can measure it against the range the build actually
            # committed). Predict picks the file up on its own from here.
            if args.perturbation:
                if run_perturbation_declaration(input_dir, args, args.dry_run):
                    report_perturbation_coverage(movies, args.dry_run)

            # Write a manifest of good (post-prescan, post-build, post-verify)
            # movies for `predict_array.sh`. This runs even with --skip-build
            # (e.g. re-verifying an already-built experiment): the manifest
            # writer independently checks find_movie_h5 per movie, so it only
            # lists movies that actually have a built h5.
            if mode == "multi":
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
