"""Prove that a movie blanked on a PC (code/sparse_trim.py) preps exactly like the original.

For each case (a movie folder and its easyWand) this runs prep twice in a scratch folder:

    A  the movie's mats copied as they are
    B  the same mats blanked: only the build's read range and the mirror check's sample frames kept

with identical arguments, and then requires:

  - MATLAB loads every blanked mat; metaData is isequal to the original's, every kept frame is
    isequal, every other frame is empty (what the raw movie and the builder see);
  - the prescan picks the same frames and the same build range;
  - the mirror check reaches the same verdict from the same numbers, to the bit;
  - the built box h5 has the same name and identical datasets (box, cropzone, frameInds, ...);
  - prescan_cam_validity.npz and calibration.h5 are identical, verify gives the same medians;
  - B's raw movie was made (JoinSparses reads every frame, blanked ones included).

Nothing uploaded from a PC may be predicted until this passes. Run it as a CPU job -- it runs
MATLAB, so not beside another prep (they stall each other):

    POSE_PROJECT=$PWD sbatch --gres=gpu:0 -p glacier --mem=16g -c4 --time=4:00:00 \\
        -J check_sparse_trim sbatch_files/sbatch_configurable.sh code/check_sparse_trim.py \\
        --case <movie_dir> <easywand.mat> [--case ...] [-- <extra prep args, e.g. --skip-flip>]
"""
import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_sparse_movies import PRESCAN_DEFAULTS, scan_movie  # noqa: E402
from sparse_trim import TRIM_FILE, blank_movie, movie_mats  # noqa: E402
from process_experiment import find_movie_h5, MATLAB_BIN  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATLAB_CHECK = r"""
function check_blanked_mats(list_file)
% Each line of list_file: <original mat>|<blanked mat>|<file of kept 1-based frame indices>
lines = strsplit(strtrim(fileread(list_file)), newline);
for k = 1:numel(lines)
    parts = strsplit(strtrim(lines{k}), '|');
    a = load(parts{1}, 'frames', 'metaData');
    b = load(parts{2}, 'frames', 'metaData');
    keep = false(numel(a.frames), 1);
    keep(load(parts{3})) = true;
    assert(numel(a.frames) == numel(b.frames), 'frame count differs: %s', parts{2});
    assert(isequal(a.metaData, b.metaData), 'metaData differs: %s', parts{2});
    for i = 1:numel(a.frames)
        if keep(i)
            assert(isequal(a.frames(i).indIm, b.frames(i).indIm), 'kept frame %d differs: %s', i, parts{2});
        else
            assert(isempty(b.frames(i).indIm), 'frame %d not empty: %s', i, parts{2});
            assert(isa(b.frames(i).indIm, class(a.frames(i).indIm)), 'frame %d class: %s', i, parts{2});
        end
    end
    fprintf('MATLAB OK %s (%d frames, %d kept)\n', parts{2}, numel(a.frames), nnz(keep));
end
end
"""


def run(cmd, log):
    print("$", " ".join(cmd), flush=True)
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT)
    print(f"  -> exit {rc} (log {log})", flush=True)
    return rc


def h5_dataset_names(f):
    names = []
    f.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
    return sorted(names)


def same_h5(a, b, block=256):
    """None when every dataset of the two h5 files is identical, else the first difference.
    Compared a block of frames at a time: a 4-camera box is several GB once decompressed."""
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        na, nb = h5_dataset_names(fa), h5_dataset_names(fb)
        if na != nb:
            return f"datasets differ: {sorted(set(na) ^ set(nb))}"
        for name in na:
            x, y = fa[name], fb[name]
            if x.shape != y.shape or x.dtype != y.dtype:
                return f"{name} shape/dtype differs"
            if x.ndim == 0 or x.shape[0] <= block:
                if not np.array_equal(x[()], y[()]):
                    return f"{name} differs"
                continue
            for start in range(0, x.shape[0], block):
                if not np.array_equal(x[start:start + block], y[start:start + block]):
                    return f"{name} differs from frame {start}"
    return None


def same_npz(a, b):
    with np.load(a) as x, np.load(b) as y:
        if sorted(x.files) != sorted(y.files):
            return "keys differ"
        for k in x.files:
            if not np.array_equal(x[k], y[k]):
                return f"{k} differs"
    return None


PRESCAN_OPTIONS = {"--prescan-min-intersection": ("min_intersection", int),
                   "--prescan-pixel-threshold": ("pixel_threshold", int),
                   "--prescan-blob-ratio": ("blob_ratio", float),
                   "--prescan-blob-distance": ("blob_distance", float),
                   "--prescan-min-edge-margin": ("min_edge_margin", float),
                   "--prescan-min-cams-in-frame": ("min_cams_in_frame", int)}


def prescan_params(extra):
    """The prescan's thresholds as prep will apply them: the defaults, with any --prescan-* given
    to prep -- the PC blanking a movie must scan it exactly as prep will."""
    params = dict(PRESCAN_DEFAULTS)
    for i, arg in enumerate(extra):
        if arg in PRESCAN_OPTIONS and i + 1 < len(extra):
            key, kind = PRESCAN_OPTIONS[arg]
            params[key] = kind(extra[i + 1])
    return params


def check_case(movie_dir, easywand, work, extra, failures, keep_lists):
    movie_dir = os.path.abspath(movie_dir)
    movie = os.path.basename(movie_dir)
    experiment = os.path.basename(os.path.dirname(movie_dir))
    print(f"\n===== {experiment}/{movie} =====", flush=True)
    sides = {}
    for side in ("A", "B"):
        exp_dir = os.path.join(work, side, experiment)
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy2(easywand, exp_dir)
        sides[side] = exp_dir
    a_movie = os.path.join(sides["A"], movie)
    os.makedirs(a_movie, exist_ok=True)
    for mat in movie_mats(movie_dir):
        shutil.copy2(mat, a_movie)
    p = prescan_params(extra)
    scan = scan_movie(movie_dir, p["pixel_threshold"], p["blob_ratio"], p["blob_distance"],
                      p["min_edge_margin"], p["min_cams_in_frame"])
    b_movie = os.path.join(sides["B"], movie)
    trim = blank_movie(movie_dir, b_movie, scan, p)
    for name, m in trim["mats"].items():
        keep_file = os.path.join(work, f"{experiment}_{movie}_{name}.keep.txt")
        with open(keep_file, "w") as f:
            for a, b in m["kept"]:
                f.write("\n".join(str(i + 1) for i in range(a, b + 1)) + "\n")
        keep_lists.append(f"{os.path.join(movie_dir, name)}|{os.path.join(b_movie, name)}|"
                          f"{keep_file}")
        print(f"  {name}: kept {m['kept_frames']}/{m['n_frames']}, "
              f"{m['bytes_original'] / 1e6:.1f} -> {m['bytes'] / 1e6:.1f} MB")
    status = {}
    for side in ("A", "B"):
        exp_dir = sides[side]
        cmd = [sys.executable, "-u", "code/process_experiment.py", exp_dir,
               "--easywand", os.path.join(exp_dir, os.path.basename(easywand)),
               "--manifest", os.path.join(exp_dir, "manifest.txt"),
               "--status-json", os.path.join(exp_dir, "status.json")] + list(extra)
        if side == "A":
            cmd.append("--skip-raw-movies")   # only B's raw movie is in question
        rc = run(cmd, os.path.join(work, f"prep_{side}_{experiment}.log"))
        if rc != 0:
            failures.append(f"{experiment}/{movie}: prep {side} exited {rc}")
        with open(os.path.join(exp_dir, "status.json")) as f:
            status[side] = json.load(f)
    sa, sb = status["A"], status["B"]
    ma, mb = sa["movies"].get(movie, {}), sb["movies"].get(movie, {})
    for key in ("prescan", "good_start", "good_end", "start_ind", "end_ind", "verify",
                "medians", "box", "in_manifest"):
        if ma.get(key) != mb.get(key):
            failures.append(f"{experiment}/{movie}: {key} {ma.get(key)!r} vs {mb.get(key)!r}")
    if sa.get("mirror") != sb.get("mirror"):
        failures.append(f"{experiment}/{movie}: mirror check differs: "
                        f"{sa.get('mirror')} vs {sb.get('mirror')}")
    if (ma.get("start_ind"), ma.get("end_ind")) != (trim["start_ind"], trim["end_ind"]):
        failures.append(f"{experiment}/{movie}: prep built {ma.get('start_ind')}-"
                        f"{ma.get('end_ind')}, the PC predicted {trim['start_ind']}-"
                        f"{trim['end_ind']}")
    box_a, box_b = find_movie_h5(a_movie), find_movie_h5(b_movie)
    if not (box_a and box_b):
        failures.append(f"{experiment}/{movie}: no box built ({box_a}, {box_b})")
    else:
        if os.path.basename(box_a) != os.path.basename(box_b):
            failures.append(f"{experiment}/{movie}: box names {os.path.basename(box_a)} vs "
                            f"{os.path.basename(box_b)}")
        problem = same_h5(box_a, box_b)
        if problem:
            failures.append(f"{experiment}/{movie}: box {problem}")
        else:
            print(f"  box identical: {os.path.basename(box_b)}")
    for name, compare in (("prescan_cam_validity.npz", same_npz),):
        a, b = os.path.join(a_movie, name), os.path.join(b_movie, name)
        problem = compare(a, b) if os.path.isfile(a) and os.path.isfile(b) else "missing"
        if problem:
            failures.append(f"{experiment}/{movie}: {name} {problem}")
    problem = same_h5(os.path.join(sides["A"], "calibration.h5"),
                      os.path.join(sides["B"], "calibration.h5"))
    if problem:
        failures.append(f"{experiment}: calibration.h5 {problem}")
    if not glob.glob(os.path.join(b_movie, "*_raw_fr*_skip*.mp4")):
        failures.append(f"{experiment}/{movie}: no raw movie from the blanked mats")
    if not os.path.isfile(os.path.join(b_movie, TRIM_FILE)):
        failures.append(f"{experiment}/{movie}: {TRIM_FILE} missing")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs=2, action="append", required=True,
                    metavar=("MOVIE_DIR", "EASYWAND"))
    ap.add_argument("--work", default=None,
                    help="scratch folder on the shared filesystem (default: "
                         "realign_jobs/check_sparse_trim_<stamp>)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch copies")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="after --: arguments for both prep runs, e.g. --skip-flip")
    args = ap.parse_args()
    extra = [a for a in args.extra if a != "--"]
    work = args.work or os.path.join(
        PROJECT, "realign_jobs",
        f"check_sparse_trim_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(work, exist_ok=True)
    print(f"work folder {work}")
    failures, keep_lists = [], []
    for movie_dir, easywand in args.case:
        try:
            check_case(movie_dir, os.path.abspath(easywand), work, extra, failures, keep_lists)
        except Exception as e:
            failures.append(f"{movie_dir}: {type(e).__name__}: {e}")
    if keep_lists:
        with open(os.path.join(work, "check_blanked_mats.m"), "w") as f:
            f.write(MATLAB_CHECK)
        list_file = os.path.join(work, "mats.txt")
        with open(list_file, "w") as f:
            f.write("\n".join(keep_lists) + "\n")
        rc = run([MATLAB_BIN, "-batch",
                  f"cd('{work}'); check_blanked_mats('{list_file}')"],
                 os.path.join(work, "matlab_load.log"))
        with open(os.path.join(work, "matlab_load.log")) as f:
            print(f.read())
        if rc != 0:
            failures.append(f"MATLAB could not load the blanked mats as expected (exit {rc})")
    report = {"cases": args.case, "extra": extra, "failures": failures,
              "passed": not failures, "at": dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(work, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("\n===== RESULT =====")
    for line in failures:
        print("  FAIL", line)
    print("PASSED: blanked movies prep exactly like the originals" if not failures
          else f"FAILED ({len(failures)} problem(s))")
    if not args.keep and not failures:
        for side in ("A", "B"):
            shutil.rmtree(os.path.join(work, side), ignore_errors=True)
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
