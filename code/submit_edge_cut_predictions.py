"""Queue predictions for the edge-cut rebuilds, one run name per EXPERIMENT.

The run name matters beyond bookkeeping. FlightAnalysis records
`self.dir = os.path.dirname(points_3D_path)` and writes it into every analysis
h5 as the `dir` dataset, so

    dir = predict_output/<output_dir>/<RUN NAME>/<movie h5 basename>

and `<RUN NAME>` is whatever `-J` the array was submitted with. The consuming
project asserts that all files it groups into one folder share
`os.path.dirname(dir)` -- i.e. one source experiment per folder -- and that the
name looks like a dated experiment. So the run name has to be the experiment,
identical for every movie of it, and it has to match what the experiment's
earlier movies already recorded.

Concretely: `roni_dark/2023_08_07_5ms/1to30/mov5` and
`roni_dark/2023_08_07_5ms/61to90/mov73` must BOTH be predicted under
`roni_dark_2023_08_07_5ms`, not under per-batch names -- otherwise one
experiment ends up split across several `dir` prefixes and the check fires.
(`submit_rebuilt_predictions.py` names per batch dir; it predates this
requirement, so do not reuse it here.)

Each movie is checked -- build complete, calibration present, reprojection
verify -- before submission, so nothing is queued that would only fail after
waiting for a GPU.

usage:
    .env/bin/python code/submit_edge_cut_predictions.py <manifest> [--dry-run]
        [--config predict_configurations/config1.json] [--concurrency 8]
"""

import argparse
import datetime
import os
import subprocess
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402

from process_experiment import (check_build_complete, find_calibration_h5,  # noqa: E402
                                find_movie_h5, verify_one_movie)

LAUNCHER = "sbatch_files/predict_array.sh"
JOB_WRAPPER = "sbatch_files/submit_predictions.sh"
CHECKER = "code/check_run_complete.py"
MANIFEST_DIR = "manifests"


def experiment_of(movie_dir):
    """'inference_datasets/roni_dark/2023_08_07_5ms/1to30/mov5'
       -> ('roni_dark/2023_08_07_5ms', 'roni_dark_2023_08_07_5ms')

    The batch subdir (1to30, 61to90, ...) is deliberately dropped: it is a
    convenience grouping of one experiment's movies, not an experiment."""
    parts = os.path.abspath(movie_dir).split(os.sep)
    if "inference_datasets" not in parts:
        raise ValueError(f"not under inference_datasets: {movie_dir}")
    i = len(parts) - 1 - parts[::-1].index("inference_datasets")
    rel = parts[i + 1:]                 # [roni_dark, 2023_08_07_5ms, 1to30, mov5]
    if len(rel) < 3:
        raise ValueError(f"cannot infer experiment from {movie_dir}")
    experiment = "/".join(rel[:-2])
    return experiment, experiment.replace("/", "_")


def check_movie(movie_dir):
    """(ok, reason). Mirrors the pre-flight submit_rebuilt_predictions does."""
    h5 = find_movie_h5(movie_dir)
    if h5 is None:
        return False, "no dataset h5"
    with h5py.File(h5, "r") as f:
        partial = check_build_complete(f)
    if partial is not None:
        return False, (f"build INCOMPLETE {partial['n_cropzone']}/"
                       f"{partial['n_expected']}")
    batch_dir = os.path.dirname(movie_dir)
    mn_dir = os.path.basename(movie_dir)
    calib = find_calibration_h5(batch_dir, "multi", [(movie_dir, mn_dir)])
    if calib is None:
        return False, f"no calibration.h5 in {batch_dir}"
    status, medians, _ = verify_one_movie(h5, calib, 15.0)
    if status != "PASS":
        return False, f"verify {status} {medians}"
    return True, os.path.basename(h5)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--config", default="predict_configurations/config1.json")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="max concurrent array tasks per experiment")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.manifest) as f:
        movie_dirs = [l.strip().rstrip("/") for l in f
                      if l.strip() and not l.startswith("#")]

    by_run = OrderedDict()
    skipped = []
    for d in movie_dirs:
        try:
            _experiment, run = experiment_of(d)
        except ValueError as e:
            skipped.append((d, str(e)))
            continue
        ok, why = check_movie(d)
        if not ok:
            skipped.append((d, why))
            continue
        by_run.setdefault(run, []).append(d)
        print(f"  OK  {d:<58} -> run '{run}'  ({why})")

    print(f"\n{len(movie_dirs) - len(skipped)}/{len(movie_dirs)} movies pass "
          f"pre-flight, in {len(by_run)} experiment(s)")
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for d, why in skipped:
            print(f"  {d}: {why}")

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    # Timestamped, never reused. The per-run manifest is the only record of what
    # a given array was asked to produce, and re-submitting (a retry, a second
    # batch of movies) used to overwrite it -- after which a completeness check
    # against it would compare the new array's output to the new array's list
    # and always pass, which is precisely when the check is needed.
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for run, dirs in by_run.items():
        mpath = os.path.join(MANIFEST_DIR, f"edge_cut_{run}_{stamp}.txt")
        if not args.dry_run:
            with open(mpath, "w") as f:
                for d in dirs:
                    f.write(d + "\n")
        n = len(dirs)
        cap = min(args.concurrency, n)
        cmd = ["sbatch", "--parsable", f"--array=0-{n - 1}%{cap}", "-J", run,
               LAUNCHER, mpath, args.config]
        print(f"\n{run}: {n} movie(s) -> {mpath}")
        print(f"  {' '.join(cmd)}")
        if args.dry_run:
            print(f"  would chain: {CHECKER} {mpath} {run}")
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  sbatch FAILED: {r.stderr.strip()}")
            continue
        array_id = r.stdout.strip().split(";")[0]
        print(f"  submitted array {array_id}")

        # Chained on afterany, not afterok: an array where some tasks failed is
        # exactly the case worth reconciling, and afterok would skip it.
        check = ["sbatch", "--parsable", f"--dependency=afterany:{array_id}",
                 "-J", f"check_{run}", JOB_WRAPPER, CHECKER, mpath, run]
        rc = subprocess.run(check, capture_output=True, text=True)
        if rc.returncode != 0:
            print(f"  completeness check NOT chained: {rc.stderr.strip()}")
            print(f"  run it by hand: .env/bin/python {CHECKER} {mpath} {run}")
        else:
            print(f"  chained completeness check {rc.stdout.strip()} "
                  f"(emails on FAIL if anything is missing)")

    print("\noutput -> predict_output/debug_outputs/<run name>/<movie h5 "
          "basename>/ ; every analysis h5 in one experiment will report the "
          "same dirname(dir).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
