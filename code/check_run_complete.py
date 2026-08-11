"""Reconcile a prediction run against the manifest it was submitted from.

A SLURM array task can die before it runs a single line: if the node cannot
reach the shared filesystem at startup, the task is killed instantly, SLURM
never creates its .out/.err, and the only trace is `sacct` showing FAILED with
a 1-second elapsed time. Nothing downstream notices -- the run directory simply
holds fewer movies than it should, and the missing ones look exactly like
movies that were never submitted.

(That is not hypothetical: four tasks were lost that way across the 60ms and
100ms edge-cut runs, in two bursts, each burst on one node with the next task
on the same node succeeding a minute later.)

So this compares what was asked for against what exists, and while it is
walking the outputs anyway it also checks the two invariants the consuming
project asserts on `dir`:

  * every analysis h5 in the run reports the same `os.path.dirname(dir)`
    (one experiment per folder, and the name must look dated)
  * no two report the same `dir` (nothing claims another's provenance)

Catching those here is the point: they are cheap locally and expensive once the
files have been handed over.

Exits non-zero when anything is missing or an invariant fails, so a chained
SLURM job turns into a FAIL notification rather than a log nobody reads.

usage:
    .env/bin/python code/check_run_complete.py <manifest> <run_name>
        [--output-dir predict_output/debug_outputs]
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402

from process_experiment import find_movie_h5  # noqa: E402

DEFAULT_OUTPUT_DIR = "predict_output/debug_outputs"
SUFFIX = "_analysis_smoothed.h5"
# The consumer requires the folder holding one experiment's files to carry a
# date, so a generic run name (`predict_array`) is a defect, not a style issue.
DATED_EXPERIMENT_RE = re.compile(r"\d{4}_\d{2}_\d{2}")


def read_dir_field(path):
    with h5py.File(path, "r") as f:
        if "dir" not in f:
            return None
        v = f["dir"][()]
    return v.decode() if isinstance(v, bytes) else str(v)


def find_output(run_dir, stem):
    """The analysis h5 for one movie, wherever it now sits under the run.

    Searched recursively rather than at `<run_dir>/<stem>/` because movies get
    hand-sorted into `bad_signal/...` after the fact. A sorted movie has still
    been predicted; reporting it MISSING would be wrong.
    """
    hits = glob.glob(os.path.join(run_dir, "**", stem, "*" + SUFFIX),
                     recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="the manifest the array was submitted from")
    ap.add_argument("run_name", help="sbatch -J name == run directory basename")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    run_dir = os.path.join(args.output_dir, args.run_name)
    with open(args.manifest) as f:
        movie_dirs = [l.strip().rstrip("/") for l in f
                      if l.strip() and not l.startswith("#")]

    print(f"run      : {run_dir}")
    print(f"manifest : {args.manifest} ({len(movie_dirs)} movie(s))\n")
    if not os.path.isdir(run_dir):
        print(f"FAIL: run directory does not exist: {run_dir}")
        return 1

    missing, ok = [], []
    for movie_dir in movie_dirs:
        h5 = find_movie_h5(movie_dir)
        if h5 is None:
            # No dataset means the build, not the prediction, is what failed.
            missing.append((movie_dir, "no built h5 (never predictable)"))
            print(f"  MISSING  {movie_dir}  -- no built h5")
            continue
        stem = os.path.basename(h5)[:-3]
        out = find_output(run_dir, stem)
        if out is None:
            missing.append((movie_dir, f"no {stem}/*{SUFFIX} under the run"))
            print(f"  MISSING  {stem}")
        else:
            ok.append(out)
            print(f"  ok       {stem}")

    print(f"\n{len(ok)}/{len(movie_dirs)} movie(s) produced an analysis h5")

    # Invariants are checked over the WHOLE run, not just this manifest: a
    # second array submitted into the same run name is exactly how a folder
    # ends up mixing experiments, and this manifest would not show it.
    all_h5 = sorted(glob.glob(os.path.join(run_dir, "**", "*" + SUFFIX),
                              recursive=True))
    parents, seen, dup = {}, {}, []
    for p in all_h5:
        d = read_dir_field(p)
        if d is None:
            continue
        parents.setdefault(os.path.dirname(d), []).append(p)
        if d in seen:
            dup.append((d, seen[d], p))
        seen[d] = p

    print(f"\ndir invariants over {len(all_h5)} file(s) in the run:")
    problems = []
    if len(parents) == 1:
        parent = next(iter(parents))
        dated = bool(DATED_EXPERIMENT_RE.search(os.path.basename(parent)))
        print(f"  1 distinct dirname(dir): {parent}"
              + ("" if dated else "   <== NOT dated"))
        if not dated:
            problems.append(f"dirname(dir) has no date: {parent}")
    elif len(parents) == 0:
        print("  (no dir fields found)")
    else:
        print(f"  {len(parents)} distinct dirname(dir) -- the consumer requires 1:")
        for k, v in parents.items():
            print(f"     {k}   ({len(v)} file(s))")
        problems.append(f"{len(parents)} distinct dirname(dir)")
    if dup:
        print(f"  {len(dup)} duplicated dir value(s):")
        for d, a, b in dup[:5]:
            print(f"     {d}\n       {a}\n       {b}")
        problems.append(f"{len(dup)} duplicate dir value(s)")
    else:
        print(f"  {len(seen)} distinct dir value(s), no duplicates")

    if missing or problems:
        print(f"\nFAIL: {len(missing)} missing, {len(problems)} invariant "
              f"problem(s)")
        for movie_dir, why in missing:
            print(f"  missing: {movie_dir}  ({why})")
        for pr in problems:
            print(f"  invariant: {pr}")
        print("\nRe-queue the missing movies with submit_edge_cut_predictions.py; "
              "they land in the same run directory and keep the same dir prefix.")
        return 1
    print("\nOK: every movie in the manifest produced an analysis h5, and the "
          "run satisfies the consumer's dir invariants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
