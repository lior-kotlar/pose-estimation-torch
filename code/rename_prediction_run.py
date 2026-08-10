"""Rename a prediction run directory and keep the analysis h5 `dir` field with it.

A run directory under `predict_output/<output_dir>/` is named after the sbatch
`-J` job name, and that name is baked into every analysis h5 the run produced:
FlightAnalysis records `self.dir = os.path.dirname(points_3D_path)`
(extract_flight_data.py) and it is written out as the `dir` dataset. Downstream
consumers treat `os.path.dirname(dir)` as the experiment identity -- it is how
they assert that one folder holds one experiment -- so a run left under a
generic job name like `predict_array` reads as an undated, unidentifiable
experiment no matter what the folder is called on disk.

Renaming the folder alone is therefore not enough: the `dir` inside each file
still names the old run. This does both, in the order that leaves the tree
consistent at every step.

Note `dir` records where a movie's results were WRITTEN, not where they now
live -- movies later sorted into `bad_signal/...` keep the original top-level
path. That is deliberate and is preserved here: only the run-name component is
rewritten, the rest of the recorded path is left exactly as it was.

Each file is patched via a temp copy and an atomic os.replace, so an
interrupted run can never leave a half-written analysis h5.

usage:
    .env/bin/python code/rename_prediction_run.py <old_run_dir> <new_name> [--dry-run]

e.g.
    .env/bin/python code/rename_prediction_run.py \\
        predict_output/debug_outputs/predict_array roni_dark_2023_08_07_10ms
"""

import argparse
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np

SUFFIX = "_analysis_smoothed.h5"


def read_dir_field(path):
    with h5py.File(path, "r") as f:
        if "dir" not in f:
            return None
        v = f["dir"][()]
    return v.decode() if isinstance(v, bytes) else str(v)


def patch_dir_field(path, new_value):
    """Rewrite the scalar `dir` dataset. It is a variable-length string, so it
    cannot be assigned in place at a different length -- the dataset has to be
    dropped and recreated. Done on a temp copy that replaces the original
    atomically, so the file is never observable in a partly-written state."""
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".h5.tmp")
    os.close(fd)
    try:
        shutil.copy2(path, tmp)
        with h5py.File(tmp, "r+") as f:
            if "dir" in f:
                del f["dir"]
            f.create_dataset("dir", data=np.bytes_(new_value))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="existing run directory to rename")
    ap.add_argument("new_name", help="new run directory basename")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip(os.sep)
    if not os.path.isdir(run_dir):
        sys.exit(f"not a directory: {run_dir}")
    parent = os.path.dirname(run_dir)
    old_name = os.path.basename(run_dir)
    new_dir = os.path.join(parent, args.new_name)
    if old_name == args.new_name:
        sys.exit("old and new names are the same")
    if os.path.exists(new_dir):
        sys.exit(f"target already exists: {new_dir}")

    # The substring rewritten inside each `dir` value. Anchored on the parent
    # so a run whose name happens to occur elsewhere in the path is untouched.
    old_frag = os.path.join(parent, old_name) + os.sep
    new_frag = os.path.join(parent, args.new_name) + os.sep

    files = sorted(
        os.path.join(r, f)
        for r, _, fs in os.walk(run_dir)
        for f in fs if f.endswith(SUFFIX)
    )
    print(f"{run_dir}  ->  {new_dir}")
    print(f"{len(files)} analysis h5 file(s); rewriting 'dir' prefix")
    print(f"   {old_frag}\n-> {new_frag}\n")

    plan, skipped, missing = [], [], []
    for p in files:
        cur = read_dir_field(p)
        if cur is None:
            missing.append(p)
            continue
        cur_n = cur.replace("\\", "/")
        if cur_n.startswith(old_frag.replace("\\", "/")):
            plan.append((p, cur, new_frag + cur_n[len(old_frag):]))
        else:
            skipped.append((p, cur))

    for p, cur, new in plan[:4]:
        print(f"   {os.path.basename(p)}\n      {cur}\n   -> {new}")
    if len(plan) > 4:
        print(f"   ... {len(plan) - 4} more")
    if skipped:
        print(f"\n   {len(skipped)} file(s) whose 'dir' does not start with the "
              f"old run path (left untouched):")
        for p, cur in skipped[:5]:
            print(f"      {os.path.basename(p)}: {cur}")
    if missing:
        print(f"\n   {len(missing)} file(s) with no 'dir' dataset")

    if args.dry_run:
        print(f"\n[dry run] would patch {len(plan)} file(s), then rename the "
              f"directory")
        return 0

    # Patch first, rename second: every path printed above is still valid while
    # the patching runs, so an interruption leaves a directory whose name and
    # contents disagree only in the direction that is trivially re-runnable.
    for i, (p, _, new) in enumerate(plan, 1):
        patch_dir_field(p, new)
        if i % 10 == 0 or i == len(plan):
            print(f"   patched {i}/{len(plan)}")
    os.rename(run_dir, new_dir)
    print(f"\nrenamed -> {new_dir}")

    bad = []
    for p, _, expected in plan:
        got = read_dir_field(p.replace(run_dir, new_dir, 1))
        if got != expected:
            bad.append((p, got, expected))
    if bad:
        print(f"VERIFY FAILED for {len(bad)} file(s)")
        for p, got, exp in bad[:5]:
            print(f"   {p}\n      got {got!r}\n      want {exp!r}")
        return 1
    print(f"verified: all {len(plan)} file(s) report the new run path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
