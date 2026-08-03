"""
ONE-OFF MIGRATION: add gravity_body to analysis files written before the field
existed, so old predictions don't have to be re-run. Delete this script once
every movie you care about has been backfilled.

For each analysis h5 it:
  1. computes gravity_body from the stored x_body / y_body / z_body using
     FlightAnalysis.gravity_in_body_axes - the same function the pipeline now
     calls, so a backfilled movie is identical to a freshly analysed one;
  2. checks the result (unit vectors, and they must rotate back to lab
     (0, 0, -1) - the reported residual should be ~1e-16) and refuses to write
     if the check fails;
  3. adds the dataset to the h5 in place and reads it back to confirm the
     write. Nothing else in the file is touched, and a file that already has
     the field is skipped, so re-running is harmless;
  4. adds gravity_body_x/y/z to the sibling *_analysis_smoothed.csv, if one
     sits next to the h5;
  5. only with --plot: writes <h5 name>_gravity_check.html beside it
     (see plot_gravity_body.py, which also runs standalone on any h5).

Everything comes out of the h5 itself - the body axes were baked in at analysis
time - so a folder of files gathered by collect_analysis_h5.py works just as
well as a full predict_output tree. The npy files, the source movie, the
calibration and the run layout are all irrelevant here.

Usage:
    # look first - touches nothing
    .env/bin/python code/backfill_gravity_body.py <collected_h5_dir> --dry-run

    # one file, with the verification plot, to eyeball before committing
    .env/bin/python code/backfill_gravity_body.py \\
        <collected_h5_dir>/<mov>_analysis_smoothed.h5 --plot

    # then the whole folder (no plots)
    .env/bin/python code/backfill_gravity_body.py <collected_h5_dir>

The target may be a single h5 or any directory - directories are searched
recursively, so predict_output/ works too.
"""
import argparse
import os
import sys

import h5py
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "prediction_code_lior")]
from extract_flight_data import FlightAnalysis          # noqa: E402
from plot_gravity_body import plot_one, SUFFIX, GRAVITY_LAB   # noqa: E402

CSV_COLUMNS = ["gravity_body_x", "gravity_body_y", "gravity_body_z"]
REQUIRED = ("x_body", "y_body", "z_body")


def find_h5(target):
    """Every analysis h5 at or under target. Matches the collision-renamed
    copies collect_analysis_h5.py writes (<stem>__<parent>.h5) as well as the
    plain *_analysis_smoothed.h5, so a collected folder is covered too."""
    if os.path.isfile(target):
        return [target]
    matches = []
    for root, _, files in os.walk(target):
        matches += [os.path.join(root, f) for f in files
                    if f.endswith(".h5") and "analysis_smoothed" in f]
    return sorted(matches)


def update_csv(h5_path, gravity_body, dry_run):
    """Mirror the new columns into the MATLAB CSV the pipeline writes next to
    the h5. Skipped (loudly) if the CSV's row count doesn't line up."""
    csv_path = h5_path[:-3] + ".csv"
    if not os.path.isfile(csv_path):
        return "no csv"
    import pandas as pd
    df = pd.read_csv(csv_path)
    if len(df) != len(gravity_body):
        return f"csv SKIPPED ({len(df)} rows vs {len(gravity_body)} frames)"
    if all(c in df.columns for c in CSV_COLUMNS):
        return "csv already had it"
    if dry_run:
        return "csv would be updated"
    for i, col in enumerate(CSV_COLUMNS):
        df[col] = gravity_body[:, i]
    df.to_csv(csv_path, index=False)
    return "csv updated"


def backfill_one(h5_path, dry_run=False, overwrite=False):
    """Returns (status string, gravity_body or None, whether the h5 changed)."""
    with h5py.File(h5_path, "r") as h5:
        if "gravity_body" in h5 and not overwrite:
            # Nothing to add, but hand the array back so the csv and the
            # verification plot are still brought up to date.
            return "already had gravity_body", h5["gravity_body"][:], False
        missing = [k for k in REQUIRED if k not in h5]
        if missing:
            return f"SKIPPED - missing {', '.join(missing)}", None, False
        x_body, y_body, z_body = (h5[k][:] for k in REQUIRED)

    gravity_body = FlightAnalysis.gravity_in_body_axes(x_body, y_body, z_body)

    # Sanity-check before writing: on the frames that survive, the vector must
    # be a unit vector that rotates back to lab (0, 0, -1).
    valid = ~np.isnan(gravity_body).any(axis=1)
    if not valid.any():
        return "SKIPPED - no frames with a defined body frame", None, False
    norms = np.linalg.norm(gravity_body[valid], axis=1)
    axes = np.stack((x_body[valid], y_body[valid], z_body[valid]), axis=1)
    residual = float(np.abs(np.einsum("fij,fi->fj", axes, gravity_body[valid])
                            - GRAVITY_LAB).max())
    if residual > 1e-9 or np.abs(norms - 1).max() > 1e-9:
        return (f"SKIPPED - failed the check (residual {residual:.2e}, "
                f"max |norm-1| {np.abs(norms - 1).max():.2e})"), None, False

    note = f"{valid.sum()}/{len(gravity_body)} frames, residual {residual:.1e}"
    if dry_run:
        return f"would add gravity_body - {note}", gravity_body, True

    with h5py.File(h5_path, "r+") as h5:
        if "gravity_body" in h5:
            del h5["gravity_body"]
        h5.create_dataset("gravity_body", data=gravity_body)

    # Read back what actually landed on disk. A collected folder is often the
    # only copy left once predict_output has been archived, so don't report
    # success on a write we haven't confirmed.
    with h5py.File(h5_path, "r") as h5:
        written = h5["gravity_body"][:]
    if written.shape != gravity_body.shape or written.tobytes() != gravity_body.tobytes():
        return "FAILED - dataset did not read back identically", None, False
    return f"added gravity_body - {note}", gravity_body, True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help=f"an *{SUFFIX} file, or a directory to search "
                                   f"recursively (e.g. predict_output)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing anything")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even where gravity_body already exists")
    ap.add_argument("--plot", action="store_true",
                    help="also write a <h5 name>_gravity_check.html verification "
                         "plot next to each file (off by default)")
    ap.add_argument("-k", "--every", type=int, default=100, metavar="K",
                    help="with --plot: draw the body frame every K frames "
                         "(default: 100)")
    args = ap.parse_args()

    if not os.path.exists(args.target):
        sys.exit(f"No such file or directory: {args.target}")

    files = find_h5(args.target)
    if not files:
        sys.exit(f"No *{SUFFIX} files found under {args.target}")
    print(f"Found {len(files)} analysis file(s)"
          f"{' (dry run - nothing will be written)' if args.dry_run else ''}\n")

    n_added = n_skipped = n_failed = 0
    for h5_path in files:
        rel = os.path.relpath(h5_path, args.target if os.path.isdir(args.target)
                              else os.path.dirname(h5_path))
        try:
            status, gravity_body, changed = backfill_one(
                h5_path, args.dry_run, args.overwrite)
        except Exception as e:
            print(f"  {rel}\n      FAILED: {e}", file=sys.stderr)
            n_failed += 1
            continue

        print(f"  {rel}\n      {status}")
        if gravity_body is None:
            n_skipped += 1
            continue
        n_added += changed
        n_skipped += not changed
        print(f"      {update_csv(h5_path, gravity_body, args.dry_run)}")

        if args.plot and not args.dry_run:
            try:
                plot_one(h5_path, args.every, scale=1.0)
            except Exception as e:
                print(f"      plot failed: {e}", file=sys.stderr)

    verb = "would be updated" if args.dry_run else "updated"
    print(f"\n{n_added} {verb}, {n_skipped} skipped, {n_failed} failed")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
