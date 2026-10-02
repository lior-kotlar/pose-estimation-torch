"""
check_body_length.py
====================

Flag movies whose reconstructed fly has the wrong length.

A fly's tail-to-head length does not change during a flight, and the flies of
one experiment differ by a few percent. A movie whose fly comes out much
shorter or longer than the others is a movie whose body axis was put in the
wrong place -- usually for the whole flight, and with nothing else to show for
it. The simulated 2-camera movies are where this was found: two views can pin
the head and tail poorly in depth, and on the Roni pilot two of twelve
(pair, movie) runs gave a fly 9-11 % short (2.26-2.27 mm against ~2.5 mm) whose
body was tilted 4-8 degrees for the entire movie. The check reads only the
analysis h5, so it works on any rig's predictions.

For each movie: the median tail-to-head distance over its analysed frames, how
far that is from a reference, and the share of frames more than the threshold
off. A movie is flagged when its median is more than --threshold-pct off. The
reference is, per movie:

    --reference-run R   the same movie's length in run R (e.g. the 4-camera
                        prediction of a simulated 2-camera movie: same fly);
    --reference-mm L    a length you know;
    otherwise           the median over the checked run's movies -- sound for
                        an experiment's worth of movies, shaky for a handful.

The default threshold, 6 %, is set from that pilot: the four 4-camera flies
spread +-2.3 % around their median, the two bad 2-camera runs were -9 and -11 %.

    .env/bin/python code/check_body_length.py predict_output/<run> [<run> ...]
        [--reference-run predict_output/<run>] [--threshold-pct 6] [--csv out.csv]

Exits 1 when any movie is flagged, so it can gate a script.
"""

import argparse
import csv
import glob
import os
import sys

import h5py
import numpy as np

DEFAULT_THRESHOLD_PCT = 6.0


def movie_lengths(run_dir):
    """{movie: per-frame tail-to-head length in mm} for every analysed movie
    directly under run_dir."""
    out = {}
    for h5 in sorted(glob.glob(os.path.join(run_dir, "*", "*_analysis_smoothed.h5"))):
        with h5py.File(h5, "r") as f:
            points = f["points_3D"][:]
            tail, head = (np.asarray(f["head_tail_inds"][:]).astype(int)
                          if "head_tail_inds" in f else (16, 17))
        out[os.path.basename(os.path.dirname(h5))] = \
            np.linalg.norm(points[:, head] - points[:, tail], axis=-1) * 1000.0
    return out


def check_run(run_dir, threshold_pct, reference_mm=None, reference=None):
    """Rows of (movie, median mm, reference mm, deviation %, % frames off,
    flagged) for one run."""
    lengths = movie_lengths(run_dir)
    if not lengths:
        return []
    run_median = float(np.median([np.nanmedian(v) for v in lengths.values()]))
    rows = []
    for movie, per_frame in lengths.items():
        if reference is not None:
            if movie not in reference:
                continue
            ref = float(np.nanmedian(reference[movie]))
        else:
            ref = reference_mm if reference_mm is not None else run_median
        median = float(np.nanmedian(per_frame))
        dev = 100.0 * (median - ref) / ref
        valid = per_frame[~np.isnan(per_frame)]
        off = (100.0 * float((np.abs(valid - ref) / ref > threshold_pct / 100).mean())
               if len(valid) else float("nan"))
        rows.append({"run": os.path.basename(os.path.normpath(run_dir)),
                     "movie": movie, "median_mm": round(median, 4),
                     "reference_mm": round(ref, 4), "deviation_pct": round(dev, 2),
                     "frames_off_pct": round(off, 1),
                     "flagged": abs(dev) > threshold_pct})
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="predict_output/<run> dirs")
    ap.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT,
                    help=f"flag a movie whose median length is more than this "
                         f"far from its reference (default {DEFAULT_THRESHOLD_PCT:g})")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--reference-run",
                       help="take each movie's reference from the same movie in "
                            "this run")
    group.add_argument("--reference-mm", type=float,
                       help="one reference length for every movie")
    ap.add_argument("--csv", help="also write the table here")
    args = ap.parse_args()

    reference = movie_lengths(args.reference_run) if args.reference_run else None
    rows = []
    for run in args.runs:
        found = check_run(run, args.threshold_pct, args.reference_mm, reference)
        if not found:
            print(f"{run}: no analysed movies")
        rows += found
    if not rows:
        sys.exit(1)
    print(f"{'run':<34}{'movie':<30}{'median':>8}{'ref':>8}{'dev %':>8}"
          f"{'frames off %':>14}")
    for r in rows:
        print(f"{r['run'][:33]:<34}{r['movie'][:29]:<30}{r['median_mm']:>8.3f}"
              f"{r['reference_mm']:>8.3f}{r['deviation_pct']:>+8.1f}"
              f"{r['frames_off_pct']:>14.1f}" + ("   <-- FLAGGED" if r["flagged"] else ""))
    flagged = [r for r in rows if r["flagged"]]
    print(f"\n{len(flagged)} of {len(rows)} movie(s) flagged (|deviation| > "
          f"{args.threshold_pct:g} % from "
          + ("the same movie in " + args.reference_run if args.reference_run
             else f"{args.reference_mm} mm" if args.reference_mm
             else "the run's median") + ")")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
