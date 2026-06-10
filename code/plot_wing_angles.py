"""
Quick validation plot: wing angles vs time from *_analysis_smoothed.h5 files.

Plots phi (stroke), theta (deviation), psi (rotation) for left and right wing
on three stacked subplots. Saves a PNG next to each h5.

Usage:
    python code/plot_wing_angles.py <dir>

<dir> may be either:
  - a directory containing exactly one *_analysis_smoothed.h5 (single mode), or
  - a directory whose immediate sub-directories each contain one
    *_analysis_smoothed.h5 (multi mode).
The mode is detected automatically.
"""
import argparse
import glob
import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLING_RATE = 16000  # Hz, matches extract_flight_data.SAMPLING_RATE
SUFFIX = "_analysis_smoothed.h5"

ANGLE_PAIRS = [
    ("wings_phi_left",   "wings_phi_right",   "phi (stroke)"),
    ("wings_theta_left", "wings_theta_right", "theta (deviation)"),
    ("wings_psi_left",   "wings_psi_right",   "psi (rotation)"),
]


def load(h5, key):
    if key not in h5:
        return None
    ds = h5[key]
    return ds[()] if ds.shape == () else ds[:]


def find_h5(directory):
    matches = sorted(glob.glob(os.path.join(directory, f"*{SUFFIX}")))
    return matches


def plot_one(h5_path, units):
    out_path = os.path.join(os.path.dirname(h5_path), "wing_angles.png")

    with h5py.File(h5_path, "r") as h5:
        first = int(load(h5, "first_analysed_frame")) if "first_analysed_frame" in h5 else 0
        series = []
        for left_k, right_k, label in ANGLE_PAIRS:
            l = load(h5, left_k)
            r = load(h5, right_k)
            if l is None or r is None:
                print(f"warning: missing {left_k}/{right_k} in {h5_path}, skipping", file=sys.stderr)
                continue
            series.append((label, l, r))

    if not series:
        print(f"No wing angle datasets found in {h5_path}.", file=sys.stderr)
        return False

    n_frames = max(len(l) for _, l, _ in series)
    if units == "s":
        x = (np.arange(n_frames) + first) / SAMPLING_RATE
        xlabel = "time (s)"
    elif units == "ms":
        x = (np.arange(n_frames) + first) / SAMPLING_RATE * 1000
        xlabel = "time (ms)"
    else:
        x = np.arange(n_frames) + first
        xlabel = "frame"

    fig, axes = plt.subplots(len(series), 1, figsize=(14, 3 * len(series)),
                             sharex=True)
    if len(series) == 1:
        axes = [axes]

    for ax, (label, l, r) in zip(axes, series):
        ax.plot(x[:len(l)], l, label="left",  color="tab:blue",   linewidth=0.9)
        ax.plot(x[:len(r)], r, label="right", color="tab:orange", linewidth=0.9)
        ax.set_ylabel(f"{label}\n(deg)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(os.path.basename(h5_path), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory",
                    help="Directory containing one *_analysis_smoothed.h5, "
                         "or a directory of such sub-directories")
    ap.add_argument("--units", choices=["s", "ms", "frames"], default="ms",
                    help="X-axis units (default: ms)")
    args = ap.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    direct = find_h5(args.directory)

    if len(direct) == 1:
        # single mode
        plot_one(direct[0], args.units)
        return

    if len(direct) > 1:
        print(f"Found {len(direct)} *{SUFFIX} files directly in {args.directory}; "
              f"expected exactly one for single mode.", file=sys.stderr)
        sys.exit(1)

    # multi mode: look in immediate sub-directories
    subdirs = sorted(d for d in (os.path.join(args.directory, name)
                                 for name in os.listdir(args.directory))
                     if os.path.isdir(d))
    n_ok = n_skip = 0
    for sub in subdirs:
        matches = find_h5(sub)
        if len(matches) == 0:
            continue
        if len(matches) > 1:
            print(f"Skipping {sub}: found {len(matches)} *{SUFFIX} files (expected 1)",
                  file=sys.stderr)
            n_skip += 1
            continue
        if plot_one(matches[0], args.units):
            n_ok += 1

    if n_ok == 0 and n_skip == 0:
        print(f"No *{SUFFIX} files found in {args.directory} or its sub-directories.",
              file=sys.stderr)
        sys.exit(1)

    print(f"done: {n_ok} plotted, {n_skip} skipped")


if __name__ == "__main__":
    main()
