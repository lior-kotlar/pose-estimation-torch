"""
Quick validation plot: wing angles vs time from *_analysis_smoothed.h5 files.

Plots phi (stroke), theta (deviation), psi (rotation) for left and right wing
on three stacked subplots. Saves a PNG next to each h5.

The x-axis is trigger-relative (frame 0 == the hardware trigger, earlier frames
negative), the same numbering as the counter burnt into the prediction mp4 and
as the `frame` column of the analysis CSV -- so a disturbance spotted here can
be looked up directly in the video. A second axis on top gives the same instant
in the other unit (ms when the primary axis is frames, and vice versa).

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


def frame_axis(h5, n_frames):
    """Trigger-relative frame number of every array index, plus the frame rate.

    create_movie_analysis_h5 stores `frame_index` (and `trigger_offset` /
    `frame_rate`) next to the per-frame series, using the same convention as the
    counter drawn in the prediction mp4 and the analysis CSV: frame 0 is the
    hardware trigger, frames recorded before it are negative. Reusing it is what
    makes this plot and the video share one x-axis.

    Returns (frames, frame_rate, is_trigger_relative). Movies analysed before
    those datasets existed fall back to plain box-frame numbering, which does
    NOT line up with the video counter.
    """
    rate = load(h5, "frame_rate")
    rate = float(rate) if rate is not None else None
    # Every per-frame series is NaN-padded at the front by `first_analysed_frame`,
    # so array index i is box frame (i - first_analysed_frame).
    first = int(load(h5, "first_analysed_frame")) if "first_analysed_frame" in h5 else 0

    frame_index = load(h5, "frame_index")
    if frame_index is not None and len(frame_index) >= n_frames:
        return np.asarray(frame_index[:n_frames], dtype=float), rate, True

    trigger_offset = load(h5, "trigger_offset")
    if trigger_offset is not None:
        # box frame 0 sits at trigger_offset
        return int(trigger_offset) + (np.arange(n_frames) - first), rate, True

    return np.arange(n_frames) - first, rate, False


def plot_one(h5_path, units):
    out_path = os.path.join(os.path.dirname(h5_path), "wing_angles.png")

    with h5py.File(h5_path, "r") as h5:
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
        frames, rate, trigger_relative = frame_axis(h5, n_frames)

    if rate is None:
        rate = SAMPLING_RATE
    if not trigger_relative:
        print(f"warning: no trigger info in {h5_path}; x-axis is box frames and "
              f"will NOT match the mp4 counter", file=sys.stderr)
    relative = " relative to trigger" if trigger_relative else " (box index)"

    if units == "s":
        x = frames / rate
        xlabel = f"time (s){relative}"
    elif units == "ms":
        x = frames / rate * 1000
        xlabel = f"time (ms){relative}"
    else:
        x = frames
        xlabel = f"frame{relative}"

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
        # Mark the trigger itself, the one landmark shared with the video.
        if trigger_relative and x[0] <= 0 <= x[-1]:
            ax.axvline(0, color="0.4", linestyle="--", linewidth=0.8)

    # Second scale on top: the same instant in the other unit, so a feature can
    # be read off either as the mp4's frame counter or as its clock.
    if units == "frames":
        secondary = axes[0].secondary_xaxis(
            "top", functions=(lambda f: f / rate * 1000, lambda t: t * rate / 1000))
        secondary.set_xlabel(f"time (ms){relative}")
    else:
        scale = rate if units == "s" else rate / 1000
        secondary = axes[0].secondary_xaxis(
            "top", functions=(lambda t: t * scale, lambda f: f / scale))
        secondary.set_xlabel(f"frame{relative}")

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
    ap.add_argument("--units", choices=["s", "ms", "frames"], default="frames",
                    help="Primary x-axis units, always trigger-relative "
                         "(default: frames, matching the mp4 counter). The "
                         "other unit is drawn as a second axis on top.")
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
