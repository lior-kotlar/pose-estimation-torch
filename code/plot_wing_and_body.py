"""
Quick validation plots from *_analysis_smoothed.h5 files.

Two PNGs are written next to each h5:

  wing_angles.png                phi (stroke), theta (deviation), psi (rotation)
                                 for left and right wing, on three stacked
                                 subplots.
  body_angular_acceleration.png  the body's angular acceleration: the three
                                 components of d(omega_body)/dt about the body
                                 axes, and its magnitude.

The x-axis is trigger-relative (frame 0 == the hardware trigger, earlier frames
negative), the same numbering as the counter burnt into the prediction mp4 and
as the `frame` column of the analysis CSV -- so a disturbance spotted here can
be looked up directly in the video. A second axis on top gives the same instant
in the other unit (ms when the primary axis is frames, and vice versa).

When the movie declares a perturbation, the axis is zeroed on the ONSET instead
(--origin), since that is the instant the experiment is about. Only the drawn
axis moves: the onset is stored trigger-relative like everything else, so this
is a shift, and the numbering in the h5, the CSV and the mp4 counter is
untouched. The axis label always names its own zero, and the trigger is still
marked, so the two readings can never be confused.

Usage:
    python code/plot_wing_and_body.py <dir>

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

WING_PNG = "wing_angles.png"
BODY_PNG = "body_angular_acceleration.png"
# Every figure this module writes, for callers that archive the products they
# are about to overwrite (see reanalyse_movies.archive_previous).
FIGURE_NAMES = (WING_PNG, BODY_PNG)

# One colour per QUANTITY rather than per slot in a panel: a signal keeps its
# colour wherever it is drawn, so the roll angle and the roll acceleration are
# the same blue in every figure and in the interactive viewer (which imports
# this map). The three spatial axes share one triple whether they are lab axes
# or body axes -- the label says which frame, the colour says which axis.
ROLE_COLORS = {
    "left":  "#1f77b4",   # tab:blue
    "right": "#ff7f0e",   # tab:orange
    "x":     "#1f77b4",   # x_lab / x_body / roll
    "y":     "#ff7f0e",   # y_lab / y_body / pitch
    "z":     "#2ca02c",   # z_lab / z_body / yaw    (tab:green)
    "mag":   "#9467bd",   # a magnitude, which has no axis  (tab:purple)
}

ANGLE_PAIRS = [
    ("wings_phi_left",   "wings_phi_right",   "phi (stroke)"),
    ("wings_theta_left", "wings_theta_right", "theta (deviation)"),
    ("wings_psi_left",   "wings_psi_right",   "psi (rotation)"),
]

# omega_body's three components are the roll, pitch and yaw rates about the
# fly's own axes (extract_flight_data names them p, q, r), so the components of
# its derivative are the angular accelerations about those same axes. The line
# style repeats the identity the colour carries, so the three stay separable
# where they overlap and for a colour-blind reader.
OMEGA_DOT_COMPONENTS = [
    (0, "about x_body (roll)",  ROLE_COLORS["x"], "-"),
    (1, "about y_body (pitch)", ROLE_COLORS["y"], "--"),
    (2, "about z_body (yaw)",   ROLE_COLORS["z"], ":"),
]

# Savitzky-Golay settings for the fallback derivative below -- the body-signal
# values from extract_flight_data.BODY_SAVGOL, copied rather than imported so
# this script stays runnable without the prediction environment. A recomputed
# omega_body_dot is then the one the analysis would write today, not a
# differently-filtered second opinion. (An h5 written before those settings
# were split per signal family carries a more lightly filtered omega_body_dot;
# this plots whatever the file holds, and only recomputes when it holds none.)
BODY_SAVGOL = {"window_length": 51, "polyorder": 4}

# A fly's body angular acceleration runs to six figures in deg/s^2, so the
# panels are drawn in thousands: the unit is spelt out in the axis label rather
# than left as an exponent floating above the axis, where it collides with the
# secondary (time) axis drawn along the top.
KILO = 1e-3
ACCEL_UNIT = r"$10^3$ deg/s$^2$"


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


def perturbation_info(h5):
    """The declared perturbation window in trigger-relative frames, or None.

    The onset is always known once an experiment is declared perturbed; the
    duration often is not (perturbation_end_known == 0), and an unknown end is
    left as None rather than guessed at.
    """
    if not load(h5, "perturbation"):
        return None
    onset = load(h5, "perturbation_start_frame")
    if onset is None:
        return None
    end_known = bool(load(h5, "perturbation_end_known"))
    end = load(h5, "perturbation_end_frame") if end_known else None
    kind = load(h5, "perturbation_type")
    label = kind.decode() if isinstance(kind, bytes) else str(kind or "perturbation")
    if not end_known:
        label += " (end unrecorded)"
    return {"onset": float(onset),
            "end": float(end) if end is not None else None,
            "end_known": end_known, "label": label}


def axis_origin(pert, mode, trigger_relative=True):
    """(frame the x-axis calls zero, name for that zero) under `mode`.

    "auto" means the onset whenever the movie declares a perturbation, because
    that is the instant such an experiment is about; without one it means the
    trigger, which is what every other product uses.
    """
    if mode in ("auto", "perturbation") and pert is not None and trigger_relative:
        return pert["onset"], "perturbation onset"
    if mode == "perturbation":
        print("warning: --origin perturbation, but this movie declares no "
              "perturbation window; falling back to the trigger", file=sys.stderr)
    return 0.0, None


def frame_to_x(rate, units, origin=0.0):
    """Map a trigger-relative frame number onto the drawn x-axis.

    One converter for both the traces and the shapes drawn at a given frame
    (the perturbation band, the trigger marker), so a landmark cannot land
    somewhere the data does not, whichever unit and origin are in force.
    """
    scale = 1.0 if units == "frames" else (1.0 / rate if units == "s" else 1000.0 / rate)
    return lambda f: (np.asarray(f, dtype=float) - float(origin)) * scale


def x_axis(frames, rate, units, trigger_relative, origin=0.0, origin_name=None):
    """(x values, axis label, the phrase naming the zero) for the requested unit.

    The label always names the zero, so a figure can never be read against the
    wrong landmark.
    """
    if origin_name:
        relative = f" relative to {origin_name}"
    else:
        relative = " relative to trigger" if trigger_relative else " (box index)"
    x = frame_to_x(rate, units, origin)(frames)
    if units == "s":
        return x, f"time (s){relative}", relative
    if units == "ms":
        return x, f"time (ms){relative}", relative
    return x, f"frame{relative}", relative


def add_secondary_axis(ax, units, rate, relative):
    """Draw the same instant in the other unit along the top of `ax`.

    Not a second measure on a second scale -- it is one x-axis relabelled, so a
    feature can be read off either as the mp4's frame counter or as its clock.
    """
    if units == "frames":
        secondary = ax.secondary_xaxis(
            "top", functions=(lambda f: f / rate * 1000, lambda t: t * rate / 1000))
        secondary.set_xlabel(f"time (ms){relative}")
    else:
        scale = rate if units == "s" else rate / 1000
        secondary = ax.secondary_xaxis(
            "top", functions=(lambda t: t * scale, lambda f: f / scale))
        secondary.set_xlabel(f"frame{relative}")
    return secondary


def savgol_dot(data, rate):
    """d(data)/dt in units-per-second, filtered the way the pipeline filters it.

    Mirrors FlightAnalysis.get_dot (including its shrink-the-window-to-fit
    behaviour) so the fallback path below produces the same curve the analysis
    would have stored. NaN-padded ends are left NaN rather than fed to the
    filter, which would spread them across the whole window.
    """
    data = np.asarray(data, dtype=float)
    out = np.full(data.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(data))
    if finite.size < 3:
        return out
    lo, hi = finite[0], finite[-1] + 1
    segment = data[lo:hi]
    if not np.all(np.isfinite(segment)):
        return out  # interior gaps: no honest way to differentiate across them
    window = min(BODY_SAVGOL["window_length"], len(segment))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        out[lo:hi] = np.gradient(segment) * rate
        return out
    from scipy.signal import savgol_filter  # only the fallback path needs scipy  # only the fallback path needs scipy
    polyorder = min(BODY_SAVGOL["polyorder"], max(1, window - 1))
    out[lo:hi] = savgol_filter(segment, window, polyorder, deriv=1, delta=1.0 / rate)
    return out


def body_angular_acceleration(h5, rate):
    """(components, magnitude) of d(omega_body)/dt in deg/s^2, or (None, None).

    Both are stored by the current pipeline. Movies analysed before
    `omega_body_dot` existed still carry `omega_body`, so differentiate that
    instead rather than refusing to plot.
    """
    omega_dot = load(h5, "omega_body_dot")
    magnitude = load(h5, "angular_acceleration_body")

    if omega_dot is None:
        omega = load(h5, "omega_body")
        if omega is None:
            return None, None
        omega = np.asarray(omega, dtype=float)
        omega_dot = np.column_stack([savgol_dot(omega[:, ax], rate)
                                     for ax in range(omega.shape[1])])
        magnitude = None  # a stored magnitude would not match a recomputed omega_dot

    omega_dot = np.asarray(omega_dot, dtype=float)
    if magnitude is None:
        magnitude = np.linalg.norm(omega_dot, axis=-1)
    return omega_dot, np.asarray(magnitude, dtype=float)


def mark_perturbation(ax, pert, to_x, frames):
    """Shade the perturbation window, when the experiment declares one.

    An unknown end is drawn as a band running to the edge of the axis rather
    than guessed at. An onset outside the built range is normal -- the prescan
    picks its range from fly visibility and knows nothing about the
    perturbation -- so the band is clipped to what was recorded and only the
    onset LINE is withheld when it falls outside.
    """
    if pert is None:
        return
    lo, hi = float(frames[0]), float(frames[-1])
    end = pert["end"] if pert["end"] is not None else hi
    band = (max(pert["onset"], lo), min(end, hi))
    if band[1] > band[0]:
        ax.axvspan(float(to_x(band[0])), float(to_x(band[1])), color="tab:red",
                   alpha=0.08, zorder=0)
    if lo <= pert["onset"] <= hi:
        ax.axvline(float(to_x(pert["onset"])), color="tab:red", linestyle="-",
                   linewidth=0.9, alpha=0.6, zorder=0, label=pert["label"])


def mark_trigger(ax, to_x, frames, trigger_relative):
    """The trigger itself, the one landmark shared with the video.

    Drawn wherever it falls, which is x=0 only while the axis is zeroed on the
    trigger -- on a perturbation-relative axis it moves to -onset, and is worth
    more there than it ever was at the origin.
    """
    if not trigger_relative or not (frames[0] <= 0 <= frames[-1]):
        return
    ax.axvline(float(to_x(0)), color="0.4", linestyle="--", linewidth=0.8)


def plot_wing_angles(h5_path, units, origin="auto"):
    out_path = os.path.join(os.path.dirname(h5_path), WING_PNG)

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
        pert = perturbation_info(h5)

    if rate is None:
        rate = SAMPLING_RATE
    if not trigger_relative:
        print(f"warning: no trigger info in {h5_path}; x-axis is box frames and "
              f"will NOT match the mp4 counter", file=sys.stderr)

    zero, zero_name = axis_origin(pert, origin, trigger_relative)
    to_x = frame_to_x(rate, units, zero)
    x, xlabel, relative = x_axis(frames, rate, units, trigger_relative, zero, zero_name)

    fig, axes = plt.subplots(len(series), 1, figsize=(14, 3 * len(series)),
                             sharex=True)
    if len(series) == 1:
        axes = [axes]

    for ax, (label, l, r) in zip(axes, series):
        # Perturbation band first, so the traces are drawn over it.
        mark_perturbation(ax, pert, to_x, frames)
        ax.plot(x[:len(l)], l, label="left",  color=ROLE_COLORS["left"],  linewidth=0.9)
        ax.plot(x[:len(r)], r, label="right", color=ROLE_COLORS["right"], linewidth=0.9)
        ax.set_ylabel(f"{label}\n(deg)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        mark_trigger(ax, to_x, frames, trigger_relative)

    # Second scale on top: the same instant in the other unit.
    add_secondary_axis(axes[0], units, rate, relative)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(os.path.basename(h5_path), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def plot_body_angular_acceleration(h5_path, units, origin="auto"):
    """Body angular acceleration vs time, on the same x-axis as the wing plot.

    Top panel: the three components of d(omega_body)/dt, i.e. how fast the roll,
    pitch and yaw rates about the fly's own axes are changing -- signed, so the
    direction of the change is visible. Bottom panel: the magnitude
    ||d(omega_body)/dt||, which is unsigned and says only how hard the body is
    being turned. Both are needed: the magnitude finds the event, the components
    say which way it went.
    """
    out_path = os.path.join(os.path.dirname(h5_path), BODY_PNG)

    with h5py.File(h5_path, "r") as h5:
        rate = load(h5, "frame_rate")
        rate = float(rate) if rate is not None else SAMPLING_RATE

        omega_dot, magnitude = body_angular_acceleration(h5, rate)
        if omega_dot is None:
            print(f"No omega_body/omega_body_dot in {h5_path}; skipping the "
                  f"body angular acceleration plot.", file=sys.stderr)
            return False

        n_frames = len(omega_dot)
        frames, rate_axis, trigger_relative = frame_axis(h5, n_frames)
        if rate_axis is not None:
            rate = rate_axis
        pert = perturbation_info(h5)

    zero, zero_name = axis_origin(pert, origin, trigger_relative)
    to_x = frame_to_x(rate, units, zero)
    x, xlabel, relative = x_axis(frames, rate, units, trigger_relative, zero, zero_name)

    fig, (ax_comp, ax_mag) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Perturbation band first, so the traces are drawn over it.
    mark_perturbation(ax_comp, pert, to_x, frames)
    mark_perturbation(ax_mag, pert, to_x, frames)

    for axis, label, color, style in OMEGA_DOT_COMPONENTS:
        ax_comp.plot(x, omega_dot[:, axis] * KILO, label=label, color=color,
                     linestyle=style, linewidth=0.9)
    ax_comp.axhline(0, color="0.6", linewidth=0.6)
    ax_comp.set_ylabel(f"angular acceleration\ncomponents ({ACCEL_UNIT})")

    ax_mag.plot(x, magnitude * KILO, color=ROLE_COLORS["mag"], linewidth=0.9,
                label=r"$\|\dot{\omega}_{body}\|$")
    ax_mag.set_ylabel(f"angular acceleration\nmagnitude ({ACCEL_UNIT})")
    ax_mag.set_ylim(bottom=0)

    for ax in (ax_comp, ax_mag):
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        mark_trigger(ax, to_x, frames, trigger_relative)

    add_secondary_axis(ax_comp, units, rate, relative)
    ax_mag.set_xlabel(xlabel)
    fig.suptitle(f"{os.path.basename(h5_path)} -- body angular acceleration",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def plot_one(h5_path, units="frames", origin="auto"):
    """Write every per-movie figure for one analysis h5.

    Returns True if at least one figure was written. One figure failing does not
    stop the other: this runs at the tail of every prediction, where a missing
    dataset must cost a plot, not the movie.
    """
    wrote = False
    for plot in (plot_wing_angles, plot_body_angular_acceleration):
        try:
            wrote |= bool(plot(h5_path, units, origin))
        except Exception as e:
            print(f"{plot.__name__} failed for {h5_path}: {e}", file=sys.stderr)
    return wrote


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
    ap.add_argument("--origin", choices=["auto", "trigger", "perturbation"],
                    default="auto",
                    help="Which instant the x-axis calls zero: 'trigger' for "
                         "the numbering the mp4 counter and the CSV use, "
                         "'perturbation' for the declared onset, 'auto' "
                         "(default) for the onset when the movie declares one "
                         "and the trigger otherwise. Only the drawn axis "
                         "moves; the stored numbering never does.")
    args = ap.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    direct = find_h5(args.directory)

    if len(direct) == 1:
        # single mode
        plot_one(direct[0], args.units, args.origin)
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
        if plot_one(matches[0], args.units, args.origin):
            n_ok += 1

    if n_ok == 0 and n_skip == 0:
        print(f"No *{SUFFIX} files found in {args.directory} or its sub-directories.",
              file=sys.stderr)
        sys.exit(1)

    print(f"done: {n_ok} plotted, {n_skip} skipped")


if __name__ == "__main__":
    main()
