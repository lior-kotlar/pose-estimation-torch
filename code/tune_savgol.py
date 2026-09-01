"""Compare Savitzky-Golay (window_length, polyorder) settings for FlightAnalysis.get_dot.

The question this answers is "which filter removes noise without removing real
features", and it is answerable exactly, because savgol with deriv=1 is a linear
filter: its frequency response is computable from the coefficients alone, with no
data and no ground truth. Put that response next to the measured power spectrum of
the signal being differentiated and the trade-off is visible directly --

  * where |H(f)| tracks the ideal differentiator 2*pi*f, the filter is faithful;
  * where it falls below, real signal is being attenuated;
  * above the signal band, whatever it passes is amplified noise.

So the criterion used here is: maximise noise rejection subject to tracking the
ideal differentiator to within TOL over the band that holds the signal's power.

Run:
    .env/bin/python code/tune_savgol.py <analysis_smoothed.h5> [-o out.png]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from scipy.signal import savgol_coeffs, freqz, savgol_filter

SAMPLING_RATE = 16000.0
# how far |H(f)| may drift from the ideal differentiator before we call the filter
# unfaithful. 5% is tight enough that a feature at f_track is essentially unattenuated.
TOL = 0.05
# The band each signal family must keep, in Hz. This is a physics decision, not
# something to infer from the spectrum: the measured spectra here range from a
# smooth power law (CM) to a sharp carrier plus harmonics (wing phi), and no
# single automatic rule reads both correctly. The spectrum panel plots the
# measured power against these lines so the assumption stays checkable.
#
#   body      -- body dynamics are slow; 99% of omega_body's power is under
#                62 Hz on these movies, so 150 Hz is comfortable margin.
#   wing      -- phi is a ~230 Hz carrier with sharp stroke reversals, so the
#                first two harmonics carry the reversal shape. 700 Hz keeps them.
REQUIRED_BAND_HZ = {"body": 150.0, "wing": 700.0}

# Candidates. Ordered from the current setting (5, 2) towards heavier smoothing.
# polyorder 4 buys a much wider passband than polyorder 2 at the same window, which
# is what makes the long windows usable at all on the wing signals.
CANDIDATES = [(5, 2), (11, 2), (51, 4), (101, 4), (201, 4)]

# Categorical slots 1-5 of the validated reference palette (light surface).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def signals_from_h5(path):
    """The three signal families get_dot is applied to, each with its own bandwidth."""
    with h5py.File(path, "r") as f:
        return [
            ("body rotation\nomega_body x", f["omega_body"][:, 0], "deg/s$^2$", "body"),
            ("body translation\nCM z", f["center_of_mass"][:, 2] * 1000.0, "mm/s", "body"),
            ("wing angle\nphi left", f["wings_phi_left"][:], "deg/s", "wing"),
        ]


def response(window, order, n=8192):
    """|H(f)| of the savgol derivative filter, and the ideal differentiator."""
    coeffs = savgol_coeffs(window, order, deriv=1, delta=1.0 / SAMPLING_RATE)
    freq, h = freqz(coeffs, worN=n, fs=SAMPLING_RATE)
    return freq, np.abs(h), 2 * np.pi * freq


def track_limit(window, order):
    """Highest frequency the filter still differentiates faithfully (within TOL)."""
    freq, mag, ideal = response(window, order)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(mag - ideal) / np.where(ideal == 0, np.nan, ideal)
    bad = np.where(err[1:] >= TOL)[0]
    return freq[1:][bad[0] - 1] if len(bad) else freq[-1]


def noise_gain(window, order):
    """White-noise amplification, relative to the current (5, 2) setting."""
    g = lambda w, o: np.sqrt(np.sum(savgol_coeffs(w, o, deriv=1, delta=1.0 / SAMPLING_RATE) ** 2))
    return g(window, order) / g(*CANDIDATES[0])


def spectrum(x):
    """Power spectrum, and the frequency below which 99% of the power sits."""
    x = x[np.isfinite(x)]
    x = x - x.mean()
    freq = np.fft.rfftfreq(len(x), 1.0 / SAMPLING_RATE)
    power = np.abs(np.fft.rfft(x)) ** 2
    f99 = freq[np.searchsorted(np.cumsum(power) / power.sum(), 0.99)]
    return freq, power, f99


def longest_finite_run(x):
    finite = np.isfinite(x)
    if not finite.any():
        return 0, 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], finite.view(np.int8), [0]))))
    starts, ends = idx[::2], idx[1::2]
    k = np.argmax(ends - starts)
    return int(starts[k]), int(ends[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5_path")
    ap.add_argument("-o", "--out", default="savgol_comparison.png")
    ap.add_argument("--zoom", type=int, default=None,
                    help="frames shown in the time-domain panel (default: per-signal)")
    args = ap.parse_args()

    signals = signals_from_h5(args.h5_path)
    stats = [(w, o, track_limit(w, o), noise_gain(w, o)) for w, o in CANDIDATES]

    print(f"\nSavitzky-Golay candidates at {SAMPLING_RATE:.0f} Hz "
          f"(faithful to within {TOL:.0%} of the ideal differentiator)\n")
    print(f"{'window':>7s} {'ms':>7s} {'order':>6s} {'passband':>11s} {'noise gain':>12s}")
    for w, o, ft, ng in stats:
        print(f"{w:>7d} {w / SAMPLING_RATE * 1000:>7.2f} {o:>6d} "
              f"{ft:>9.0f} Hz {ng:>11.3f}x")

    print(f"\n{'signal':>26s} {'99% power':>11s} {'band required':>14s}   "
          f"recommended (widest window that still passes it)")
    for name, data, _, family in signals:
        _, _, f99 = spectrum(data)
        need = REQUIRED_BAND_HZ[family]
        safe = [(w, o, ng) for w, o, ft, ng in stats if ft >= need]
        best = min(safe, key=lambda t: t[2]) if safe else None
        label = f"window={best[0]}, polyorder={best[1]}  ({best[2]:.3f}x noise)" if best \
            else "none -- every candidate would attenuate real signal"
        print(f"{name.replace(chr(10), ' '):>26s} {f99:>8.0f} Hz {need:>11.0f} Hz   {label}")

    n_rows = len(signals)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16.5, 3.6 * n_rows))
    fig.patch.set_facecolor("#fcfcfb")

    for row, (name, data, unit_out, family) in enumerate(signals):
        freq, power, f99 = spectrum(data)
        need = REQUIRED_BAND_HZ[family]
        lo, hi = longest_finite_run(data)
        segment = data[lo:hi]

        # --- col 0: where the signal's power actually is -----------------------
        ax = axes[row, 0]
        keep = freq > 0
        ax.loglog(freq[keep], power[keep] / power[keep].max(), lw=0.8,
                  color=INK_MUTED, alpha=0.75)
        ax.axvspan(freq[1], need, color="#2a78d6", alpha=0.10, lw=0)
        ax.axvline(need, color=INK, lw=1.4, ls="--")
        ax.annotate(f"band required:\n{need:.0f} Hz\n(99% of power\nbelow {f99:.0f} Hz)", xy=(need, 0.25),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center")
        ax.set_title(f"{name}\nsignal power spectrum", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("frequency (Hz)", fontsize=9, color=INK_MUTED)
        ax.set_ylabel("normalised power", fontsize=9, color=INK_MUTED)

        # --- col 1: does each filter differentiate faithfully there? -----------
        ax = axes[row, 1]
        for (w, o), color in zip(CANDIDATES, SERIES_COLORS):
            f, mag, ideal = response(w, o)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = mag[1:] / ideal[1:]
            ax.semilogx(f[1:], ratio, lw=1.8, color=color, label=f"w={w}, p={o}")
        ax.axhline(1.0, color=INK, lw=1.0, ls=":")
        ax.axvspan(freq[1], need, color="#2a78d6", alpha=0.10, lw=0)
        ax.set_ylim(0, 1.45)
        ax.set_xlim(1, SAMPLING_RATE / 2)
        ax.set_title("fidelity: |H(f)| / ideal differentiator\n"
                     "1.0 = exact; shaded = band that must be kept",
                     fontsize=10, color=INK, loc="left")
        ax.set_xlabel("frequency (Hz)", fontsize=9, color=INK_MUTED)
        ax.set_ylabel("gain relative to ideal", fontsize=9, color=INK_MUTED)
        if row == 0:
            ax.legend(fontsize=8.5, frameon=False, labelcolor=INK, loc="lower left")

        # --- col 2: the derivatives themselves, overlaid -----------------------
        ax = axes[row, 2]
        span = args.zoom or (250 if "wing" in name else 900)
        span = min(span, len(segment) - 1)
        start = max(0, len(segment) // 2 - span // 2)
        sl = slice(start, start + span)
        t_ms = np.arange(span) / SAMPLING_RATE * 1000.0
        for (w, o), color in zip(CANDIDATES, SERIES_COLORS):
            if w >= len(segment):
                continue
            deriv = savgol_filter(segment, w, o, deriv=1, delta=1.0 / SAMPLING_RATE)
            ax.plot(t_ms, deriv[sl], lw=1.4, color=color, alpha=0.9,
                    label=f"w={w}, p={o}")
        ax.set_title(f"resulting derivative ({unit_out})\n"
                     f"same {span} frames, every candidate",
                     fontsize=10, color=INK, loc="left")
        ax.set_xlabel("time (ms)", fontsize=9, color=INK_MUTED)
        ax.set_ylabel(unit_out, fontsize=9, color=INK_MUTED)

    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5)

    fig.suptitle("Savitzky-Golay derivative filters: passband vs noise rejection      "
                 f"({os.path.basename(args.h5_path)})",
                 fontsize=12.5, color=INK, x=0.005, ha="left", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
