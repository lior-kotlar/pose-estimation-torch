"""
compare_camera_subsets.py
=========================

Score the camera subsets that make_camera_subset_movies.py cut out of an
experiment -- e.g. the (bottom, side) pairs of a simulated 2-camera rig --
against the all_cams reference cut from the same movies, frame for frame.

Every subset of a movie covers the same frames, so each pair's prediction is
compared with the reference's at identical trigger-relative frames
(`frame_index` in the analysis h5), on what the consumer receives: the
analysed 3D points and angles.

The reference is the 4-camera ensemble, NOT ground truth -- on the held-out
test frames a 4-camera ensemble is itself ~0.09 mm off -- so every distance
here is "how far a 2-camera rig lands from what the 4-camera rig says", its
own error and the reference's together.

For each (movie, pair):
  3D         distance to the reference per joint group (median, p95, share of
             points > 0.5 mm)
  swaps      frames whose wings are labelled the other way round from the
             reference (wing centres fit it better exchanged, by the ensemble's
             own margin, wing_labels.WING_LABEL_SWAP_MIN_MARGIN)
  angles     |difference| of wing phi/theta/psi and body yaw/pitch/roll, deg
  rigidity   the pipeline's own self-consistency score (compare_ensembles),
             for the pair and the reference
  coverage   frames the analysis left NaN
plus how far the pairs land from each other, and which ensemble members each
subset's selector picked.

Writes <out>/per_movie.csv, summary.json, report.md and
fig_pairs_vs_reference.png (median and p95 per joint group and per angle).

    .env/bin/python code/compare_camera_subsets.py inference_datasets/simulated/shalev/21to40
"""

import argparse
import csv
import glob
import json
import os
import sys
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "prediction_code_lior"))
from compare_ensembles import RAW_NAME, SMOOTHED_NAME, rigidity    # noqa: E402
from make_camera_subset_movies import (REFERENCE_SUBSET, REPO_ROOT,  # noqa: E402
                                       run_name)
from process_experiment import DERIVED_FROM_FILE, find_movie_h5      # noqa: E402
from wing_labels import WING_LABEL_SWAP_MIN_MARGIN                   # noqa: E402

JOINT_GROUPS = {"left wing": list(range(0, 7)), "right wing": list(range(8, 15)),
                "wing hinges": [7, 15], "head & tail": [16, 17],
                "all": list(range(18))}
WING_POINTS = (list(range(0, 7)), list(range(8, 15)))
ANGLES = {"phi L": "wings_phi_left", "theta L": "wings_theta_left",
          "psi L": "wings_psi_left", "phi R": "wings_phi_right",
          "theta R": "wings_theta_right", "psi R": "wings_psi_right",
          "yaw": "yaw_angle", "pitch": "pitch_angle", "roll": "roll_angle"}
FAR_MM = 0.5                       # "a point this far off is a miss"
# Figure: categorical slots 1-3 in fixed order (validated: CVD dE >= 9.2);
# slot 3 sits under 3:1 on the surface, so every series is also labelled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def find_movies(cut_dir):
    """({movie: {subset: movie dir}}, the source experiment) for every
    finished cut under cut_dir."""
    movies, experiments = {}, set()
    for marker in sorted(glob.glob(os.path.join(cut_dir, "*", "mov*",
                                                DERIVED_FROM_FILE))):
        with open(marker) as f:
            rec = json.load(f)
        if "state" in rec:                          # a cut that never finished
            continue
        movie_dir = os.path.dirname(marker)
        movies.setdefault(os.path.basename(movie_dir), {})[
            rec["subset"]["name"]] = movie_dir
        experiments.add(rec["source"]["experiment"])
    if len(experiments) > 1:
        sys.exit(f"{cut_dir} holds cuts of several experiments: "
                 f"{', '.join(sorted(experiments))}; point at one of them")
    return movies, (experiments.pop() if experiments else None)


def prediction_dir(predict_output, experiment, subset, movie_dir):
    h5 = find_movie_h5(movie_dir)
    base = os.path.splitext(os.path.basename(h5))[0] if h5 else ""
    return os.path.join(predict_output, run_name(experiment, subset), base)


def load_run(pred_dir):
    """The analysed points and angles of one prediction, or None when it has
    not finished (no analysis h5 yet)."""
    base = os.path.basename(pred_dir)
    analysis = os.path.join(pred_dir, f"{base}_analysis_smoothed.h5")
    if not os.path.isfile(analysis):
        return None
    with h5py.File(analysis, "r") as f:
        run = {"frames": f["frame_index"][:], "points": f["points_3D"][:],
               "angles": {k: f[v][:] for k, v in ANGLES.items() if v in f}}
    run["rigidity"] = float("nan")
    smoothed = os.path.join(pred_dir, SMOOTHED_NAME)
    if os.path.isfile(smoothed):
        run["rigidity"] = rigidity(np.load(smoothed))[0]
    raw = os.path.join(pred_dir, RAW_NAME)
    if os.path.isfile(raw):
        run["rigidity_raw"] = rigidity(np.load(raw))[0]
    sel = os.path.join(pred_dir, "ensemble_model_selection_summary.json")
    if os.path.isfile(sel):
        with open(sel) as f:
            overall = json.load(f).get("overall", {})
        run["selection"] = {name.split(" ")[0]: v["fraction_of_frames_selected"]
                            for name, v in overall.items()}
    return run


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------
def aligned(a, b):
    """Indices into a and b of the frames both have."""
    common, ia, ib = np.intersect1d(a["frames"], b["frames"], return_indices=True)
    return ia, ib


def wrapped_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def swapped_frames(p, r):
    """Frames whose wings fit the reference better exchanged (the rule of
    wing_labels.harmonize_wing_labels, with the reference as the median)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pl, pr = (np.nanmean(p[:, w], axis=1) for w in WING_POINTS)
        rl, rr = (np.nanmean(r[:, w], axis=1) for w in WING_POINTS)
    keep = np.linalg.norm(pl - rl, axis=-1) + np.linalg.norm(pr - rr, axis=-1)
    exchange = np.linalg.norm(pl - rr, axis=-1) + np.linalg.norm(pr - rl, axis=-1)
    return (keep - exchange) > WING_LABEL_SWAP_MIN_MARGIN


def compare(pair, ref):
    """Per-frame arrays of one pair against the reference."""
    ip, ir = aligned(pair, ref)
    p, r = pair["points"][ip], ref["points"][ir]
    out = {"n_frames": len(ip),
           "dist_mm": np.linalg.norm(p - r, axis=-1) * 1000.0,   # (frames, 18)
           "swapped": swapped_frames(p, r),
           "nan_frames": int(np.isnan(p).any(axis=(1, 2)).sum()),
           "ref_nan_frames": int(np.isnan(r).any(axis=(1, 2)).sum())}
    out["angles"] = {k: np.abs(wrapped_deg(pair["angles"][k][ip]
                                           - ref["angles"][k][ir]))
                     for k in ANGLES if k in pair["angles"] and k in ref["angles"]}
    return out


def stats(values):
    v = np.asarray(values, dtype=float).ravel()
    v = v[~np.isnan(v)]
    if not len(v):
        return {"median": float("nan"), "p95": float("nan"), "n": 0}
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "n": int(len(v))}


def summarize(frames):
    """Pool a list of compare() results into one set of numbers."""
    dist = np.concatenate([c["dist_mm"] for c in frames], axis=0)
    swapped = np.concatenate([c["swapped"] for c in frames])
    out = {"n_frames": int(sum(c["n_frames"] for c in frames)),
           "nan_frames": int(sum(c["nan_frames"] for c in frames)),
           "swapped_frames": int(swapped.sum()),
           "swapped_pct": 100.0 * float(swapped.mean()) if len(swapped) else float("nan"),
           "groups": {}, "angles": {}}
    for g, joints in JOINT_GROUPS.items():
        d = dist[:, joints]
        s = stats(d)
        valid = d[~np.isnan(d)]
        s["far_pct"] = 100.0 * float((valid > FAR_MM).mean()) if len(valid) else float("nan")
        out["groups"][g] = s
    for k in ANGLES:
        if all(k in c["angles"] for c in frames):
            out["angles"][k] = stats(np.concatenate([c["angles"][k] for c in frames]))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_report(path, experiment, pooled, per_movie, between, selection,
                 missing, pairs):
    L = [f"# Simulated 2-camera rig vs the 4-camera reference: {experiment}", "",
         "Each (bottom, side) pair was cut from the same frames as the all_cams "
         "reference and predicted as a 2-camera movie. Distances are to the "
         "4-camera ensemble, which is not ground truth (about 0.09 mm itself on "
         "held-out frames).", ""]
    if missing:
        L += ["Not predicted yet (left out): " + ", ".join(missing), ""]
    L += ["## 3D distance to the reference, all movies pooled (mm)", "",
          "| pair | frames | all median | all p95 | % > 0.5 mm | left wing p95 | "
          "right wing p95 | hinges p95 | head & tail p95 | swapped frames |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pair in pairs:
        s = pooled[pair]
        g = s["groups"]
        L.append(f"| {pair} | {s['n_frames']} | {fmt(g['all']['median'])} | "
                 f"{fmt(g['all']['p95'])} | {fmt(g['all']['far_pct'], 1)} | "
                 f"{fmt(g['left wing']['p95'])} | {fmt(g['right wing']['p95'])} | "
                 f"{fmt(g['wing hinges']['p95'])} | {fmt(g['head & tail']['p95'])} | "
                 f"{s['swapped_frames']} ({fmt(s['swapped_pct'], 1)}%) |")
    L += ["", "## Angle differences to the reference, all movies pooled "
          "(degrees, median / p95)", "",
          "| pair | " + " | ".join(ANGLES) + " |",
          "|---|" + "---:|" * len(ANGLES)]
    for pair in pairs:
        a = pooled[pair]["angles"]
        L.append(f"| {pair} | " + " | ".join(
            f"{fmt(a[k]['median'], 1)} / {fmt(a[k]['p95'], 1)}" if k in a else "–"
            for k in ANGLES) + " |")
    L += ["", "## Per movie", "",
          "| movie | pair | frames | all median mm | all p95 mm | swapped | NaN "
          "frames | rigidity pair | rigidity reference |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in per_movie:
        L.append(f"| {row['movie']} | {row['pair']} | {row['n_frames']} | "
                 f"{fmt(row['all_median_mm'])} | {fmt(row['all_p95_mm'])} | "
                 f"{row['swapped_frames']} | {row['nan_frames']} | "
                 f"{fmt(row['rigidity_pair'] * 1e6, 1)} | "
                 f"{fmt(row['rigidity_ref'] * 1e6, 1)} |")
    L += ["", "Rigidity is the pipeline's own score (mean std of the wing edge "
          "lengths, µm; lower is steadier), which the ensemble selector "
          "minimises -- necessary, not sufficient.", ""]
    if between:
        L += ["## The pairs against each other, all movies pooled (mm, all joints)",
              "", "| pairs | median | p95 |", "|---|---:|---:|"]
        for (a, b), s in between.items():
            L.append(f"| {a} vs {b} | {fmt(s['median'])} | {fmt(s['p95'])} |")
        L.append("")
    if selection:
        members = sorted({m for sel in selection.values() for m in sel})
        L += ["## Which members the ensemble picked (share of frames, averaged "
              "over movies and joint groups)", "",
              "| subset | " + " | ".join(members) + " |",
              "|---|" + "---:|" * len(members)]
        for subset, sel in selection.items():
            L.append(f"| {subset} | " + " | ".join(
                fmt(sel.get(m), 2) if m in sel else "–" for m in members) + " |")
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


def make_figure(path, experiment, pooled, pairs):
    """Two panels, one per unit: 3D distance per joint group (mm) and angle
    difference (deg). Each pair: a dot at the median, a line up to the p95."""
    panels = [("3D distance to the 4-camera reference (mm)",
               [g for g in JOINT_GROUPS if g != "all"],
               lambda s, k: s["groups"][k]),
              ("Angle difference to the reference (degrees)", list(ANGLES),
               lambda s, k: s["angles"].get(k))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [4, 9]})
    width = 0.8 / max(len(pairs), 1)
    for ax, (title, keys, get) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for i, pair in enumerate(pairs[:len(SERIES_COLORS)]):
            color = SERIES_COLORS[i]
            for j, k in enumerate(keys):
                st = get(pooled[pair], k)
                if not st or np.isnan(st["median"]):
                    continue
                x = j + (i - (len(pairs) - 1) / 2) * width
                ax.plot([x, x], [st["median"], st["p95"]], color=color, lw=2,
                        solid_capstyle="round", zorder=2)
                ax.plot(x, st["median"], "o", ms=8, color=color, mec=SURFACE,
                        mew=2, zorder=3, label=pair if j == 0 else None)
                if j == 0:              # selective direct label: first group only
                    ax.annotate(pair, (x, st["p95"]), xytext=(0, 6),
                                textcoords="offset points", ha="center",
                                fontsize=8, color=INK_2, rotation=90, va="bottom")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right",
                   title="pair (dot = median, line to p95)", title_fontsize=9)
    fig.suptitle(f"Simulated 2-camera rig vs the 4-camera reference, {experiment}",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cut_dir", help="one experiment's cut subsets, e.g. "
                                    "inference_datasets/simulated/shalev/21to40")
    ap.add_argument("--predict-output", default=os.path.join(REPO_ROOT, "predict_output"))
    ap.add_argument("--out", help="default: comparison_data/sim_<experiment>")
    args = ap.parse_args()

    cut_dir = os.path.abspath(args.cut_dir)
    movies, experiment = find_movies(cut_dir)
    if not movies:
        sys.exit(f"no cut movies under {cut_dir}")
    out = args.out or os.path.join(REPO_ROOT, "comparison_data",
                                   run_name(experiment, "").rstrip("_"))

    runs, missing = {}, []
    for movie, subsets in sorted(movies.items()):
        for subset, movie_dir in sorted(subsets.items()):
            run = load_run(prediction_dir(args.predict_output, experiment,
                                          subset, movie_dir))
            if run is None:
                missing.append(f"{movie}/{subset}")
            else:
                runs[(movie, subset)] = run

    pairs = sorted({s for (_, s) in runs if s != REFERENCE_SUBSET})
    per_pair, per_movie, between_frames = {p: [] for p in pairs}, [], {}
    for movie in sorted(movies):
        ref = runs.get((movie, REFERENCE_SUBSET))
        if ref is None:
            continue
        here = [p for p in pairs if (movie, p) in runs]
        for pair in here:
            c = compare(runs[(movie, pair)], ref)
            per_pair[pair].append(c)
            s = summarize([c])
            per_movie.append({
                "movie": movie, "pair": pair, "n_frames": c["n_frames"],
                "all_median_mm": s["groups"]["all"]["median"],
                "all_p95_mm": s["groups"]["all"]["p95"],
                "far_pct": s["groups"]["all"]["far_pct"],
                **{f"{g} p95 mm": s["groups"][g]["p95"] for g in JOINT_GROUPS},
                **{f"{k} median deg": v["median"] for k, v in s["angles"].items()},
                **{f"{k} p95 deg": v["p95"] for k, v in s["angles"].items()},
                "swapped_frames": s["swapped_frames"], "nan_frames": c["nan_frames"],
                "ref_nan_frames": c["ref_nan_frames"],
                "rigidity_pair": runs[(movie, pair)]["rigidity"],
                "rigidity_ref": ref["rigidity"]})
        for i, a in enumerate(here):
            for b in here[i + 1:]:
                ia, ib = aligned(runs[(movie, a)], runs[(movie, b)])
                d = np.linalg.norm(runs[(movie, a)]["points"][ia]
                                   - runs[(movie, b)]["points"][ib], axis=-1) * 1000
                between_frames.setdefault((a, b), []).append(d)

    pairs = [p for p in pairs if per_pair[p]]
    if not pairs:
        sys.exit("no pair has both its own prediction and the reference's yet"
                 + (f" (missing: {', '.join(missing)})" if missing else ""))
    pooled = {p: summarize(per_pair[p]) for p in pairs}
    between = {k: stats(np.concatenate(v)) for k, v in between_frames.items()}
    selection = {}
    for subset in pairs + [REFERENCE_SUBSET]:
        sels = [r["selection"] for (m, s), r in runs.items()
                if s == subset and "selection" in r]
        if sels:
            members = sorted({k for sel in sels for k in sel})
            selection[subset] = {k: float(np.mean([sel.get(k, 0.0) for sel in sels]))
                                 for k in members}

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "per_movie.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_movie[0]))
        w.writeheader()
        w.writerows(per_movie)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"experiment": experiment, "reference": REFERENCE_SUBSET,
                   "pooled": pooled, "missing": missing,
                   "between_pairs": {f"{a} vs {b}": s for (a, b), s in between.items()},
                   "selection": selection}, f, indent=2)
    write_report(os.path.join(out, "report.md"), experiment, pooled, per_movie,
                 between, selection, missing, pairs)
    make_figure(os.path.join(out, "fig_pairs_vs_reference.png"), experiment,
                pooled, pairs)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}/ (per_movie.csv, summary.json, "
          f"report.md, fig_pairs_vs_reference.png): {len(per_movie)} (movie, "
          f"pair) comparisons"
          + (f"; not predicted yet: {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
