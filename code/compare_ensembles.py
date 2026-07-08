#!/usr/bin/env python
"""
Compare two ensemble prediction run directories on the SAME movie.

Typical use: baseline = the earlier 2-model ensemble run dir, new = the 9-model
ensemble run dir. Both must contain the standard ensemble outputs:
    points_3D_ensemble_best_method.npy           (raw fused 3D)
    points_3D_smoothed_ensemble_best_method.npy  (smoothed fused 3D)

Because there is NO ground truth, comparison relies on self-consistency
metrics. They fall in two groups:

  Optimized-by-the-selector (report, but read with care):
    * rigidity_std  -- avg across-frame std of wing edge lengths
                       (== the pipeline's get_validation_score, in README).
                       The selector explicitly minimizes this, so a larger
                       ensemble can win it almost by construction. Necessary,
                       not sufficient.

  Independent of the selector (the real tie-breakers):
    * acceleration  -- mean ||p[t+1]-2p[t]+p[t-1]|| per joint (trajectory jitter)
    * raw_vs_smoothed_residual -- how much smoothing had to move the raw 3D
    * edge_length_plausibility -- mean wing edge lengths (should not drift)

Plus:
    * agreement -- per-joint mean 3D distance between the two runs (localises
                   where they disagree; does not say which is better)
    * model_selection -- for the NEW run, how often each member was chosen
                         (read from ensemble_model_selection_summary.json)

Usage:
    python compare_ensembles.py <baseline_run_dir> <new_run_dir> \
        [--out OUTDIR] [--labels BASELINE_LABEL NEW_LABEL]

Distances are in the 3D coordinate units of the pipeline (typically mm).
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_code_lior"))
from Validation import Validation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_NAME = "points_3D_ensemble_best_method.npy"
SMOOTHED_NAME = "points_3D_smoothed_ensemble_best_method.npy"
WING_EDGE_LABELS = ["L0-1", "L1-2", "L2-3", "L3-4", "L4-5", "L5-6", "L0-6",
                    "R0-1", "R1-2", "R2-3", "R3-4", "R4-5", "R5-6", "R0-6", "side"]
JOINT_LABELS = [f"L{i}" for i in range(7)] + ["Lside"] + \
               [f"R{i}" for i in range(7)] + ["Rside", "tail", "head"]


def load_points(run_dir):
    raw_p = os.path.join(run_dir, RAW_NAME)
    smo_p = os.path.join(run_dir, SMOOTHED_NAME)
    raw = np.load(raw_p) if os.path.isfile(raw_p) else None
    smoothed = np.load(smo_p) if os.path.isfile(smo_p) else None
    if smoothed is None and raw is None:
        raise FileNotFoundError(f"No ensemble points found in {run_dir}")
    return raw, smoothed


def rigidity(points):
    """avg across-frame std of edge lengths, plus per-edge std and mean length."""
    avg_std, _, distances_std, mean_distance, _ = Validation.get_wings_distances_variance(points)
    return float(avg_std), np.asarray(distances_std), np.asarray(mean_distance)


def mean_acceleration(points):
    """Per-joint mean magnitude of the discrete 2nd time-derivative (jitter)."""
    acc = points[2:] - 2 * points[1:-1] + points[:-2]          # (frames-2, joints, 3)
    per_joint = np.linalg.norm(acc, axis=2).mean(axis=0)       # (joints,)
    return per_joint


def raw_vs_smoothed_residual(raw, smoothed):
    """Per-joint RMS distance between raw and smoothed 3D (raw noise level)."""
    n = min(len(raw), len(smoothed))
    d = np.linalg.norm(raw[:n] - smoothed[:n], axis=2)         # (n, joints)
    return np.sqrt((d ** 2).mean(axis=0))                     # (joints,)


def agreement(points_a, points_b):
    """Per-joint mean 3D distance between the two runs' (smoothed) points."""
    n = min(len(points_a), len(points_b))
    if len(points_a) != len(points_b):
        print(f"  [warn] frame counts differ ({len(points_a)} vs {len(points_b)}); "
              f"comparing first {n} frames for agreement.")
    d = np.linalg.norm(points_a[:n] - points_b[:n], axis=2)    # (n, joints)
    return d.mean(axis=0), n                                   # (joints,)


def summarize_run(run_dir, label):
    raw, smoothed = load_points(run_dir)
    out = {"run_dir": run_dir, "label": label,
           "num_frames": int(len(smoothed if smoothed is not None else raw))}
    if smoothed is not None:
        avg_std, edge_std, edge_len = rigidity(smoothed)
        out["rigidity_std_smoothed"] = avg_std
        out["edge_std_smoothed"] = edge_std.tolist()
        out["edge_length_mean"] = edge_len.tolist()
        out["acceleration_per_joint_smoothed"] = mean_acceleration(smoothed).tolist()
        out["mean_acceleration_smoothed"] = float(np.mean(out["acceleration_per_joint_smoothed"]))
    if raw is not None:
        out["rigidity_std_raw"] = rigidity(raw)[0]
    if raw is not None and smoothed is not None:
        resid = raw_vs_smoothed_residual(raw, smoothed)
        out["raw_vs_smoothed_residual_per_joint"] = resid.tolist()
        out["mean_raw_vs_smoothed_residual"] = float(resid.mean())
    return out, raw, smoothed


def read_model_selection(run_dir):
    p = os.path.join(run_dir, "ensemble_model_selection_summary.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def pct_change(base, new):
    if base == 0:
        return float("nan")
    return 100.0 * (new - base) / base


def make_plots(base_sum, new_sum, base_smoothed, new_smoothed, out_dir, labels):
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(len(WING_EDGE_LABELS))

    # per-edge rigidity std
    if "edge_std_smoothed" in base_sum and "edge_std_smoothed" in new_sum:
        fig, ax = plt.subplots(figsize=(13, 5))
        w = 0.4
        ax.bar(x - w / 2, base_sum["edge_std_smoothed"], w, label=labels[0])
        ax.bar(x + w / 2, new_sum["edge_std_smoothed"], w, label=labels[1])
        ax.set_xticks(x); ax.set_xticklabels(WING_EDGE_LABELS, rotation=45, ha="right")
        ax.set_ylabel("across-frame std of edge length")
        ax.set_title("Wing-edge rigidity (lower = better)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "rigidity_per_edge.png"), dpi=120)
        plt.close(fig)

    # per-joint acceleration
    if "acceleration_per_joint_smoothed" in base_sum and "acceleration_per_joint_smoothed" in new_sum:
        xj = np.arange(len(JOINT_LABELS))
        fig, ax = plt.subplots(figsize=(13, 5))
        w = 0.4
        ax.bar(xj - w / 2, base_sum["acceleration_per_joint_smoothed"], w, label=labels[0])
        ax.bar(xj + w / 2, new_sum["acceleration_per_joint_smoothed"], w, label=labels[1])
        ax.set_xticks(xj); ax.set_xticklabels(JOINT_LABELS, rotation=45, ha="right")
        ax.set_ylabel("mean |2nd derivative| (jitter)")
        ax.set_title("Per-joint trajectory jitter (lower = smoother)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "acceleration_per_joint.png"), dpi=120)
        plt.close(fig)

    # per-joint agreement
    if base_smoothed is not None and new_smoothed is not None:
        agree, _ = agreement(base_smoothed, new_smoothed)
        xj = np.arange(len(JOINT_LABELS))
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(xj, agree, color="tab:purple")
        ax.set_xticks(xj); ax.set_xticklabels(JOINT_LABELS, rotation=45, ha="right")
        ax.set_ylabel("mean 3D distance between runs")
        ax.set_title(f"Where the two ensembles disagree ({labels[0]} vs {labels[1]})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "agreement_per_joint.png"), dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compare two ensemble run directories.")
    ap.add_argument("baseline_dir")
    ap.add_argument("new_dir")
    ap.add_argument("--out", default=None, help="output dir (default: <new_dir>/comparison_vs_baseline)")
    ap.add_argument("--labels", nargs=2, default=["baseline", "new"], metavar=("BASE", "NEW"))
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.new_dir, "comparison_vs_baseline")
    labels = args.labels

    base_sum, base_raw, base_smoothed = summarize_run(args.baseline_dir, labels[0])
    new_sum, new_raw, new_smoothed = summarize_run(args.new_dir, labels[1])

    agree_vec = None
    if base_smoothed is not None and new_smoothed is not None:
        agree_vec, n_used = agreement(base_smoothed, new_smoothed)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit(f"Ensemble comparison   {labels[0]}  vs  {labels[1]}")
    emit("=" * 78)
    emit(f"{labels[0]}: {args.baseline_dir}  ({base_sum['num_frames']} frames)")
    emit(f"{labels[1]}: {args.new_dir}  ({new_sum['num_frames']} frames)")
    emit("")
    emit("-- PRIMARY (optimized by selector: lower is better, but see caveat) --")
    for key, title in [("rigidity_std_smoothed", "rigidity std (smoothed)"),
                       ("rigidity_std_raw", "rigidity std (raw)")]:
        if key in base_sum and key in new_sum:
            b, nw = base_sum[key], new_sum[key]
            emit(f"  {title:<26} {labels[0]}={b:.6e}   {labels[1]}={nw:.6e}   "
                 f"change={pct_change(b, nw):+.1f}%")
    emit("  [caveat] the selector MINIMIZES rigidity std, and the larger ensemble")
    emit("           has more candidates, so it tends to win this almost by design.")
    emit("")
    emit("-- INDEPENDENT tie-breakers (not optimized by the selector) --")
    for key, title in [("mean_acceleration_smoothed", "mean jitter (accel, smoothed)"),
                       ("mean_raw_vs_smoothed_residual", "raw-vs-smoothed residual")]:
        if key in base_sum and key in new_sum:
            b, nw = base_sum[key], new_sum[key]
            emit(f"  {title:<30} {labels[0]}={b:.5f}   {labels[1]}={nw:.5f}   "
                 f"change={pct_change(b, nw):+.1f}%")
    emit("")
    emit("-- EDGE-LENGTH plausibility (mean wing edge lengths; should be stable) --")
    if "edge_length_mean" in base_sum and "edge_length_mean" in new_sum:
        bl = np.array(base_sum["edge_length_mean"])
        nl = np.array(new_sum["edge_length_mean"])
        emit(f"  mean |length change| across {len(bl)} edges: {np.abs(nl - bl).mean():.5f}")
        emit(f"  max  |length change|                       : {np.abs(nl - bl).max():.5f}")
    emit("")
    if agree_vec is not None:
        emit("-- AGREEMENT (per-joint mean 3D distance between the two runs) --")
        emit(f"  overall mean: {agree_vec.mean():.5f}   max: {agree_vec.max():.5f} "
             f"(joint {JOINT_LABELS[int(np.argmax(agree_vec))]})")
        order = np.argsort(agree_vec)[::-1][:5]
        emit("  top-5 most-divergent joints: " +
             ", ".join(f"{JOINT_LABELS[j]}={agree_vec[j]:.4f}" for j in order))
        emit("")

    sel = read_model_selection(args.new_dir)
    if sel and "overall" in sel:
        emit(f"-- MODEL SELECTION in {labels[1]} (fraction of frames each member was chosen) --")
        ov = sel["overall"]
        for name, stats in sorted(ov.items(), key=lambda kv: -kv[1].get("fraction_of_frames_selected", 0)):
            emit(f"  {stats.get('fraction_of_frames_selected', float('nan')):.3f}  {name}")
        never = [n for n, s in ov.items() if s.get("fraction_of_frames_selected", 0) == 0]
        if never:
            emit(f"  [note] never selected (candidates to drop): {', '.join(never)}")
        emit("")

    emit("Interpretation: prefer the run that is at least as good on the INDEPENDENT")
    emit("metrics (jitter, residual) and plausible on edge lengths. If the 9-model")
    emit("run only wins on rigidity std but not on jitter, the gain may be illusory.")
    emit("=" * 78)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "compare_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "compare_summary.json"), "w") as f:
        payload = {"baseline": base_sum, "new": new_sum}
        if agree_vec is not None:
            payload["agreement_per_joint"] = agree_vec.tolist()
            payload["joint_labels"] = JOINT_LABELS
        json.dump(payload, f, indent=2)
    make_plots(base_sum, new_sum, base_smoothed, new_smoothed, out_dir, labels)
    print(f"\nWrote comparison to {out_dir}")


if __name__ == "__main__":
    main()
