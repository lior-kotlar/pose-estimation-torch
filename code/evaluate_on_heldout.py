"""Score trained models on the TEST frames of the frozen split, per camera setting.

Every ensemble member since split_v1 trains on the same 139 frames and never on
the 41 test frames (training_datasets/*.split_v1.npz, frame_split == 2), so all
of them can be scored on those frames and compared point by point. This scores
several models in one run and writes one row per predicted point.

Three camera settings, since the question differs by rig:

  pair    a 2-camera rig: the bottom camera plus ONE side camera, once per side
          camera. 2D error on those two cameras, 3D from triangulating the two.
  all4    the current 4-camera rig: 2D error on every camera, 3D as the median
          of the 6 camera-pair triangulations (what the ensemble's candidates are).
  side3   the old 3-camera rig: the three side cameras.

Each setting scores every model that can run on that rig (which ones a
prediction ensemble then chooses is its models' "movie cameras"):

  2-camera (bottom + side) model   pair: on (bottom, s).  all4: once per side
                                   camera, bottom confmaps averaged over the runs
                                   -- exactly Predictor.predict_wing_bottom_pairs.
  4-camera model                   all4: on all 4. Nowhere else.
  3-camera model                   side3: on the side triad. Nowhere else.
  per-camera model                 every setting: one camera at a time.

Besides every model, two more things are scored:

  COMBINED_MEDIAN  per setting, the median of the 3D points of every model that
                   runs in it -- a time-free stand-in for the ensemble. The real
                   ensemble picks members by how steady their points stay over
                   31-219 neighbouring frames, which isolated test frames do not
                   have, so it can only be scored on labelled frames inside
                   whole movies.
  val vs test      each model's best-epoch validation error beside its test 2D
                   error in the setting it was built for (same metric: mean
                   distance between predicted and labelled confmap peaks). An
                   informal ball-park only: the validation frames also picked
                   the checkpoint, and there are 21 of them.

A deployed member's prediction_models/<name> folder can be given as well, to
compare the ensemble in use with the new models. It is scored on the same frames
but predates the split and trained on most of them (optimistic), and gets its
own stand-in, COMBINED_MEDIAN_DEPLOYED.

Writes points.csv (one row per predicted point), summary.json, report.md and
one figure per rig (median and 95th-percentile 3D error per model) plus
fig_error_tails.png (how often the best model of each family errs by more than
a given distance).

The input is built once, by the real Preprocessor with all 4 cameras in one
sample, and any camera's 4 input channels / 10 confmap channels are sliced out
of it, so the eval input cannot drift from the training input.

    sbatch -J eval_heldout -p salmon,dogfish,catfish --gres=gpu:1 --time=4:00:00 \\
        sbatch_files/sbatch_configurable.sh code/evaluate_on_heldout.py \\
        --models "train_output/debug_outputs/<run>" ... --out comparison_data/heldout_2cam

Needs a batch node's memory (the Preprocessor loads the whole labelled set).
"""
import argparse
import csv
import itertools
import json
import os
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "training_code"))

from utils import TrainConfig, torch_find_peaks, find_bottom_camera   # noqa: E402
import Preprocessor as PreprocessorModule                            # noqa: E402

# The per-wing channel order: 7 wing edge points, the wing joint, then tail and
# head (the two body points every wing sample carries).
JOINT_NAMES = [f"LE{i}" for i in range(1, 8)] + ["wing_joint", "tail", "head"]
POINTS_PER_CAM = len(JOINT_NAMES)
IN_CHANNELS_PER_CAM = 4          # 3 time channels + this wing's mask
IMAGE_HEIGHT = 800               # the rig's full frame; validated by reproject_check
TEST = 2                         # make_heldout_split's label for a test frame


# --------------------------------------------------------------------------
# geometry (from the side-only experiment's scorer, where it was verified:
# GT 3D reprojects onto the GT labels at 0.25 px)
# --------------------------------------------------------------------------
def load_geometry(data_path):
    """DLT projection matrices, lab rotation, cropzone, GT 3D points and 2D labels.
    `cameras_dlt_array` is stored (col, row, cam)."""
    with h5py.File(data_path, "r") as f:
        d = f["cameras_dlt_array"][:]
        P = np.stack([np.stack([d[:, i, c] for i in range(3)], axis=0)
                      for c in range(d.shape[2])], axis=0)      # (cams, 3, 4)
        R = f["rotation_matrix"][:]
        cropzone = f["cropZone"][:].astype(float)               # (frames, cams, 2)
        points_3D = np.transpose(f["points_3D"][:], [1, 2, 0])  # (frames, 18, 3)
        joints = f["joints"][:]                                 # (2, 18, cams, frames)
        centers = f["camera_centers"][:]
    return P, R, cropzone, points_3D, joints, centers


def uncrop(xy, cropzone_fc):
    """Crop-space (x, y) -> full-frame calibration coords (Triangulator.uncrop)."""
    x = cropzone_fc[1] + xy[..., 0]
    y = cropzone_fc[0] + xy[..., 1]
    return np.stack([x, IMAGE_HEIGHT + 1 - y], axis=-1)


def triangulate_dlt(uv, Psub):
    """Linear DLT triangulation of one point from n >= 2 views."""
    A = np.stack([uv[i, 0] * Psub[i, 2] - Psub[i, 0] for i in range(len(uv))] +
                 [uv[i, 1] * Psub[i, 2] - Psub[i, 1] for i in range(len(uv))])
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3]


def reproject_check(P, R, cropzone, points_3D, joints, frames):
    """Fail loudly if the 2D<->3D convention does not reproduce ground truth."""
    errs = []
    for fr in frames:
        Xc = (R.T @ points_3D[fr].T).T
        Xh = np.concatenate([Xc, np.ones((len(Xc), 1))], axis=1)
        for c in range(P.shape[0]):
            p = (P[c] @ Xh.T).T
            uv = p[:, :2] / p[:, 2:3]
            gt = uncrop(joints[:, :, c, fr].T, cropzone[fr, c])
            errs.append(np.linalg.norm(uv - gt, axis=1))
    med = float(np.median(np.concatenate(errs)))
    if med > 2.0:
        raise SystemExit(f"2D<->3D convention check failed: GT 3D reprojects {med:.1f} px "
                         f"from the GT 2D labels. Refusing to report 3D error.")
    return med


def gt_joint(block, j):
    """(GT joint index, name) for channel j of a wing block.

    Preprocessor.split_per_wing crosses the wings: its "left" block carries GT
    joints 8..15 (the anatomical right wing) and its "right" block 0..7. Tail is
    GT 16, head 17."""
    side = "right" if block == "left" else "left"
    if j < 8:
        return (8 if block == "left" else 0) + j, f"{side}_{JOINT_NAMES[j]}"
    return (16 if JOINT_NAMES[j] == "tail" else 17), f"{side}_{JOINT_NAMES[j]}"


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------
def build_all_camera_samples(template_config_path, scratch_dir):
    """Preprocess the labelled set as a 4-camera ALL_CAMS_PER_WING model would:
    one sample per (wing block, frame) with every camera in it, cameras in
    index order. Returns box (B, 4*4, H, W), confmaps (B, 4*10, H, W), the
    sample's frame and wing block."""
    with open(template_config_path) as f:
        cfg = json.load(f)
    cfg["model type"] = "ALL_CAMS_PER_WING"
    cfg.pop("number of cameras", None)
    cfg.pop("required camera", None)
    os.makedirs(scratch_dir, exist_ok=True)
    tmp = os.path.join(scratch_dir, "eval_config.json")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    pp = PreprocessorModule.Preprocessor(TrainConfig(config_path=tmp))
    pp.do_preprocess()
    groups = np.asarray(pp.sample_group_ids)
    n = pp.num_frames
    block = np.where(np.arange(len(groups)) < n, "left", "right")
    return pp.box, pp.confmaps, groups, block, pp.num_cams


def cam_input(box, c):
    return box[:, c * IN_CHANNELS_PER_CAM:(c + 1) * IN_CHANNELS_PER_CAM]


def cam_confmaps(confmaps, c):
    return confmaps[:, c * POINTS_PER_CAM:(c + 1) * POINTS_PER_CAM]


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def describe_model(model_dir):
    """A training run folder, or a deployed member's prediction_models/<name>
    folder -- then the weights are its own and the config its source run's."""
    deployed = os.path.isfile(os.path.join(model_dir, "model.json"))
    if deployed:
        with open(os.path.join(model_dir, "model.json")) as f:
            run_dir = json.load(f)["source"]
    else:
        run_dir = model_dir
    with open(os.path.join(run_dir, "configuration.json")) as f:
        cfg = json.load(f)
    if cfg["model type"].startswith("MODEL_PER_CAM"):
        kind = "per_cam"
    elif cfg.get("required camera") == "bottom":
        kind = f"bottom{cfg['number of cameras']}"
    else:
        kind = f"all{cfg.get('number of cameras') or 4}"
    # The run folder, not the run tag: two families can share a tag (DIL3_HELDOUT).
    name = os.path.basename(run_dir.rstrip("/")).replace("ALL_CAMS_PER_WING_", "all_cams_") \
        .replace("MODEL_PER_CAM_PER_WING_", "per_cam_")
    name = re.sub(r"_[A-Z][a-z]{2} \d{2}(_\d+)?$", "", name)      # the run's date
    if deployed:
        name = "deployed_" + os.path.basename(model_dir.rstrip("/"))
    return {"name": name, "deployed": deployed,
            "kind": kind, "dir": run_dir, "weights": os.path.join(model_dir, "best_model.pt"),
            "data path": cfg["data path"], "split file": cfg.get("split file"),
            "val_px": None if deployed else best_epoch_val_px(run_dir)}


def best_epoch_val_px(run_dir):
    """The validation pixel error at the epoch that became best_model.pt."""
    try:
        with open(os.path.join(run_dir, "best_model_info.txt")) as f:
            epoch = int(f.readline().split(":")[1])
        with open(os.path.join(run_dir, "history.csv")) as f:
            rows = list(csv.DictReader(f))
        return float(rows[epoch - 1]["val l2"])
    except (OSError, ValueError, KeyError, IndexError):
        return None


# The setting a model kind was built for, where its test error is the fair
# counterpart of its validation error.
NATIVE_SETTINGS = {"per_cam": ("all4",), "all4": ("all4",), "all3": ("side3",),
                   "bottom2": ("pair",)}


class Runner:
    def __init__(self, weights, device, batch_size):
        self.model = torch.jit.load(weights, map_location=device).to(device).eval()
        self.device, self.batch_size = device, batch_size

    def confmaps(self, x):
        out = []
        with torch.no_grad():
            for s in range(0, len(x), self.batch_size):
                t = torch.tensor(x[s:s + self.batch_size], dtype=torch.float32).to(self.device)
                out.append(self.model(t).cpu().numpy())
        return np.concatenate(out, axis=0)


def peaks(confmaps):
    """(B, C, H, W) -> (B, C, 2) crop-pixel (x, y)."""
    return np.transpose(torch_find_peaks(confmaps)[:, :2, :], (0, 2, 1))


def predict_cameras(model, kind, box, cams, bottom, num_cams):
    """{camera: (B, 10, 2) peaks} for the cameras of this setting, or None when
    this kind of model is not run in it."""
    if kind == "per_cam":
        return {c: peaks(model.confmaps(cam_input(box, c))) for c in cams}
    if kind == f"all{num_cams}":
        if len(cams) != num_cams:
            return None     # it needs every camera: a 2- or 3-camera rig cannot run it
        cm = model.confmaps(box)
        return {c: peaks(cam_confmaps(cm, c)) for c in cams}
    if kind == "all3":
        if len(cams) != 3:
            return None
        cm = model.confmaps(np.concatenate([cam_input(box, c) for c in cams], axis=1))
        return {c: peaks(cam_confmaps(cm, k)) for k, c in enumerate(cams)}
    if kind == "bottom2":
        if bottom not in cams or len(cams) not in (2, num_cams):
            return None     # a 2-camera rig, or the full rig once per pair; not the side triad
        sides = [c for c in cams if c != bottom]
        out, bottom_sum = {}, 0.0
        for s in sides:
            cm = model.confmaps(np.concatenate([cam_input(box, bottom), cam_input(box, s)], axis=1))
            bottom_sum = bottom_sum + cam_confmaps(cm, 0)
            out[s] = peaks(cam_confmaps(cm, 1))
        out[bottom] = peaks(bottom_sum / len(sides))
        return out
    raise SystemExit(f"unknown model kind {kind}")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_3d(pred, cams, groups, block, P, R, cropzone, points_3D):
    """[(frame, joint name, error in calibration units)] and the (samples,
    points, 3) estimates. 2 cameras: their triangulation. More: the median of
    every pair's triangulation. Head and tail come once per wing block; each
    is scored separately."""
    pairs = list(itertools.combinations(cams, 2))
    X_all = np.zeros((len(groups), POINTS_PER_CAM, 3))
    for i, fr in enumerate(groups):
        for j in range(POINTS_PER_CAM):
            Xs = [triangulate_dlt(np.stack([uncrop(pred[c][i, j], cropzone[fr, c]) for c in pair]),
                                  P[list(pair)]) for pair in pairs]
            X_all[i, j] = np.median(np.stack(Xs), axis=0)
    return errors_3d(X_all, groups, block, R, points_3D), X_all


def errors_3d(X_all, groups, block, R, points_3D):
    rows = []
    for i, (fr, blk) in enumerate(zip(groups, block)):
        gt3 = (R.T @ points_3D[fr].T).T
        for j in range(POINTS_PER_CAM):
            gi, jname = gt_joint(blk, j)
            rows.append((int(fr), jname, float(np.linalg.norm(X_all[i, j] - gt3[gi]))))
    return rows


def settings(bottom, num_cams):
    sides = [c for c in range(num_cams) if c != bottom]
    out = [(f"pair_{bottom}{s}", (bottom, s)) for s in sides]
    out.append(("all4", tuple(range(num_cams))))
    out.append(("side3", tuple(sides)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="training run folders, or prediction_models/<name> folders")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    models = [describe_model(d) for d in args.models]
    # Deployed members predate the split: they are scored on the same frames,
    # but the frames, the input and the checks come from the split's models.
    split_models = [m for m in models if not m["deployed"]]
    if not split_models:
        raise SystemExit("at least one model must be trained on the split; it defines the test frames")
    for key in ("data path", "split file"):
        if len({m[key] for m in split_models}) != 1:
            raise SystemExit(f"the models disagree on '{key}'; they cannot share test frames")
    data_path, split_file = split_models[0]["data path"], split_models[0]["split file"]
    # A deployed run may record the project's old location (pose-estimation/,
    # before the move to pose-estimation-torch/), so it need only name the same file.
    for m in models:
        if m["deployed"] and os.path.basename(m["data path"]) != os.path.basename(data_path):
            raise SystemExit(f"{m['name']} was trained on {m['data path']}, not {data_path}")
    test_frames = np.flatnonzero(np.load(split_file)["frame_split"] == TEST)

    P, R, cropzone, points_3D, joints, centers = load_geometry(data_path)
    bottom = find_bottom_camera(centers)
    if bottom is None:
        raise SystemExit(f"no bottom camera in the camera positions of {data_path}")
    convention_px = reproject_check(P, R, cropzone, points_3D, joints, test_frames)

    box, confmaps, groups, block, num_cams = build_all_camera_samples(
        os.path.join(split_models[0]["dir"], "configuration.json"), os.path.join(args.out, "_scratch"))
    keep = np.isin(groups, test_frames)
    box, confmaps, groups, block = box[keep], confmaps[keep], groups[keep], block[keep]
    gt_peaks = {c: peaks(cam_confmaps(confmaps, c)) for c in range(num_cams)}
    print(f"{len(test_frames)} test frames, {len(groups)} wing samples, bottom camera {bottom}, "
          f"convention check {convention_px:.2f} px", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    point_rows, summary = [], []
    # The ground-truth confmap peaks, scored like a model: the floor no model can
    # beat once blob quantisation and label noise are counted.
    entries = [{"name": "GT_PEAK_FLOOR", "kind": "gt"}] + models
    points_by_setting = {}      # (combined name, setting) -> {model name: (samples, points, 3)}
    pair_errors = {}            # (model, kind) -> ([2D errors], [3D errors]) over the pairs
    errors_by_rig = {}          # setting or "pair" -> {model name: (kind, 3D errors mm)}
    for m in entries:
        runner = None if m["kind"] == "gt" else Runner(m["weights"],
                                                     device, args.batch_size)
        for setting, cams in settings(bottom, num_cams):
            if m["kind"] == "gt":
                pred = {c: gt_peaks[c] for c in cams}
            else:
                pred = predict_cameras(runner, m["kind"], box, cams, bottom, num_cams)
            if pred is None:
                continue
            err2d = np.concatenate([np.linalg.norm(pred[c] - gt_peaks[c], axis=2).ravel() for c in cams])
            for c in cams:
                e = np.linalg.norm(pred[c] - gt_peaks[c], axis=2)
                for i in range(len(groups)):
                    for j in range(POINTS_PER_CAM):
                        point_rows.append([m["name"], m["kind"], setting, int(groups[i]), block[i],
                                           c, gt_joint(block[i], j)[1], f"{e[i, j]:.4f}", ""])
            e3, X_all = score_3d(pred, cams, groups, block, P, R, cropzone, points_3D)
            if m["kind"] != "gt":
                combined = "COMBINED_MEDIAN_DEPLOYED" if m["deployed"] else "COMBINED_MEDIAN"
                points_by_setting.setdefault((combined, setting), {})[m["name"]] = X_all
            for fr, jname, d in e3:
                point_rows.append([m["name"], m["kind"], setting, fr, "-", -1, jname, "", f"{d * 1000:.5f}"])
            d3 = np.array([d for _, _, d in e3]) * 1000
            summary.append(stats_row(m["name"], m["kind"], setting, cams, err2d, d3))
            errors_by_rig.setdefault(setting, {})[m["name"]] = (m["kind"], d3)
            if setting.startswith("pair_"):
                acc = pair_errors.setdefault((m["name"], m["kind"]), ([], []))
                acc[0].append(err2d); acc[1].append(d3)
            s = summary[-1]
            print(f"{m['name']:>24} {setting:>7}  2D mean {s['mean_px']:6.2f} px  median "
                  f"{s['median_px']:5.2f}  p95 {s['p95_px']:6.2f}   3D median "
                  f"{s['median_3d_mm']:.4f} mm  p95 {s['p95_3d_mm']:.4f}", flush=True)

    # (c) the time-free stand-in for the ensemble, per setting -- one for the
    # models trained on the split and one for the deployed members, if any
    for (combined, setting), by_model in points_by_setting.items():
        if len(by_model) < 2:
            continue
        X = np.median(np.stack(list(by_model.values())), axis=0)
        e3 = errors_3d(X, groups, block, R, points_3D)
        for fr, jname, d in e3:
            point_rows.append([combined, "combined", setting, fr, "-", -1, jname, "",
                               f"{d * 1000:.5f}"])
        d3 = np.array([d for _, _, d in e3]) * 1000
        summary.append(stats_row(combined, "combined", setting,
                                 dict(settings(bottom, num_cams))[setting], None, d3,
                                 members=sorted(by_model)))
        errors_by_rig.setdefault(setting, {})[combined] = ("combined", d3)
        if setting.startswith("pair_"):
            pair_errors.setdefault((combined, "combined"), ([], []))[1].append(d3)
        print(f"{combined:>24} {setting:>7}  3D median {summary[-1]['median_3d_mm']:.4f} mm "
              f"over {len(by_model)} models", flush=True)

    # The two-camera rig as one row per model: the pairs' points pooled.
    for (name, kind), (e2, e3s) in pair_errors.items():
        summary.append(stats_row(name, kind, "pair", (), np.concatenate(e2) if e2 else None,
                                 np.concatenate(e3s)))
        errors_by_rig.setdefault("pair", {})[name] = (kind, np.concatenate(e3s))

    with open(os.path.join(args.out, "points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind", "setting", "frame", "wing_block", "cam", "joint",
                    "err_px", "err_3d_mm"])
        w.writerows(point_rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"data path": data_path, "split file": split_file,
                   "test frames": test_frames.tolist(), "bottom camera": int(bottom),
                   "convention check px": convention_px,
                   "models": {m["name"]: os.path.dirname(m["weights"]) for m in models},
                   "val px (best epoch)": {m["name"]: m["val_px"] for m in models},
                   "results": summary}, f, indent=2)
    write_report(os.path.join(args.out, "report.md"), summary, models, len(test_frames))
    write_figures(args.out, errors_by_rig, len(test_frames), {m["name"] for m in models if m["deployed"]})
    print(f"wrote {args.out}/points.csv, summary.json, report.md and the figures")


def stats_row(name, kind, setting, cams, err2d, d3, **extra):
    """One summary row. err2d None = no 2D prediction of its own (the combined row)."""
    row = {"model": name, "kind": kind, "setting": setting, "cams": [int(c) for c in cams],
           "mean_px": None, "median_px": None, "p95_px": None, "frac_over_10px": None,
           "mean_3d_mm": float(d3.mean()), "median_3d_mm": float(np.median(d3)),
           "p95_3d_mm": float(np.percentile(d3, 95)), **extra}
    if err2d is not None:
        row.update(mean_px=float(err2d.mean()), median_px=float(np.median(err2d)),
                   p95_px=float(np.percentile(err2d, 95)),
                   frac_over_10px=float((err2d > 10).mean()))
    return row


def _fmt(v, digits):
    return "–" if v is None else f"{v:.{digits}f}"


RIGS = {"pair": "Two-camera rig: bottom + one side camera (pooled over the pairs)",
        "all4": "Four-camera rig", "side3": "Side triad (old 3-camera rig)"}
RIG_FIGURES = {"pair": "fig_two_camera_rig.png", "all4": "fig_four_camera_rig.png",
               "side3": "fig_side_triad.png"}
FAMILIES = {"per_cam": ("per-camera", "#e8683a"), "all4": ("4-camera", "#1cae7a"),
            "all3": ("3-camera", "#8a5cc7"), "bottom2": ("2-camera (bottom + side)", "#2a78d6"),
            "combined": ("combined median", "#888888")}


def write_figures(out, errors_by_rig, n_test_frames, deployed=()):
    """Per rig, every model's median and 95th-percentile 3D error as bars,
    coloured by family, with the GT-peak floor as a dashed line; and one figure
    of error tails: for the best model of each family, the fraction of points
    whose 3D error exceeds each distance, beside the ensemble stand-ins.
    Deployed members (hatched) trained on most of the test frames."""
    deployed = set(deployed) | {"COMBINED_MEDIAN_DEPLOYED"}
    rigs = [r for r in RIGS if r in errors_by_rig]
    for rig in rigs:
        by_model = errors_by_rig[rig]
        floor = by_model.get("GT_PEAK_FLOOR")
        rows = sorted(((n, k, e) for n, (k, e) in by_model.items() if k != "gt"),
                      key=lambda r: np.median(r[2]))
        fig, axes = plt.subplots(1, 2, figsize=(13, 1.6 + 0.5 * len(rows)), sharey=True)
        y = np.arange(len(rows))[::-1]
        for ax, stat, label in ((axes[0], np.median, "median 3D error (mm)"),
                                (axes[1], lambda e: np.percentile(e, 95), "95th-percentile 3D error (mm)")):
            vals = [stat(e) for _, _, e in rows]
            ax.barh(y, vals, color=[FAMILIES[k][1] for _, k, _ in rows], height=0.65,
                    hatch=["//" if n in deployed else "" for n, _, _ in rows], edgecolor="white")
            for yi, v in zip(y, vals):
                ax.text(v, yi, f" {v:.3f}", va="center", fontsize=9)
            if floor is not None:
                ax.axvline(stat(floor[1]), color="#333333", ls="--", lw=1)
                ax.text(stat(floor[1]), y[0] + 0.6, " GT-peak floor", fontsize=8, color="#333333")
            ax.set_xlabel(label)
            ax.set_xlim(0, max(vals) * 1.18)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_yticks(y, [n for n, _, _ in rows])
        kinds = list(dict.fromkeys(k for _, k, _ in rows))
        handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILIES[k][1]) for k in kinds]
        labels = [FAMILIES[k][0] for k in kinds]
        if any(n in deployed for n, _, _ in rows):
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", hatch="//", edgecolor="white"))
            labels.append("deployed (trained on most test frames)")
        fig.legend(handles=handles, labels=labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.97), frameon=False)
        fig.suptitle(f"{RIGS[rig]}, {n_test_frames} test frames", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(out, RIG_FIGURES[rig]), dpi=120)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(rigs), figsize=(6 * len(rigs), 4.5), sharey=True, squeeze=False)
    thresholds = np.linspace(0, 2, 401)
    for ax, rig in zip(axes[0], rigs):
        best = {}
        for name, (kind, e) in errors_by_rig[rig].items():
            if kind not in ("gt", "combined") and name not in deployed and \
                    (kind not in best or np.median(e) < np.median(best[kind][1])):
                best[kind] = (name, e)
        for kind, (name, e) in best.items():
            ax.plot(thresholds, [(e > t).mean() for t in thresholds], color=FAMILIES[kind][1], lw=2,
                    label=f"{name} ({FAMILIES[kind][0]})")
        for name, ls in (("COMBINED_MEDIAN", "-"), ("COMBINED_MEDIAN_DEPLOYED", ":")):
            if name in errors_by_rig[rig]:
                e = errors_by_rig[rig][name][1]
                ax.plot(thresholds, [(e > t).mean() for t in thresholds], color="#555555", ls=ls,
                        lw=2, label=name)
        if "GT_PEAK_FLOOR" in errors_by_rig[rig]:
            e = errors_by_rig[rig]["GT_PEAK_FLOOR"][1]
            ax.plot(thresholds, [(e > t).mean() for t in thresholds], color="#333333", ls="--",
                    lw=1, label="GT-peak floor")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.05)
        ax.set_title(RIGS[rig].split(":")[0].split(" (")[0])
        ax.set_xlabel("3D error threshold (mm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("fraction of points with a larger error")
    fig.suptitle("How often the best model of each family makes a big 3D error")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_error_tails.png"), dpi=120)
    plt.close(fig)


def write_report(path, summary, models, n_test_frames):
    """report.md: one table per camera setting, sorted by 3D median, then the
    informal validation-vs-test table."""
    rig = RIGS
    lines = [f"# Held-out test frames ({n_test_frames} frames)", "",
             "COMBINED_MEDIAN is the median of the 3D points of every model in the table: a "
             "time-free stand-in for the ensemble, whose selector needs neighbouring frames.", ""]
    if any(m["deployed"] for m in models):
        lines += ["deployed_* are the members in prediction_models/, and COMBINED_MEDIAN_DEPLOYED "
                  "their stand-in (COMBINED_MEDIAN then covers only the models trained on the "
                  "split). They predate the split and trained on most of these test frames, so "
                  "their scores are optimistic: a model that beats them has beaten them for sure, "
                  "one that loses to them may not really be worse.", ""]
    for key, title in rig.items():
        rows = [r for r in summary if r["setting"] == key]
        if not rows:
            continue
        lines += [f"## {title}", "",
                  "| model | 2D mean px | 2D median px | 2D p95 px | 3D median mm | 3D mean mm | 3D p95 mm |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in sorted(rows, key=lambda r: r["median_3d_mm"]):
            lines.append(f"| {r['model']} | {_fmt(r['mean_px'], 2)} | {_fmt(r['median_px'], 2)} | "
                         f"{_fmt(r['p95_px'], 2)} | {r['median_3d_mm']:.3f} | {r['mean_3d_mm']:.3f} | "
                         f"{r['p95_3d_mm']:.3f} |")
        lines.append("")
    lines += ["## Validation vs test (informal)", "",
              "Best-epoch validation error beside the test 2D error in the setting the model was "
              "built for. Ball-park only: the 21 validation frames also chose the checkpoint.", "",
              "| model | val px | test px | test / val |", "|---|---:|---:|---:|"]
    for m in (m for m in models if not m["deployed"]):     # a deployed run's validation leaked
        native = NATIVE_SETTINGS[m["kind"]]
        rows = [r for r in summary if r["model"] == m["name"] and r["setting"] in native]
        test = rows[0]["mean_px"] if rows else None
        ratio = test / m["val_px"] if test and m["val_px"] else None
        lines.append(f"| {m['name']} | {_fmt(m['val_px'], 2)} | {_fmt(test, 2)} | {_fmt(ratio, 2)} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
