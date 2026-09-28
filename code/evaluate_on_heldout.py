"""Score trained models on the TEST frames of the frozen split, per camera setting.

Every ensemble member since split_v1 trains on the same 139 frames and never on
the 41 test frames (training_datasets/*.split_v1.npz, frame_split == 2), so all
of them can be scored on those frames and compared point by point. This scores
several models in one run and writes one row per predicted point.

Two camera settings, since the question differs by rig:

  pair    a 2-camera rig: the bottom camera plus ONE side camera, once per side
          camera. 2D error on those two cameras, 3D from triangulating the two.
  all4    the current 4-camera rig: 2D error on every camera, 3D as the median
          of the 6 camera-pair triangulations (what the ensemble's candidates are).

How each kind of model is run in each setting:

  2-camera (bottom + side) model   pair: on (bottom, s).  all4: once per side
                                   camera, bottom confmaps averaged over the runs
                                   -- exactly Predictor.predict_wing_bottom_pairs.
  4-camera model                   pair: on all 4, only the pair's outputs kept
                                   (it saw more cameras: a reference, not a rival).
                                   all4: on all 4.
  3-camera model                   pair: skipped (it needs a third camera).
                                   all4: skipped (never run on 4-camera movies).
                                   side3: on the side triad, as on the old rig
                                   (a 4-camera model is skipped there).
  per-camera model                 every setting: one camera at a time.

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
import sys

import h5py
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
def describe_model(run_dir):
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
        .replace("MODEL_PER_CAM_PER_WING_", "per_cam_").replace("_Sep 27", "")
    return {"name": name,
            "kind": kind, "dir": run_dir, "data path": cfg["data path"],
            "split file": cfg["split file"]}


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
        if len(cams) == num_cams - 1 and bottom not in cams:
            return None     # the side triad: this model would still see the bottom camera
        cm = model.confmaps(box)
        return {c: peaks(cam_confmaps(cm, c)) for c in cams}
    if kind == "all3":
        if len(cams) != 3:
            return None
        cm = model.confmaps(np.concatenate([cam_input(box, c) for c in cams], axis=1))
        return {c: peaks(cam_confmaps(cm, k)) for k, c in enumerate(cams)}
    if kind == "bottom2":
        if bottom not in cams:
            return None
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
    """[(frame, joint name, error in calibration units)]. 2 cameras: their
    triangulation. More: the median of every pair's triangulation. Head and
    tail come once per wing block; each is scored separately."""
    pairs = list(itertools.combinations(cams, 2))
    rows = []
    for i, (fr, blk) in enumerate(zip(groups, block)):
        gt3 = (R.T @ points_3D[fr].T).T
        for j in range(POINTS_PER_CAM):
            Xs = [triangulate_dlt(np.stack([uncrop(pred[c][i, j], cropzone[fr, c]) for c in pair]),
                                  P[list(pair)]) for pair in pairs]
            X = np.median(np.stack(Xs), axis=0)
            gi, jname = gt_joint(blk, j)
            rows.append((int(fr), jname, float(np.linalg.norm(X - gt3[gi]))))
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
    ap.add_argument("--models", nargs="+", required=True, help="training run folders")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    models = [describe_model(d) for d in args.models]
    for key in ("data path", "split file"):
        if len({m[key] for m in models}) != 1:
            raise SystemExit(f"the models disagree on '{key}'; they cannot share test frames")
    data_path, split_file = models[0]["data path"], models[0]["split file"]
    test_frames = np.flatnonzero(np.load(split_file)["frame_split"] == TEST)

    P, R, cropzone, points_3D, joints, centers = load_geometry(data_path)
    bottom = find_bottom_camera(centers)
    if bottom is None:
        raise SystemExit(f"no bottom camera in the camera positions of {data_path}")
    convention_px = reproject_check(P, R, cropzone, points_3D, joints, test_frames)

    box, confmaps, groups, block, num_cams = build_all_camera_samples(
        os.path.join(models[0]["dir"], "configuration.json"), os.path.join(args.out, "_scratch"))
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
    for m in entries:
        runner = None if m["kind"] == "gt" else Runner(os.path.join(m["dir"], "best_model.pt"),
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
            e3 = score_3d(pred, cams, groups, block, P, R, cropzone, points_3D)
            for fr, jname, d in e3:
                point_rows.append([m["name"], m["kind"], setting, fr, "-", -1, jname, "", f"{d * 1000:.5f}"])
            d3 = np.array([d for _, _, d in e3]) * 1000
            summary.append({"model": m["name"], "kind": m["kind"], "setting": setting,
                            "cams": list(map(int, cams)),
                            "mean_px": float(err2d.mean()), "median_px": float(np.median(err2d)),
                            "p95_px": float(np.percentile(err2d, 95)),
                            "frac_over_10px": float((err2d > 10).mean()),
                            "mean_3d_mm": float(d3.mean()), "median_3d_mm": float(np.median(d3)),
                            "p95_3d_mm": float(np.percentile(d3, 95))})
            s = summary[-1]
            print(f"{m['name']:>24} {setting:>7}  2D mean {s['mean_px']:6.2f} px  median "
                  f"{s['median_px']:5.2f}  p95 {s['p95_px']:6.2f}   3D median "
                  f"{s['median_3d_mm']:.4f} mm  p95 {s['p95_3d_mm']:.4f}", flush=True)

    with open(os.path.join(args.out, "points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind", "setting", "frame", "wing_block", "cam", "joint",
                    "err_px", "err_3d_mm"])
        w.writerows(point_rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"data path": data_path, "split file": split_file,
                   "test frames": test_frames.tolist(), "bottom camera": int(bottom),
                   "convention check px": convention_px,
                   "models": {m["name"]: m["dir"] for m in models},
                   "results": summary}, f, indent=2)
    print(f"wrote {args.out}/points.csv and summary.json")


if __name__ == "__main__":
    main()
