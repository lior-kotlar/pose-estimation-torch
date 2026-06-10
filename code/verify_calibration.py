"""
verify_calibration.py — reprojection-error sanity check for the dataset h5 +
calibration h5, independent of the pose-estimation network.

For each frame we use the blob silhouette centroid in each camera's cropped
image as a "free measurement" of where the insect is in that camera's full
image. We then:

  1. Uncrop + apply the y-flip the Triangulator applies (y -> H+1-y),
     producing per-cam pixel coordinates in the DLT's input convention.
  2. Triangulate the 4-cam measurement to a 3D point via linear-SVD on the
     stacked DLT rows from the calibration h5.
  3. Reproject that 3D point back through each camera's DLT and measure the
     pixel error against the per-cam measurement.

Two passes:

  (A) ALL-CAM: triangulate using all 4 valid cams, reproject every cam.
      Each cam contributes its own measurement, so errors here are biased
      *down* when a cam is broken (it drags the 3D point toward itself).

  (B) LEAVE-ONE-OUT: for each cam c, triangulate using only the other 3 cams,
      then reproject cam c. The held-out cam's error is unbiased — this is
      the smoking-gun test. If cam c's DLT or data is inconsistent with the
      rest, LOO error spikes (often into the hundreds of pixels).

Optional `--flip-cam N` pre-multiplies cam N's DLT by F = diag(1, -1, 1) with
translation +H+1 — i.e. bakes a vertical-flip into that cam's calibration.
Useful for testing the hypothesis: "I flipped cam5's source data, so its DLT
(which was calibrated on the mirrored frames) now expects flipped pixels."
"""

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_calibration(path):
    """Load DLT matrices in (cam, row, col) layout."""
    with h5py.File(path, "r") as f:
        # h5 files written from MATLAB are column-major; .T restores the
        # logical (cam, row, col) order the Triangulator expects.
        M = f["camera_matrices"][:].T          # (num_cams, 3, 4)
    return M


def to_matlab_orientation(box_frame_channel):
    """h5py reads `box` from a MATLAB-written h5 file with the last two axes
    reversed vs the displayed image. The Visualizer transposes before display
    (line 1038); we do the same so our (x, y) match the Triangulator's
    cropzone-relative coordinates."""
    return box_frame_channel.T


def blob_centroid(img):
    """Centroid of the blob silhouette in (x, y) crop coords.
    `img` is the 192x192 crop after to_matlab_orientation; outside-blob pixels
    are exactly 0 (mat2gray's min), inside-blob pixels are > 0."""
    mask = img > 0
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def collect_measurements(box, cropzone, image_height, num_cams,
                         time_channels=3, time_idx=1, apply_yflip=True):
    """Returns:
      meas: (n_frames, num_cams, 2) of [u, v] in DLT-input coords
            (u = full-image column, v = (H+1) - full-image row).
      valid: (n_frames, num_cams) bool.
    """
    n_frames = box.shape[0]
    meas = np.full((n_frames, num_cams, 2), np.nan)
    valid = np.zeros((n_frames, num_cams), dtype=bool)

    for k in range(n_frames):
        for cam in range(num_cams):
            chan = cam * time_channels + time_idx   # middle (present) time
            img = to_matlab_orientation(box[k, chan])
            c = blob_centroid(img)
            if c is None:
                continue
            x_in_crop, y_in_crop = c
            # cropzone layout (MATLAB column written to h5): [row_top, col_left]
            row_top = float(cropzone[k, cam, 0])
            col_left = float(cropzone[k, cam, 1])
            x_full = col_left + x_in_crop
            y_full = row_top + y_in_crop
            v_dlt = (image_height + 1) - y_full if apply_yflip else y_full
            meas[k, cam] = [x_full, v_dlt]
            valid[k, cam] = True

    return meas, valid


def triangulate(meas_frame, M, valid_mask):
    """Linear SVD triangulation. Returns 3D point or None."""
    cams_used = np.nonzero(valid_mask)[0]
    if len(cams_used) < 2:
        return None
    rows = []
    for cam in cams_used:
        u, v = meas_frame[cam]
        P = M[cam]                       # (3, 4)
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-12:
        return None
    return X_h[:3] / X_h[3]


def reproject(X, M_cam):
    p = M_cam @ np.append(X, 1.0)
    return p[:2] / p[2]


def per_cam_errors(meas, valid, M, mode="all"):
    """Compute per-frame, per-cam reprojection error in pixels.
       mode='all' : triangulate from all valid cams.
       mode='loo' : for each cam c, triangulate from cams != c, reproject c.
    """
    n_frames, num_cams, _ = meas.shape
    errs = np.full((n_frames, num_cams), np.nan)
    for k in range(n_frames):
        if mode == "all":
            X = triangulate(meas[k], M, valid[k])
            if X is None:
                continue
            for cam in range(num_cams):
                if not valid[k, cam]:
                    continue
                uv = reproject(X, M[cam])
                errs[k, cam] = np.linalg.norm(uv - meas[k, cam])
        else:  # loo
            for held in range(num_cams):
                if not valid[k, held]:
                    continue
                loo_mask = valid[k].copy()
                loo_mask[held] = False
                X = triangulate(meas[k], M, loo_mask)
                if X is None:
                    continue
                uv = reproject(X, M[held])
                errs[k, held] = np.linalg.norm(uv - meas[k, held])
    return errs


def report_table(label, errs):
    print(f"\n{label}")
    print(f"  {'cam':>4} {'n':>6} {'mean':>10} {'median':>10} {'p90':>10} {'max':>10}")
    out = {}
    for cam in range(errs.shape[1]):
        e = errs[:, cam]; e = e[~np.isnan(e)]
        if len(e) == 0:
            print(f"  {cam:>4} {0:>6}    no data")
            continue
        stats = (len(e), e.mean(), np.median(e), np.percentile(e, 90), e.max())
        out[cam] = stats
        print(f"  {cam:>4} {stats[0]:>6} {stats[1]:>10.2f} {stats[2]:>10.2f} "
              f"{stats[3]:>10.2f} {stats[4]:>10.2f}")
    return out


def parse_frame_selector(arg, n_total):
    if arg is None:
        return slice(None)
    if ":" in arg:
        a, b = arg.split(":")
        return slice(int(a or 0), int(b or n_total))
    if "," in arg:
        return np.array([int(x) for x in arg.split(",")])
    return slice(0, int(arg))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("h5_path", help="dataset h5 with /box and /cropzone")
    ap.add_argument("calib_path", help="calibration h5 with /camera_matrices")
    ap.add_argument("--image-height", type=int, default=800)
    ap.add_argument("--num-cams", type=int, default=4)
    ap.add_argument("--frames", default=None,
                    help="N (first N), A:B (range), or comma list. Default: all.")
    ap.add_argument("--flip-cam", type=int, default=None,
                    help="bake F = vertical-flip into this cam's DLT before testing")
    ap.add_argument("--out", default="calib_check.png", help="output histogram PNG")
    ap.add_argument("--print-frame", type=int, default=None,
                    help="print per-cam measured vs reprojected (u,v) for this frame "
                         "(0-indexed in the loaded subset). Helps diagnose the error structure.")
    ap.add_argument("--no-yflip", action="store_true",
                    help="skip the uniform y = H+1-y step when uncropping. Use this to test "
                         "whether the calibration was built with a y-up vs y-down convention "
                         "different from what the Triangulator assumes.")
    args = ap.parse_args()

    M = load_calibration(args.calib_path)
    print(f"calibration: camera_matrices shape {M.shape}")
    if M.shape[0] < args.num_cams:
        sys.exit(f"calibration has {M.shape[0]} cams but --num-cams={args.num_cams}")

    if args.flip_cam is not None:
        H = args.image_height
        F = np.array([[1.0, 0, 0], [0, -1.0, H + 1.0], [0, 0, 1.0]])
        before = M[args.flip_cam].copy()
        M[args.flip_cam] = F @ M[args.flip_cam]
        print(f"applied F flip to cam {args.flip_cam}; "
              f"top-row before [{before[0,0]:.3g}, ...], "
              f"after [{M[args.flip_cam,0,0]:.3g}, ...]")

    with h5py.File(args.h5_path, "r") as f:
        n_total = f["box"].shape[0]
        sel = parse_frame_selector(args.frames, n_total)
        box = f["box"][sel]
        cropzone = f["cropzone"][sel]
    print(f"loaded {box.shape[0]} of {n_total} frames; "
          f"box shape {box.shape}, cropzone shape {cropzone.shape}")

    meas, valid = collect_measurements(box, cropzone,
                                       args.image_height, args.num_cams,
                                       apply_yflip=not args.no_yflip)
    print(f"valid measurements per cam: {valid.sum(axis=0).tolist()}")
    print(f"y-flip applied: {not args.no_yflip}")

    if args.print_frame is not None:
        k = args.print_frame
        print(f"\n--- frame {k} per-cam detail ---")
        print(f"  cropzone[{k}] (row_top, col_left): "
              + ", ".join(f"cam{c}=({cropzone[k,c,0]},{cropzone[k,c,1]})"
                          for c in range(args.num_cams)))
        # use all valid cams to triangulate, then show measured vs reprojected for each
        X = triangulate(meas[k], M, valid[k])
        print(f"  3D estimate (all-cam): {X}")
        print(f"  {'cam':>4} {'meas (u,v)':>22} {'reproj (u,v)':>22} {'delta (du,dv)':>22} {'|err|':>10}")
        for cam in range(args.num_cams):
            if not valid[k, cam]:
                print(f"  {cam:>4}    invalid"); continue
            u_meas, v_meas = meas[k, cam]
            u_rep, v_rep = reproject(X, M[cam])
            du, dv = u_rep - u_meas, v_rep - v_meas
            err = (du**2 + dv**2) ** 0.5
            print(f"  {cam:>4} ({u_meas:>9.2f},{v_meas:>9.2f}) "
                  f"({u_rep:>9.2f},{v_rep:>9.2f}) "
                  f"({du:>9.2f},{dv:>9.2f}) {err:>10.2f}")
        print("---")

    errs_all = per_cam_errors(meas, valid, M, mode="all")
    errs_loo = per_cam_errors(meas, valid, M, mode="loo")

    stats_all = report_table("ALL-CAM TRIANGULATION (biased — each cam pulls 3D toward itself):",
                             errs_all)
    stats_loo = report_table("LEAVE-ONE-OUT (unbiased — strongest signal):",
                             errs_loo)

    # plot histograms (LOO is the informative one)
    fig, axs = plt.subplots(1, args.num_cams, figsize=(4 * args.num_cams, 3.2),
                            sharey=True)
    if args.num_cams == 1:
        axs = [axs]
    for cam in range(args.num_cams):
        e = errs_loo[:, cam]; e = e[~np.isnan(e)]
        if len(e) == 0:
            axs[cam].set_title(f"cam{cam} (no data)"); continue
        clip = np.clip(e, 0, max(50, np.percentile(e, 99)))
        axs[cam].hist(clip, bins=40, color="steelblue", edgecolor="black", linewidth=0.3)
        axs[cam].set_xlabel("LOO reproj err (px)")
        axs[cam].set_title(f"cam{cam}  med={np.median(e):.1f}px  p90={np.percentile(e,90):.1f}px")
    axs[0].set_ylabel("frames")
    fig.suptitle(
        f"Leave-one-out reprojection error  ({args.h5_path}"
        + (f"  +F on cam{args.flip_cam}" if args.flip_cam is not None else "")
        + ")",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=120)
    print(f"\nhistogram saved to {args.out}")

    # heuristic verdict
    print("\nHeuristic verdict:")
    if not stats_loo:
        print("  no frames triangulated — can't diagnose.")
        return
    loo_medians = {c: s[2] for c, s in stats_loo.items()}
    overall = np.median(list(loo_medians.values()))
    outliers = [c for c, m in loo_medians.items() if m > 5 * max(overall, 1)]
    if all(m < 5 for m in loo_medians.values()):
        print("  All cams have LOO median < 5px. Calibration looks consistent with the data.")
        print("  If pose-estimation output is still garbage, the bug is downstream of triangulation.")
    elif outliers:
        print(f"  Cam(s) {outliers} have LOO median >> the others. "
              f"That cam's DLT and data disagree.")
        print(f"  If exactly one cam is the outlier and you recently flipped its source .mat, "
              f"try re-running with --flip-cam <that-cam-idx> to test the F-flip hypothesis.")
    else:
        print("  All cams have elevated LOO error. Likely culprit: wrong calibration h5, "
              "wrong easyWand mat, or a cam-ordering mismatch between sparse mats "
              "(alphabetical) and easyWand's cams_array.")


if __name__ == "__main__":
    main()
