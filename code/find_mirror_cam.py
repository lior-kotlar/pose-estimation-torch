"""
find_mirror_cam.py
==================

Work out WHICH camera (if any) is the mirrored one, by testing every flip
hypothesis against the calibration's own geometry.

Each rig has one camera that views the arena through a mirror, so its frames
are vertically inverted relative to the DLT that easyWand solved. The pipeline
fixes that in the FLIP step (`process_experiment.py --cam <name>`), which
rewrites that camera's `*_sparse.mat` in place. Two things make guessing
expensive: the flip is NOT idempotent (running it twice silently undoes it),
and the mirror cam is a per-rig fact the data does not record -- these mats
have no `metaData.isFlipped`. Get it wrong and triangulation is quietly wrong
everywhere.

    .env/bin/python code/find_mirror_cam.py <movie_dir|experiment_dir> \\
        --easywand <easyWandData.mat>

WHY THIS NEEDS NEITHER MATLAB NOR A BUILT h5
--------------------------------------------
Flipping a camera's SOURCE DATA and flipping that camera's DLT are the same
test. Writing F = [[1,0,0],[0,-1,H+1],[0,0,1]], a camera's ray constraint is

    (F M) X ~ (u, v, 1)   <=>   M X ~ F^-1 (u, v, 1) = (u, (H+1)-v, 1)

and (H+1)-v is exactly the measurement you get after flipping the mat. Same
ray, so same triangulation and the same leave-one-out error -- for the flipped
camera and for every other camera. So the whole hypothesis space can be swept
on the RAW mats, before anything is flipped or built.

For the same reason the DLT is read straight out of the easyWand `.mat`:
`calibration.h5`'s `camera_matrices` is just `reshape([coefs; 1], 3, 4)` per
camera (verified exactly against a built calibration), so MATLAB is not needed
to get it. Pass `--calibration <calibration.h5>` instead if one already exists.

WHAT IT REPORTS
---------------
All 2^n flip subsets, ranked by the WORST camera's leave-one-out reprojection
error. LOO is the discriminating statistic: the held-out camera contributes
nothing to the 3D point, so its error is unbiased. A correct configuration
gives every camera a few px; a wrong one gives the offending camera hundreds.

The all-cameras-flipped row is the same hypothesis as
`verify_calibration.py --no-yflip` (a global y-up/y-down convention flip), and
is labelled as such -- if that one wins, the problem is the convention, not a
mirror.

NOTE ON CAMERA ORDER
--------------------
Camera index = position in the SORTED `*_sparse.mat` listing, because that is
the order the MATLAB builder assembles the box in (`dir()` + `endsWith`). The
report prints the matching cam NAME so the result can be handed straight to
`--cam`.
"""

import argparse
import glob
import itertools
import os
import re
import sys

import h5py
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def dlt_from_easywand(path):
    """(num_cams, 3, 4) DLT matrices straight from an easyWand .mat.

    easyWand stores the 11-parameter DLT per camera in `coefs` (11, nCams);
    the camera matrix is those 11 with a 1 appended, reshaped row-major. That
    is bit-for-bit what `create_camera_calibration_h5.m` writes into
    `camera_matrices`, so this is the same calibration, not an approximation.
    """
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "easyWandData" not in d:
        sys.exit(f"{path}: no 'easyWandData' struct (is this an easyWand mat?)")
    ew = d["easyWandData"]
    coefs = np.asarray(ew.coefs, dtype=float)
    if coefs.ndim == 1:
        coefs = coefs[:, None]
    n_cams = coefs.shape[1]
    M = np.stack([np.append(coefs[:, c], 1.0).reshape(3, 4)
                  for c in range(n_cams)])
    dist = int(np.asarray(ew.distortionMode).squeeze()) if hasattr(ew, "distortionMode") else 0
    if dist:
        print(f"  WARNING: distortionMode={dist}. This DLT is only valid on "
              f"pixels that were undistorted first, and nothing here does "
              f"that -- the errors below will be inflated for every hypothesis.")
    return M


def dlt_from_calibration_h5(path):
    with h5py.File(path, "r") as f:
        # MATLAB writes column-major; .T restores (cam, row, col).
        return f["camera_matrices"][:].T


# ---------------------------------------------------------------------------
# Measurements straight off the sparse mats
# ---------------------------------------------------------------------------
def largest_blob_centroid(rows, cols, vals, bg_t, min_pixels):
    """(row, col) centroid of the fly's blob, in the mat's 1-based full-image
    coordinates -- or None.

    Mirrors what the MATLAB builder keeps: it forms the negative image
    `bg - value`, discards everything outside the single largest connected
    component, and crops around what is left. The centroid of those surviving
    pixels is what `verify_calibration.py` measures off the built box, so
    computing it here from the raw mat gives the same quantity without a build.
    """
    # bg_t is the h5py view of metaData.bg, i.e. MATLAB's bg transposed:
    # bg_t[col-1, row-1] == bg(row, col).
    neg = bg_t[cols - 1, rows - 1].astype(np.int32) - vals.astype(np.int32)
    keep = neg > 0
    if keep.sum() < min_pixels:
        return None
    rows, cols = rows[keep], cols[keep]

    r_lo, c_lo = rows.min(), cols.min()
    img = np.zeros((rows.max() - r_lo + 1, cols.max() - c_lo + 1), dtype=np.uint8)
    img[rows - r_lo, cols - c_lo] = 1
    labels, n_blobs = ndi.label(img)
    if n_blobs > 1:
        sizes = np.bincount(labels.ravel())[1:]
        big = int(np.argmax(sizes)) + 1
        rs, cs = np.where(labels == big)
        if len(rs) < min_pixels:
            return None
        return float(rs.mean() + r_lo), float(cs.mean() + c_lo)
    return float(rows.mean()), float(cols.mean())


def collect_from_movie(movie_dir, n_samples, min_pixels, image_height):
    """(meas, n_cams, cam_names) for one movie.

    meas is (n_frames_kept, n_cams, 2) of [u, v] in the DLT's input
    convention: u = full-image column, v = (H+1) - full-image row. Only frames
    where EVERY camera yields a blob are kept, so every row can be
    triangulated from any camera subset.
    """
    mats = sorted(glob.glob(os.path.join(movie_dir, "*_sparse.mat")))
    if len(mats) < 2:
        return None, 0, []
    cam_names = [_cam_name(m) for m in mats]
    n_cams = len(mats)

    per_cam = []
    n_frames = None
    for p in mats:
        with h5py.File(p, "r") as f:
            bg_t = f["metaData/bg"][()]
            refs = f["frames/indIm"][0]
            n = len(refs)
            n_frames = n if n_frames is None else min(n_frames, n)
            stride = max(1, n // max(n_samples, 1))
            idx = list(range(0, n, stride))
            pts = {}
            for i in idx:
                d = f[refs[i]]
                if d.ndim != 2 or d.shape[1] < min_pixels:
                    continue
                a = d[:]
                c = largest_blob_centroid(a[0].astype(int), a[1].astype(int),
                                          a[2], bg_t, min_pixels)
                if c is not None:
                    pts[i] = c
            per_cam.append(pts)

    common = sorted(set.intersection(*(set(p) for p in per_cam)))
    if not common:
        return None, n_cams, cam_names
    meas = np.empty((len(common), n_cams, 2))
    for k, i in enumerate(common):
        for cam in range(n_cams):
            r, c = per_cam[cam][i]
            meas[k, cam] = [c, (image_height + 1) - r]
    return meas, n_cams, cam_names


def _cam_name(mat_path):
    m = re.search(r"_(cam\d+)_sparse\.mat$", os.path.basename(mat_path), re.I)
    return m.group(1) if m else os.path.basename(mat_path)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def triangulate(meas_frame, M, cams):
    rows = []
    for cam in cams:
        u, v = meas_frame[cam]
        P = M[cam]
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.stack(rows))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def loo_medians(meas, M):
    """Median leave-one-out reprojection error (px) per camera.

    Held-out camera c is reprojected from a 3D point triangulated WITHOUT it,
    so nothing pulls the point toward c and the error is an honest test of
    whether c's DLT agrees with the rest.
    """
    n_frames, n_cams, _ = meas.shape
    errs = np.full((n_frames, n_cams), np.nan)
    for k in range(n_frames):
        for held in range(n_cams):
            others = [c for c in range(n_cams) if c != held]
            X = triangulate(meas[k], M, others)
            if X is None:
                continue
            p = M[held] @ np.append(X, 1.0)
            if abs(p[2]) < 1e-12:
                continue
            errs[k, held] = np.linalg.norm(p[:2] / p[2] - meas[k, held])
    return np.array([np.nanmedian(errs[:, c]) if np.any(~np.isnan(errs[:, c]))
                     else np.inf for c in range(n_cams)])


def flip_matrix(image_height):
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, image_height + 1.0],
                     [0.0, 0.0, 1.0]])



# ---------------------------------------------------------------------------
# Core: sweep the hypotheses and reach a verdict
# ---------------------------------------------------------------------------
# A verdict is only acted on when it is UNAMBIGUOUS: the winning hypothesis has
# to be clean in absolute terms, and clearly better than the next one. On real
# data the separation is enormous -- ex241220 gave 6 px for the truth against
# 167 px for the nearest rival, a factor of 25 -- so these gates never fire on
# a healthy experiment, and an experiment they do fire on is one where nobody
# should be flipping anything on this evidence.
DEFAULT_CLEAN_PX = 15.0     # worst-camera LOO median the winner must beat
DEFAULT_MARGIN = 3.0        # how many times better than the runner-up


def gather_measurements(movie_dirs, samples, min_pixels, image_height,
                        verbose=True):
    """(meas, n_cams, cam_names) pooled over several movies, or (None, ...)."""
    all_meas, cam_names, n_cams = [], None, None
    for d in movie_dirs:
        meas, nc, names = collect_from_movie(d, samples, min_pixels, image_height)
        if meas is None or not len(meas):
            if verbose:
                print(f"  {os.path.basename(d):>10}: no frame had all cameras -- skipped")
            continue
        if n_cams is None:
            n_cams, cam_names = nc, names
        elif nc != n_cams:
            if verbose:
                print(f"  {os.path.basename(d):>10}: {nc} cams, expected {n_cams} -- skipped")
            continue
        if verbose:
            print(f"  {os.path.basename(d):>10}: {len(meas):>4} usable frames "
                  f"({', '.join(names)})")
        all_meas.append(meas)
    if not all_meas:
        return None, n_cams, cam_names
    return np.concatenate(all_meas, axis=0), n_cams, cam_names


def evaluate_hypotheses(meas, M, n_cams, image_height):
    """Every flip subset, ranked by the WORST camera's LOO median.

    All 2**n subsets, including the all-flipped one -- which is the same
    hypothesis as a global y-up/y-down convention error rather than a mirror,
    and is labelled as such in the report."""
    F = flip_matrix(image_height)
    results = []
    for r in range(n_cams + 1):
        for subset in itertools.combinations(range(n_cams), r):
            Mf = M.copy()
            for c in subset:
                Mf[c] = F @ Mf[c]
            med = loo_medians(meas, Mf)
            results.append((subset, med, float(np.max(med))))
    results.sort(key=lambda t: t[2])
    return results


def verdict_from_results(results, cam_names, n_cams,
                         clean_px=DEFAULT_CLEAN_PX, margin=DEFAULT_MARGIN):
    """Turn the ranked hypotheses into an actionable verdict.

    `conclusive` False means "do not act on this" -- either nothing is clean
    (a calibration problem, which flipping cannot fix) or two hypotheses are
    too close to separate. `flip` is the list of camera NAMES to flip; empty
    means the data already agrees with the calibration.
    """
    best_subset, _, best_worst = results[0]
    runner_up = results[1][2] if len(results) > 1 else float("inf")
    v = {
        "flip": [cam_names[c] for c in best_subset],
        "worst": best_worst,
        "runner_up": runner_up,
        "cam_names": cam_names,
        "results": results,
        "conclusive": False,
        "reason": "",
    }
    if not np.isfinite(best_worst) or best_worst > clean_px:
        v["reason"] = (f"no hypothesis is clean -- the best still leaves "
                       f"{best_worst:.1f} px on its worst camera. A mirror "
                       f"flip is not what is wrong here; suspect the wrong "
                       f"easyWand for this experiment, a distortion-mode "
                       f"calibration, or cameras that are not frame-aligned.")
        return v
    if runner_up < margin * max(best_worst, 1e-6):
        v["reason"] = (f"the best two hypotheses are too close to separate "
                       f"({best_worst:.2f} px vs {runner_up:.2f} px, under "
                       f"{margin:g}x) -- refusing to guess.")
        return v
    if len(best_subset) == n_cams:
        v["reason"] = (f"every camera wants flipping, which is a y-up/y-down "
                       f"convention mismatch rather than a mirror -- the mats "
                       f"are not what is wrong.")
        return v
    if len(best_subset) > 1:
        v["reason"] = (f"more than one camera wants flipping "
                       f"({', '.join(v['flip'])}), which a single-mirror rig "
                       f"should never produce.")
        return v
    v["conclusive"] = True
    return v


def frame_height(movie_dir):
    """Image height (rows) from a movie's own metaData.frameSize, or None.

    The flip is `row -> (H+1) - row`, so H has to be the height these mats were
    actually recorded at -- a hardcoded default silently skews every hypothesis
    if a rig ever differs."""
    mats = sorted(glob.glob(os.path.join(movie_dir, "*_sparse.mat")))
    if not mats:
        return None
    try:
        with h5py.File(mats[0], "r") as f:
            return int(np.asarray(f["metaData/frameSize"][()]).flatten()[0])
    except (OSError, KeyError):
        return None


def detect_mirror_cam(movie_dirs, easywand=None, calibration=None,
                      samples=100, min_pixels=50, image_height=None,
                      clean_px=DEFAULT_CLEAN_PX, margin=DEFAULT_MARGIN,
                      verbose=True):
    """Which camera (if any) needs the vertical flip, as a verdict dict.

    Reads the RAW mats and tests the hypotheses against the calibration, so it
    is valid before anything has been flipped or built -- and, because a flip
    that has already been applied makes the flipped hypothesis the wrong one,
    it equally reports "flip nothing" on data that is already correct. That is
    what lets it stand in for the idempotency marker the mats do not carry.

    Returns None when it could not run at all (no calibration, no usable
    frames); otherwise a dict from verdict_from_results.
    """
    if not (easywand or calibration):
        return None
    if image_height is None:
        image_height = frame_height(movie_dirs[0]) if movie_dirs else None
        if image_height is None:
            return None
        if verbose:
            print(f"  image height {image_height} px (from metaData.frameSize)")
    M = (dlt_from_calibration_h5(calibration) if calibration
         else dlt_from_easywand(easywand))
    meas, n_cams, cam_names = gather_measurements(
        movie_dirs, samples, min_pixels, image_height, verbose=verbose)
    if meas is None:
        return None
    if M.shape[0] != n_cams:
        return {"conclusive": False, "flip": [], "worst": float("inf"),
                "runner_up": float("inf"), "cam_names": cam_names,
                "results": [],
                "reason": (f"the calibration describes {M.shape[0]} cameras "
                           f"but the movies have {n_cams} -- wrong easyWand "
                           f"for this experiment?")}
    results = evaluate_hypotheses(meas, M, n_cams, image_height)
    return verdict_from_results(results, cam_names, n_cams, clean_px, margin)


def print_hypothesis_table(verdict):
    """The full ranked sweep, so a verdict can always be second-guessed."""
    results, cam_names = verdict["results"], verdict["cam_names"]
    if not results:
        return
    n_cams = len(cam_names)
    width = max(len("flip nothing"),
                max(len(", ".join(cam_names[c] for c in s)) for s, _, _ in results))
    print(f"  {'hypothesis':<{width}}  "
          + "  ".join(f"{n:>8}" for n in cam_names) + f"  {'worst':>8}")
    print(f"  {'-' * width}  " + "  ".join("-" * 8 for _ in cam_names) + "  " + "-" * 8)
    for subset, med, worst in results:
        label = ", ".join(cam_names[c] for c in subset) if subset else "flip nothing"
        note = "   (== verify_calibration --no-yflip)" if len(subset) == n_cams else ""
        print(f"  {label:<{width}}  "
              + "  ".join(f"{m:>8.2f}" for m in med)
              + f"  {worst:>8.2f}{note}")


def discover_movie_dirs(path, limit=None):
    """Movie dirs under `path`, or [path] when it is itself one.

    When limiting, the sample is spread across the listing rather than taken
    from the front -- consecutive movies are the most likely to share whatever
    made one of them unusable."""
    if glob.glob(os.path.join(path, "*_sparse.mat")):
        return [path]
    dirs = sorted(d for d in glob.glob(os.path.join(path, "mov*"))
                  if os.path.isdir(d)
                  and glob.glob(os.path.join(d, "*_sparse.mat")))
    if limit and len(dirs) > limit:
        step = max(1, len(dirs) // limit)
        dirs = dirs[::step][:limit]
    return dirs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a mov<N>/ dir, or an experiment dir of them")
    ap.add_argument("--easywand", help="easyWand .mat to read the DLT from")
    ap.add_argument("--calibration", help="calibration.h5 to read the DLT from")
    ap.add_argument("--movies", type=int, default=3,
                    help="how many movies to sample in experiment mode "
                         "(default 3; more is slower but steadier)")
    ap.add_argument("--samples", type=int, default=150,
                    help="frames to sample per movie (default 150)")
    ap.add_argument("--min-pixels", type=int, default=50,
                    help="blob pixels required for a camera to count (default 50)")
    ap.add_argument("--image-height", type=int, default=None,
                    help="override the height read from metaData.frameSize")
    args = ap.parse_args()

    if not (args.easywand or args.calibration):
        sys.exit("need --easywand or --calibration")
    print(f"calibration: {args.calibration or args.easywand}")
    movie_dirs = discover_movie_dirs(args.path, args.movies)
    if not movie_dirs:
        sys.exit(f"no movies with *_sparse.mat under {args.path}")

    verdict = detect_mirror_cam(movie_dirs, easywand=args.easywand,
                                calibration=args.calibration,
                                samples=args.samples,
                                min_pixels=args.min_pixels,
                                image_height=args.image_height)
    if verdict is None:
        sys.exit("no usable measurements")
    print()
    print_hypothesis_table(verdict)

    print("\nVerdict:")
    if not verdict["conclusive"]:
        print(f"  INCONCLUSIVE -- {verdict['reason']}")
        return 1
    if not verdict["flip"]:
        print(f"  No flip needed. Every camera already agrees with the "
              f"calibration (worst LOO median {verdict['worst']:.2f} px, vs "
              f"{verdict['runner_up']:.2f} px for the next hypothesis).")
        print(f"  Run the pipeline with --skip-flip.")
        return 0
    cam = verdict["flip"][0]
    print(f"  Flip {cam}  (worst LOO median {verdict['worst']:.2f} px, vs "
          f"{verdict['runner_up']:.2f} px for the next hypothesis).")
    print(f"  Run the pipeline with --cam {cam}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
