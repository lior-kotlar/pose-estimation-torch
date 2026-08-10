"""
scan_sparse_movies.py
=====================

Pre-flight tool: look at a movie's 4 *_sparse.mat files and decide whether
the fly is trackable by all 4 cameras (with no second fly) for enough frames
to be worth processing.

Three per-frame, per-cam conditions are checked:

  1. "Fly visible":  the cam has more than --pixel-threshold non-zero pixels
     for that frame.
  2. "Single fly":   connected-component labeling of those pixels finds at
     most one significant blob, where a second blob counts as a separate fly
     only if its size is >= --blob-ratio * the largest blob's size AND its
     centroid is >= --blob-distance px away from the largest's centroid. This
     avoids false positives when a single fly's own wing/leg detaches from
     the body in the binary mask.
  3. "Fully in frame": the largest blob's bounding box stays at least
     --min-edge-margin px away from all four image borders. A fly flying out
     of the field of view is still "visible" (a fly 95% out of frame leaves
     far more than 50 pixels behind) and is still a single blob, so
     conditions 1-2 happily keep frames where the cam sees half a fly. The
     builder then hands the network a crop of a truncated fly, and the 2D
     detections there are meaningless -- which propagates into the
     triangulation and poisons the 3D pose for every camera at once. This
     is why bad predictions cluster at the END of a movie: the fly leaves
     the frame and never comes back.

A frame is "good" if all three conditions hold for ALL 4 cams. The longest
contiguous run of good frames is what BUILD will actually extract into the
h5. If that run is shorter than --min-intersection (default 500), the movie
is flagged BAD and skipped, sparing the heavy MATLAB build (~5 min/movie)
and 3-hour GPU prediction.

USAGE
-----
    # Scan one movie:
    .env/bin/python code/scan_sparse_movies.py path/to/mov<N>/

    # Scan whole experiment:
    .env/bin/python code/scan_sparse_movies.py path/to/<experiment>/

    # Override thresholds:
    .env/bin/python code/scan_sparse_movies.py path/to/exp/ \\
        --min-intersection 1000 --pixel-threshold 100

    # Keep the fly a comfortable distance from the border (or disable with 0):
    .env/bin/python code/scan_sparse_movies.py path/to/exp/ --min-edge-margin 15

Per-movie cost ~5 s. Returns nonzero exit code if any BAD movies found
(useful for shell pipelines).
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import scipy.ndimage as ndi

# Clearance, in pixels, the fly's blob must keep from every image border for a
# frame to be usable. 0 disables the check (any frame where the blob merely
# touches the border still counts as in-frame). A small non-zero default is
# deliberate: the last few pixels before the border are already a fly whose
# silhouette is being clipped, and the network was trained on whole flies.
DEFAULT_MIN_EDGE_MARGIN = 5


def _blob_stats(indIm: np.ndarray, ratio_threshold: float,
                distance_threshold: float) -> tuple:
    """Connected-component analysis of a frame's sparse pixels. Returns
    (is_single, bbox) where:

      is_single: True if there's only one "significant" blob — i.e. either ≤1
        connected component, or the second-largest is < `ratio_threshold` *
        largest, or its centroid is within `distance_threshold` px of the
        largest's (so it's plausibly the fly's own wing/leg detached in the
        binary mask, not a separate fly).
      bbox: (row_min, row_max, col_min, col_max) of the LARGEST blob, in the
        mat's 1-based full-image coordinates. That is the blob the MATLAB
        builder keeps (`bwareafilt(full_im>0,1)`) and crops around, so it is
        the one whose distance to the image border decides whether the crop
        holds a whole fly or a truncated one."""
    rows = indIm[0].astype(int)
    cols = indIm[1].astype(int)
    r_lo, r_hi = rows.min(), rows.max()
    c_lo, c_hi = cols.min(), cols.max()
    img = np.zeros((r_hi - r_lo + 1, c_hi - c_lo + 1), dtype=np.uint8)
    img[rows - r_lo, cols - c_lo] = 1
    labels, n_blobs = ndi.label(img)
    if n_blobs <= 1:
        return True, (r_lo, r_hi, c_lo, c_hi)
    sizes = np.bincount(labels.ravel())[1:]  # drop background label 0
    order = np.argsort(sizes)[::-1]
    mask1 = labels == (order[0] + 1)
    rs1, cs1 = np.where(mask1)
    bbox = (int(rs1.min()) + r_lo, int(rs1.max()) + r_lo,
            int(cs1.min()) + c_lo, int(cs1.max()) + c_lo)
    s1, s2 = int(sizes[order[0]]), int(sizes[order[1]])
    if s2 < ratio_threshold * s1:
        return True, bbox
    mask2 = labels == (order[1] + 1)
    rs2, cs2 = np.where(mask2)
    dist = float(np.hypot(rs1.mean() - rs2.mean(),
                          cs1.mean() - cs2.mean()))
    return dist < distance_threshold, bbox


def analyze_cam_frames(mat_path: str, pixel_threshold: int,
                       blob_ratio: float,
                       blob_distance: float,
                       min_edge_margin: float = DEFAULT_MIN_EDGE_MARGIN) -> tuple:
    """Per-frame analysis of one cam's sparse mat. Returns:
        (visible_mask, single_fly_mask, in_frame_mask) — bool arrays of
        length n_frames.
      visible_mask[i]: True iff frame i has > pixel_threshold pixels.
      single_fly_mask[i]: True iff frame i has at most one significant blob
        (by the size+distance rule). Empty frames are "single" vacuously.
      in_frame_mask[i]: True iff the largest blob's bounding box keeps at
        least min_edge_margin px of clearance from every image border, i.e.
        the fly is wholly inside the field of view. Frames where the cam sees
        nothing are "in frame" vacuously — condition 1 already rejects those,
        and marking them False as well would only muddy the per-cam counts."""
    with h5py.File(mat_path, "r") as f:
        # frameSize is MATLAB's [height; width] and is what the builder uses
        # to clamp its crop window, so it's the right border reference.
        frame_size = f["metaData/frameSize"][()].flatten()
        height, width = int(frame_size[0]), int(frame_size[1])
        refs = f["frames/indIm"][0]
        n = len(refs)
        visible = np.zeros(n, dtype=bool)
        single = np.ones(n, dtype=bool)    # default True (vacuous for empty)
        in_frame = np.ones(n, dtype=bool)  # default True (vacuous for empty)
        for i in range(n):
            d = f[refs[i]]
            n_pix = d.shape[1] if d.ndim == 2 else 0
            if n_pix == 0:
                continue
            visible[i] = n_pix > pixel_threshold
            if visible[i]:
                single[i], bbox = _blob_stats(d[:], blob_ratio, blob_distance)
                r_lo, r_hi, c_lo, c_hi = bbox
                # 1-based coords: the image spans rows 1..height, cols 1..width.
                margin = min(r_lo - 1, height - r_hi,
                             c_lo - 1, width - c_hi)
                in_frame[i] = margin >= min_edge_margin
    return visible, single, in_frame


def scan_movie(movie_dir: str, pixel_threshold: int,
               blob_ratio: float = 0.30,
               blob_distance: float = 100.0,
               min_edge_margin: float = DEFAULT_MIN_EDGE_MARGIN) -> dict:
    """Scan all 4 cam sparse mats. Returns a dict with per-cam visibility,
    multi-blob and out-of-frame counts, plus the longest contiguous run of
    frames that satisfy ALL THREE conditions (visible AND single-fly AND
    fully in frame) across all 4 cams. On failure returns {'error': str}."""
    mats = sorted(glob.glob(os.path.join(movie_dir, "*_sparse.mat")))
    if len(mats) != 4:
        return {"error": f"expected 4 *_sparse.mat, found {len(mats)}"}
    visible_masks, single_masks, in_frame_masks, lengths = [], [], [], []
    for p in mats:
        try:
            vis, sgl, inf = analyze_cam_frames(p, pixel_threshold,
                                               blob_ratio, blob_distance,
                                               min_edge_margin)
        except Exception as e:
            return {"error": f"failed reading {os.path.basename(p)}: {e}"}
        lengths.append(len(vis))
        visible_masks.append(vis)
        single_masks.append(sgl)
        in_frame_masks.append(inf)
    # Align to shortest (defensive — should all match)
    n_min = min(lengths)
    visible_masks = [v[:n_min] for v in visible_masks]
    single_masks = [s[:n_min] for s in single_masks]
    in_frame_masks = [i[:n_min] for i in in_frame_masks]
    visible_all = np.logical_and.reduce(visible_masks)
    single_all = np.logical_and.reduce(single_masks)
    in_frame_all = np.logical_and.reduce(in_frame_masks)
    intersection = visible_all & single_all & in_frame_all
    # Longest contiguous run of "visible AND single-fly AND fully in frame,
    # in all cams"
    pad = np.r_[0, intersection.astype(np.int8), 0]
    diff = np.diff(pad)
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]   # exclusive
    if len(run_starts):
        lengths_run = run_ends - run_starts
        best = int(np.argmax(lengths_run))
        good_start = int(run_starts[best])
        good_end = int(run_ends[best])
    else:
        good_start = good_end = 0
    return {
        "n_frames": n_min,
        "per_cam_visible_counts": [int(v.sum()) for v in visible_masks],
        # A frame is "multi-blob" when it's visible but NOT single-fly.
        "per_cam_multi_blob_counts": [int((v & ~s).sum())
                                      for v, s in zip(visible_masks, single_masks)],
        # ...and "out of frame" when it's visible but the blob runs into a
        # border. Counted only over visible frames so it reads as "of the
        # frames this cam saw the fly in, how many showed only part of it".
        "per_cam_out_of_frame_counts": [int((v & ~i).sum())
                                        for v, i in zip(visible_masks,
                                                        in_frame_masks)],
        "intersection_count": int(intersection.sum()),
        "good_start": good_start,
        "good_end": good_end,
        "good_run_length": good_end - good_start,
        "mats": mats,
    }


def parse_movie_num(d: str):
    # Case-insensitive and leading-zero tolerant (mov3, Mov003, MOV03 -> 3).
    m = re.match(r"^mov(\d+)$", os.path.basename(d), re.IGNORECASE)
    return int(m.group(1)) if m else None


def detect_mode(input_dir: str) -> tuple:
    n_direct = len(glob.glob(os.path.join(input_dir, "*_sparse.mat")))
    if n_direct == 4:
        mn = parse_movie_num(input_dir)
        if mn is None:
            sys.exit(f"Single-movie mode requires 'mov<N>' basename; got "
                     f"'{os.path.basename(input_dir)}'")
        return "single", [(input_dir, mn)]
    if n_direct != 0:
        sys.exit(f"Ambiguous: {input_dir} contains {n_direct} *_sparse.mat "
                 f"(expected 0 or 4)")
    # os.listdir (not a case-sensitive glob) so 'Mov001' Windows exports are
    # discovered alongside canonical 'mov1'.
    movies = []
    for name in sorted(os.listdir(input_dir)):
        sub = os.path.join(input_dir, name)
        if not os.path.isdir(sub):
            continue
        mn = parse_movie_num(sub)
        if mn is None:
            continue
        if len(glob.glob(os.path.join(sub, "*_sparse.mat"))) == 4:
            movies.append((sub, mn))
    movies.sort(key=lambda t: t[1])
    if not movies:
        sys.exit(f"No 'mov<N>/' subdirs with 4 *_sparse.mat in {input_dir}")
    return "multi", movies


def scan_experiment(input_dir: str, min_intersection: int = 500,
                    pixel_threshold: int = 50,
                    blob_ratio: float = 0.30,
                    blob_distance: float = 100.0,
                    print_results: bool = True,
                    min_edge_margin: float = DEFAULT_MIN_EDGE_MARGIN) -> list:
    """Scans all movies under input_dir. Returns list of result dicts:
        {'movie_dir': str, 'movie_num': int,
         'verdict': 'OK'|'BAD'|'ERR',
         'intersection': int,                  # total frames passing all 3 tests
         'good_start': int, 'good_end': int,   # longest contiguous run
         'good_run_length': int,
         'n_frames': int,
         'per_cam_visible': [int]*4,
         'per_cam_multi_blob': [int]*4,        # # of frames flagged 2+ flies
         'per_cam_out_of_frame': [int]*4,      # # of frames with a cut fly
         'error': str | None}
    """
    mode, movies = detect_mode(input_dir)
    if print_results:
        print(f"Mode: {mode}; scanning {len(movies)} movie(s)")
        print(f"  thresholds: longest_run >= {min_intersection}, "
              f"pixel_count > {pixel_threshold}, "
              f"blob_ratio >= {blob_ratio:.2f}, "
              f"blob_distance >= {blob_distance:.0f} px, "
              f"edge_margin >= {min_edge_margin:.0f} px")
    results = []
    for movie_dir, mn in movies:
        info = scan_movie(movie_dir, pixel_threshold, blob_ratio,
                          blob_distance, min_edge_margin)
        if "error" in info:
            results.append({"movie_dir": movie_dir, "movie_num": mn,
                            "verdict": "ERR", "error": info["error"]})
            if print_results:
                print(f"  [mov{mn:3}] ERR  {info['error']}")
            continue
        # Verdict is based on the longest contiguous run — that's what we'll
        # actually build, so it's the right metric for "is this movie worth
        # processing". (Total intersection_count may be larger if there are
        # gaps in the middle, but we wouldn't use those scattered frames.)
        ok = info["good_run_length"] >= min_intersection
        results.append({
            "movie_dir": movie_dir, "movie_num": mn,
            "verdict": "OK" if ok else "BAD",
            "intersection": info["intersection_count"],
            "good_start": info["good_start"],
            "good_end": info["good_end"],
            "good_run_length": info["good_run_length"],
            "n_frames": info["n_frames"],
            "per_cam_visible": info["per_cam_visible_counts"],
            "per_cam_multi_blob": info["per_cam_multi_blob_counts"],
            "per_cam_out_of_frame": info["per_cam_out_of_frame_counts"],
            "error": None,
        })
        if print_results:
            vis = info["per_cam_visible_counts"]
            mb = info["per_cam_multi_blob_counts"]
            oof = info["per_cam_out_of_frame_counts"]
            vis_str = ", ".join(f"cam{i+1}={c}" for i, c in enumerate(vis))
            mb_str = ", ".join(f"cam{i+1}={c}" for i, c in enumerate(mb))
            oof_str = ", ".join(f"cam{i+1}={c}" for i, c in enumerate(oof))
            print(f"  [mov{mn:3}] {'OK ' if ok else 'BAD'}  "
                  f"run=[{info['good_start']:4}, {info['good_end']:5}) "
                  f"({info['good_run_length']:5}/{info['n_frames']})")
            print(f"          visible:      {vis_str}")
            print(f"          multi-blob:   {mb_str}")
            print(f"          out-of-frame: {oof_str}")
    if print_results:
        ok_n = sum(1 for r in results if r["verdict"] == "OK")
        bad_n = sum(1 for r in results if r["verdict"] == "BAD")
        err_n = sum(1 for r in results if r["verdict"] == "ERR")
        print(f"\nSummary: {ok_n} OK, {bad_n} BAD, {err_n} ERR "
              f"(of {len(results)} total)")
        if bad_n:
            print("\nBAD movies (skip from build/predict):")
            for r in results:
                if r["verdict"] == "BAD":
                    print(f"  mov{r['movie_num']}  "
                          f"longest_run={r['good_run_length']}/{r['n_frames']}")
    return results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_dir",
                    help="single mov<N>/ dir, or experiment dir with mov*/ subdirs")
    ap.add_argument("--min-intersection", type=int, default=500,
                    help="all-cams intersection below this = BAD (default: 500)")
    ap.add_argument("--pixel-threshold", type=int, default=50,
                    help="per-frame non-zero pixel count for 'fly visible' "
                         "(default: 50)")
    ap.add_argument("--blob-ratio", type=float, default=0.30,
                    help="a 2nd blob is considered a separate fly if its size "
                         "is at least this fraction of the largest blob's "
                         "size (default: 0.30)")
    ap.add_argument("--blob-distance", type=float, default=100.0,
                    help="a 2nd blob is considered a separate fly if its "
                         "centroid is at least this many pixels from the "
                         "largest blob's centroid (default: 100)")
    ap.add_argument("--min-edge-margin", type=float,
                    default=DEFAULT_MIN_EDGE_MARGIN,
                    help="the fly's blob must stay at least this many pixels "
                         "clear of every image border; frames where it "
                         "doesn't show a truncated fly and are excluded "
                         f"(default: {DEFAULT_MIN_EDGE_MARGIN}, 0 disables)")
    args = ap.parse_args()
    if not os.path.isdir(args.input_dir):
        sys.exit(f"input_dir is not a directory: {args.input_dir}")
    results = scan_experiment(args.input_dir, args.min_intersection,
                              args.pixel_threshold,
                              args.blob_ratio, args.blob_distance,
                              print_results=True,
                              min_edge_margin=args.min_edge_margin)
    bad_n = sum(1 for r in results if r["verdict"] != "OK")
    sys.exit(1 if bad_n else 0)


if __name__ == "__main__":
    main()
