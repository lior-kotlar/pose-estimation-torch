"""Rebuild movies whose old build range ran past the point the fly left frame.

The prescan gained an in-frame test (see scan_sparse_movies) after movies were
found holding a truncated -- then absent -- fly for their last few hundred
frames. Movies built before that carry the bad tail baked into their h5, and
nothing downstream can tell: the reprojection VERIFY passes on them (its
all-cams test only fires on `cropzone == [1,1]`, which MATLAB writes solely
when it finds NO blob at all).

For each movie in the manifest this:
  1. re-runs the fixed prescan on the raw *_sparse.mat to get the new range;
  2. skips the movie if the range is unchanged, or if the new run falls under
     the min-intersection floor (that movie should be dropped, not rebuilt);
  3. moves the old h5 into `<batch>/_superseded_builds/mov<N>/`;
  4. rebuilds in place at the new range via process_experiment.build_one_movie.

Two things it deliberately does NOT do:

  * FLIP. The mirror cam's mat was already flipped in place when the movie was
    first processed and the flip is not idempotent -- running it again would
    silently un-flip the camera.
  * build anywhere other than the movie's own directory. The h5's location is
    what predict.py records as `experiment` / `source_movie_dir` / `box_h5`,
    so building into a scratch tree makes the provenance in every downstream
    analysis h5 a lie.

The archive lives OUTSIDE the movie dir on purpose: predict.py's
configure_movie_list scans the data directory *and its immediate
subdirectories* for `mov*.h5`, so an archive inside `mov<N>/` would be picked
up and predicted as a second movie.

usage:
    .env/bin/python code/rebuild_edge_cut_movies.py <manifest> [--dry-run]
      [--min-edge-margin N] [--min-intersection N]

The manifest is one movie directory per line (repo-relative or absolute).
"""

import argparse
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process_experiment import (  # noqa: E402
    DEFAULT_MIN_INTERSECTION,
    MATLAB_TIME_JUMP_MARGIN,
    build_one_movie,
    find_movie_h5,
)
from scan_sparse_movies import DEFAULT_MIN_EDGE_MARGIN, scan_movie  # noqa: E402

ARCHIVE_DIRNAME = "_superseded_builds"


def parse_range(h5_path):
    """(start_ind, end_ind) encoded in a built h5's filename, or None."""
    m = re.match(r"mov_(\d+)_(\d+)_(\d+)_ds_", os.path.basename(h5_path))
    return (int(m.group(2)), int(m.group(3))) if m else None


def movie_num(movie_dir):
    m = re.match(r"^mov(\d+)$", os.path.basename(movie_dir), re.IGNORECASE)
    return int(m.group(1)) if m else None


def new_range(info):
    """The prescan's good run in MATLAB's 1-based inclusive convention, with
    the same time-jump clamping run_prescan applies."""
    start_ind = max(info["good_start"] + 1, MATLAB_TIME_JUMP_MARGIN + 1)
    end_ind = min(info["good_end"], info["n_frames"] - MATLAB_TIME_JUMP_MARGIN)
    return start_ind, end_ind


def archive_old_build(movie_dir, h5_path, dry_run):
    """Move the superseded h5 to <batch>/_superseded_builds/mov<N>/."""
    batch_dir = os.path.dirname(movie_dir)
    dest_dir = os.path.join(batch_dir, ARCHIVE_DIRNAME,
                            os.path.basename(movie_dir))
    dest = os.path.join(dest_dir, os.path.basename(h5_path))
    if dry_run:
        print(f"   would archive -> {dest}")
        return
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest):
        print(f"   archive already holds {os.path.basename(dest)}; "
              f"leaving it and removing the live copy")
        os.remove(h5_path)
    else:
        shutil.move(h5_path, dest)
        print(f"   archived -> {dest}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="one movie directory per line")
    ap.add_argument("--min-edge-margin", type=float,
                    default=DEFAULT_MIN_EDGE_MARGIN)
    ap.add_argument("--min-intersection", type=int,
                    default=DEFAULT_MIN_INTERSECTION)
    ap.add_argument("--pixel-threshold", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true",
                    help="prescan and report the new ranges, build nothing")
    args = ap.parse_args()

    with open(args.manifest) as f:
        movie_dirs = [line.strip().rstrip("/") for line in f
                      if line.strip() and not line.startswith("#")]

    print(f"{len(movie_dirs)} movie(s); edge margin {args.min_edge_margin:.0f} px, "
          f"floor {args.min_intersection} frames"
          + ("   [DRY RUN]" if args.dry_run else ""))

    rebuilt, unchanged, dropped, failed = [], [], [], []
    for movie_dir in movie_dirs:
        mn = movie_num(movie_dir)
        print(f"\n== {movie_dir}")
        if mn is None:
            print("   not a mov<N> directory; skipping")
            failed.append(movie_dir)
            continue
        old_h5 = find_movie_h5(movie_dir)
        old = parse_range(old_h5) if old_h5 else None
        info = scan_movie(movie_dir, args.pixel_threshold,
                          min_edge_margin=args.min_edge_margin)
        if "error" in info:
            print(f"   prescan failed: {info['error']}")
            failed.append(movie_dir)
            continue
        start_ind, end_ind = new_range(info)
        n_new = end_ind - start_ind + 1
        print(f"   old range {old}  ->  new [{start_ind}, {end_ind}]  "
              f"({n_new} frames of {info['n_frames']})")
        print(f"   out-of-frame per cam: {info['per_cam_out_of_frame_counts']}")

        if info["good_run_length"] < args.min_intersection or n_new <= 0:
            print(f"   new run is under the {args.min_intersection}-frame "
                  f"floor; this movie should be DROPPED, not rebuilt")
            dropped.append(movie_dir)
            continue
        if old == (start_ind, end_ind):
            print("   range unchanged; nothing to do")
            unchanged.append(movie_dir)
            continue
        if old_h5:
            archive_old_build(movie_dir, old_h5, args.dry_run)
        if args.dry_run:
            print(f"   would build [{start_ind}, {end_ind}]")
            rebuilt.append(movie_dir)
            continue
        rc, n_built = build_one_movie(
            sparse_folder_path=os.path.dirname(movie_dir),
            movie_dir=movie_dir, movie_num=mn, max_frames=None,
            start_ind=start_ind, end_ind=end_ind)
        if rc == 0:
            print(f"   built {n_built} frames")
            rebuilt.append(movie_dir)
        else:
            print(f"   BUILD FAILED (rc={rc}); see build_mov{mn}.log")
            failed.append(movie_dir)

    print(f"\n{'=' * 70}")
    print(f"rebuilt   {len(rebuilt)}")
    print(f"unchanged {len(unchanged)}" + (f"  {unchanged}" if unchanged else ""))
    print(f"dropped   {len(dropped)}" + (f"  {dropped}" if dropped else ""))
    print(f"failed    {len(failed)}" + (f"  {failed}" if failed else ""))
    if rebuilt and not args.dry_run:
        out = os.path.splitext(args.manifest)[0] + "_rebuilt.txt"
        with open(out, "w") as f:
            for d in rebuilt:
                f.write(d + "\n")
        print(f"\npredict manifest ({len(rebuilt)} movies): {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
