"""
Vertically flip a sparse-format camera .mat file in place, faster than the
MATLAB version. Operates directly on the v7.3 HDF5 layout via h5py.

Usage:

    # single file:
    .env/bin/python code/flip_sparse_cam_mat.py <path/to/movN_camK_sparse.mat>

    # batch over many movies (e.g. flip cam1 in every movN/ subdir):
    .env/bin/python code/flip_sparse_cam_mat.py <movies_dir> --cam cam1
    .env/bin/python code/flip_sparse_cam_mat.py <movies_dir> --cam cam1 --dry-run

Batch-mode semantics:
- <movies_dir> is expected to contain `mov*/` subdirectories.
- For each subdir, find files matching `*<cam>_sparse.mat`. If exactly one
  match is found, it is flipped. Subdirs with zero or multiple matches are
  skipped with a warning.
- Failures on individual files are reported but do not abort the batch.

Notes:
- Only works on v7.3 MATLAB files (files saved with `save(..., '-v7.3')`).
  These are HDF5 under the hood. If the file is older v7 format, h5py will
  fail to open it; use the MATLAB version `matlab/flip_sparse_cam_mat.m`.
- The flip is performed in-place; the original file is overwritten. Running
  the script twice on the same file undoes the flip — there is no idempotency
  marker. Track manually which files have been flipped.
"""

import argparse
import glob
import os
import sys
import traceback

import h5py


def flip_sparse_cam_mat(mat_path: str) -> bool:
    """Flip a single sparse .mat in place. Returns True on success, False on
    failure. Prints progress / errors to stdout."""
    if not os.path.isfile(mat_path):
        print(f"  ERROR: file not found: {mat_path}")
        return False

    try:
        with h5py.File(mat_path, "r+") as f:
            if "metaData" not in f or "frames" not in f:
                print(f"  ERROR: missing 'metaData' or 'frames' in {mat_path}; "
                      f"not a sparse cam .mat file.")
                return False

            # Determine frame height H from metaData.frameSize (= [H, W]).
            frame_size = f["metaData/frameSize"][()].flatten()
            H = int(frame_size[0])
            if H == 0:
                print(f"  ERROR: frameSize parsing failed; got {frame_size}")
                return False

            # Phase 1: read all per-frame indIm into memory, compute flipped
            # versions. If anything errors here, the file is still untouched.
            refs = f["frames/indIm"][()]   # shape (1, N_frames), dtype=object refs
            refs_flat = refs.flatten()
            n_frames = refs_flat.size
            new_data_per_frame = [None] * n_frames
            n_to_flip = 0
            for i, ref in enumerate(refs_flat):
                ind = f[ref]
                data = ind[()]
                # Real per-frame indIm is (3, N_pixels). Placeholders may be
                # 1-D / 0-element — leave them alone.
                if data.ndim != 2 or data.shape[0] != 3 or data.shape[1] == 0:
                    continue
                data[0, :] = (H + 1) - data[0, :]
                new_data_per_frame[i] = data
                n_to_flip += 1

            # Compute flipped bg.
            bg = f["metaData/bg"][()]
            bg_flipped = bg[:, ::-1]

            # Phase 2: commit writes. Past the heavy validation now.
            for i, new_data in enumerate(new_data_per_frame):
                if new_data is None:
                    continue
                f[refs_flat[i]][...] = new_data
            f["metaData/bg"][...] = bg_flipped

            print(f"  flipped {n_to_flip}/{n_frames} frames + bg (H={H})")
            return True

    except Exception as e:
        print(f"  ERROR during flip: {e}")
        traceback.print_exc()
        return False


def find_target_in_movie_dir(movie_dir: str, cam: str) -> "str | None":
    """Find the single *<cam>_sparse.mat in `movie_dir`. Returns the path,
    or None on zero/multiple matches (a warning is printed)."""
    pattern = os.path.join(movie_dir, f"*{cam}_sparse.mat")
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        print(f"  skip: no '*{cam}_sparse.mat' in {movie_dir}")
        return None
    if len(matches) > 1:
        print(f"  skip: multiple '*{cam}_sparse.mat' matches in {movie_dir}: "
              f"{[os.path.basename(m) for m in matches]}")
        return None
    return matches[0]


def run_batch(movies_dir: str, cam: str, dry_run: bool) -> None:
    """Process every mov*/ subdir under movies_dir, flipping the matching cam."""
    subdirs = sorted(
        d for d in glob.glob(os.path.join(movies_dir, "mov*"))
        if os.path.isdir(d)
    )
    if not subdirs:
        sys.exit(f"No 'mov*/' subdirectories found in {movies_dir}")

    print(f"Found {len(subdirs)} movie subdirectories; cam pattern: '*{cam}_sparse.mat'")
    if dry_run:
        print("(dry-run mode: no files will be modified)\n")

    n_ok = 0
    n_fail = 0
    n_skip = 0
    failed_movs = []
    for sub in subdirs:
        mov_name = os.path.basename(sub)
        target = find_target_in_movie_dir(sub, cam)
        if target is None:
            n_skip += 1
            continue
        if dry_run:
            print(f"[{mov_name}] would flip: {target}")
            n_ok += 1
            continue
        print(f"[{mov_name}] flipping: {os.path.basename(target)}")
        ok = flip_sparse_cam_mat(target)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failed_movs.append(mov_name)

    print(f"\nSummary: {n_ok} flipped, {n_fail} failed, {n_skip} skipped "
          f"(of {len(subdirs)} subdirs)")
    if failed_movs:
        print(f"Failed in: {', '.join(failed_movs)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", help="single .mat file path, OR a movies/ directory "
                                "containing mov*/ subdirs")
    p.add_argument("--cam", default=None,
                   help="cam-name substring to match in batch mode "
                        "(e.g. 'cam1'). Required when `path` is a directory; "
                        "ignored when `path` is a file.")
    p.add_argument("--dry-run", action="store_true",
                   help="batch mode only: list what would be flipped, do not "
                        "modify any files.")
    args = p.parse_args()

    if os.path.isfile(args.path):
        ok = flip_sparse_cam_mat(args.path)
        sys.exit(0 if ok else 1)

    if os.path.isdir(args.path):
        if not args.cam:
            sys.exit("--cam is required when `path` is a directory "
                     "(e.g. --cam cam1)")
        run_batch(args.path, args.cam, args.dry_run)
        return

    sys.exit(f"Path is neither a file nor a directory: {args.path}")


if __name__ == "__main__":
    main()
