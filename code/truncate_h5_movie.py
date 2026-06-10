"""
Truncate an H5 movie file to the first N frames and save as a new file.

Usage:
    python truncate_h5_movie.py <input.h5> <N> [-o <output.h5>]

Examples:
    python truncate_h5_movie.py path/to/mov_1_8_5505_ds_3tc_7tj.h5 1500
    python truncate_h5_movie.py path/to/movie.h5 400 -o path/to/movie_short.h5

If -o is omitted, the output is saved alongside the input with a "_firstN" suffix.
"""

import argparse
import os
import sys

import h5py


# datasets whose first axis is frames
FRAME_FIRST_KEYS = ('box', 'cropzone', 'frameInds')
# datasets whose LAST axis is frames
FRAME_LAST_KEYS = ('best_frames_mov_idx',)


def truncate(src_path, dst_path, n):
    if not os.path.isfile(src_path):
        sys.exit(f"Input file not found: {src_path}")
    if os.path.exists(dst_path):
        sys.exit(f"Output file already exists: {dst_path}")

    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        for key in src.keys():
            ds = src[key]
            if key in FRAME_FIRST_KEYS:
                total = ds.shape[0]
                take = min(n, total)
                data = ds[:take]
            elif key in FRAME_LAST_KEYS:
                total = ds.shape[-1]
                take = min(n, total)
                data = ds[..., :take]
            else:
                data = ds[()]
                take = None

            dst.create_dataset(
                key, data=data,
                compression='gzip', compression_opts=1
            )
            shape_str = str(data.shape) if hasattr(data, 'shape') else 'scalar'
            note = f' (truncated from {total} → {take})' if take is not None else ''
            print(f"  {key}: {shape_str}{note}")

        for k, v in src.attrs.items():
            dst.attrs[k] = v

    src_mb = os.path.getsize(src_path) / 1024**2
    dst_mb = os.path.getsize(dst_path) / 1024**2
    print(f"\nDone. {src_path} ({src_mb:.0f} MB) → {dst_path} ({dst_mb:.0f} MB)")


def main():
    p = argparse.ArgumentParser(description="Truncate an H5 movie file to the first N frames.")
    p.add_argument('input', help='Path to the source H5 file')
    p.add_argument('n', type=int, help='Number of frames to keep')
    p.add_argument('-o', '--output', default=None, help='Output path (default: <input>_first<N>.h5)')
    args = p.parse_args()

    if args.output is None:
        root, ext = os.path.splitext(args.input)
        args.output = f"{root}_first{args.n}{ext}"

    truncate(args.input, args.output, args.n)


if __name__ == '__main__':
    main()
