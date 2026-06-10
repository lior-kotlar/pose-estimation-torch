"""
Collect every *analysis_smoothed.h5 under a run directory into one folder.

Walks <src_dir> recursively (e.g. predict_output/debug_outputs, whose
immediate subdirs are per-movie run folders) and copies every file ending in
`analysis_smoothed.h5` into <dest_dir> (created if missing).

Usage:
    python code/collect_analysis_h5.py <src_dir> <dest_dir> [--move] [--dry-run]

Examples:
    python code/collect_analysis_h5.py predict_output/debug_outputs collected_h5
    python code/collect_analysis_h5.py predict_output/101to110 /tmp/h5 --dry-run

Name collisions: if two source files share a basename, the second (and later)
copies are renamed <stem>__<parent_dir><ext> using their parent directory name
to disambiguate, so nothing is silently overwritten.
"""
import argparse
import os
import shutil
import sys

SUFFIX = "analysis_smoothed.h5"


def find_files(src_dir):
    """Yield absolute paths of every file ending in SUFFIX under src_dir."""
    for root, _dirs, files in os.walk(src_dir):
        for name in files:
            if name.endswith(SUFFIX):
                yield os.path.join(root, name)


def unique_dest(dest_dir, src_path, taken):
    """Pick a collision-free destination path for src_path inside dest_dir."""
    base = os.path.basename(src_path)
    candidate = os.path.join(dest_dir, base)
    if candidate not in taken and not os.path.exists(candidate):
        return candidate
    # disambiguate with the immediate parent directory name
    parent = os.path.basename(os.path.dirname(src_path))
    stem, ext = os.path.splitext(base)
    candidate = os.path.join(dest_dir, f"{stem}__{parent}{ext}")
    n = 1
    while candidate in taken or os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{stem}__{parent}_{n}{ext}")
        n += 1
    return candidate


def main():
    p = argparse.ArgumentParser(
        description=f"Collect every *{SUFFIX} under a directory tree into one folder.")
    p.add_argument("src_dir", help="root directory to search recursively")
    p.add_argument("dest_dir", help="directory to copy the matching files into")
    p.add_argument("--move", action="store_true",
                   help="move instead of copy (removes the originals)")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would happen without touching the filesystem")
    args = p.parse_args()

    if not os.path.isdir(args.src_dir):
        sys.exit(f"src_dir is not a directory: {args.src_dir}")
    if os.path.abspath(args.src_dir) == os.path.abspath(args.dest_dir):
        sys.exit("src_dir and dest_dir must differ")

    matches = sorted(find_files(args.src_dir))
    if not matches:
        print(f"No *{SUFFIX} files found under {args.src_dir}")
        return

    if not args.dry_run:
        os.makedirs(args.dest_dir, exist_ok=True)

    verb = "move" if args.move else "copy"
    taken = set()
    n_done = 0
    for src in matches:
        dest = unique_dest(args.dest_dir, src, taken)
        taken.add(dest)
        print(f"{verb}: {src} -> {dest}")
        if args.dry_run:
            continue
        if args.move:
            shutil.move(src, dest)
        else:
            shutil.copy2(src, dest)
        n_done += 1

    if args.dry_run:
        print(f"\n[dry-run] {len(matches)} file(s) would be {verb}d into {args.dest_dir}")
    else:
        print(f"\n{verb.capitalize()}d {n_done} file(s) into {args.dest_dir}")


if __name__ == "__main__":
    main()
