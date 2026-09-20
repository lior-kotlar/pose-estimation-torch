"""
Collect every *analysis_smoothed.h5 under a directory tree into one folder, one
subfolder per experiment.

Walks <src_dir> recursively (e.g. predict_output, or one experiment's folder)
and copies every file ending in `analysis_smoothed.h5` into
<dest_dir>/<experiment>/, where <experiment> is the experiment's path under
inference_datasets -- e.g. Tsory/ex210825_dark_yaw_t0, roni_dark/2023_08_06_40ms.
It is read from the movie's own provenance (the h5, else source.json, else a
saved member config), with build batch folders such as `1to30` dropped, so the
same movie lands in the same place no matter where or under what folder names
its predictions are stored. A movie that records no provenance at all goes under
local_only/<the experiment folder holding it>. --flat puts everything in one
folder instead.

A file already in <dest_dir> is replaced only when the new one differs, and the
old version is moved into superseded_<timestamp>/ next to it rather than
overwritten; an identical file is left alone. So collecting the same tree again
after a re-analysis updates exactly the movies that changed.

Directories named `bad_wings` or `bad_signal` (and their subtrees) are skipped
by default, so known-bad movies don't contaminate the collected set. Override
with --exclude / --include-bad.

Archives are ALWAYS skipped, whatever the flags: `superseded_*` (where
reanalyse_movies.py and realign_ensemble.py move the products they replace) and
hidden directories such as `.realign_staging`. They hold earlier versions of the
same movies' h5.

The same script does the collect step off the cluster: local_reanalysis.py calls
it after re-analysing movies on a PC, then uploads the result (see
LOCAL_REANALYSIS.md).

Usage:
    python code/collect_analysis_h5.py <src_dir> <dest_dir> [--flat] [--move] [--dry-run]
                                       [--exclude DIR ...] [--include-bad]

Examples:
    python code/collect_analysis_h5.py predict_output collected_h5
    python code/collect_analysis_h5.py predict_output/ex210825_dark_yaw_t0 collected_h5 --dry-run

Name collisions: two movies of one experiment with the same file name in a
single run (a movie kept in both <exp>/ and <exp>/fixed/, say) are a warning;
the second copy is renamed <stem>__<parent_dir><ext> so nothing is lost.
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import shutil
import sys

import h5py

SUFFIX = "analysis_smoothed.h5"

# Directories holding movies that must never be collected (bad/contaminating
# data). Pruned during the walk so os.walk never descends into them.
DEFAULT_EXCLUDE_DIRS = {"bad_wings", "bad_signal"}

# A build batch inside an experiment (inference_datasets/roni_dark/2023_08_06_40ms/1to30):
# batches are how one experiment was split for building, not separate experiments.
BATCH_DIR = re.compile(r"^\d+to\d+$")
LOCAL_ONLY = "local_only"
# Subfolders movies are sorted into inside one experiment folder; never experiments themselves.
WORKFLOW_DIRS = {"bad_signal", "bad_wings", "fixed", "run_whole_pipeline_again", "maybe_cyclic_issue"}


def is_archive_dir(name):
    """superseded_<stamp>/ and dot-dirs hold replaced copies of a movie's products."""
    return name.startswith("superseded_") or name.startswith(".")


def find_files(src_dir, exclude_dirs):
    """Yield absolute paths of every file ending in SUFFIX under src_dir,
    skipping any directory whose name is in exclude_dirs (and its subtree)."""
    skipped = []
    for root, dirs, files in os.walk(src_dir):
        # prune excluded dirs in place so os.walk won't descend into them
        pruned = [d for d in dirs if d in exclude_dirs]
        for d in pruned:
            skipped.append(os.path.join(root, d))
        dirs[:] = sorted(d for d in dirs if d not in exclude_dirs and not is_archive_dir(d))
        for name in sorted(files):
            if name.endswith(SUFFIX):
                yield os.path.join(root, name)
    for s in skipped:
        print(f"  (excluded subtree: {s})")


def normalise_experiment(path):
    """An experiment path as a/b/c: separators unified, batch folders and '..' dropped."""
    parts = [p for p in str(path).replace("\\", "/").split("/") if p not in ("", ".", "..")]
    while parts and BATCH_DIR.match(parts[-1]):
        parts.pop()
    return "/".join(parts) or None


def experiment_from_movie_path(movie_path):
    """inference_datasets/<experiment>/<movN>/<box>.h5 -> <experiment>, or None."""
    parts = str(movie_path).replace("\\", "/").split("/")
    if "inference_datasets" not in parts:
        return None
    start = len(parts) - parts[::-1].index("inference_datasets")
    return normalise_experiment("/".join(parts[start:-2]))


def _text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def recorded_experiment(movie_dir):
    """The experiment a movie's own records name, or None when it records nothing usable."""
    for h5_path in sorted(glob.glob(os.path.join(movie_dir, f"*_{SUFFIX}"))):
        try:
            with h5py.File(h5_path, "r") as hdf:
                if "experiment" in hdf:
                    found = normalise_experiment(_text(hdf["experiment"][()]))
                    if found:
                        return found
                if "box_h5" in hdf:
                    found = experiment_from_movie_path(_text(hdf["box_h5"][()]))
                    if found:
                        return found
        except OSError:
            pass
    try:
        with open(os.path.join(movie_dir, "source.json"), encoding="utf-8") as f:
            source = json.load(f)
        found = (normalise_experiment(source["experiment"]) if source.get("experiment") else None) \
            or (experiment_from_movie_path(source["box_h5"]) if source.get("box_h5") else None)
        if found:
            return found
    except (OSError, ValueError, AttributeError):
        pass
    for pattern in ("*/specific_configuration.json", "*/configuration.json"):
        for cfg_path in sorted(glob.glob(os.path.join(movie_dir, pattern))):
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    movie_path = json.load(f).get("movie path")
            except (OSError, ValueError, AttributeError):
                continue
            found = experiment_from_movie_path(movie_path) if movie_path else None
            if found:
                return found
    return None


def experiment_key(movie_dir, root=None):
    """Where a movie's h5 goes under the collection: its recorded experiment, else
    local_only/<the experiment folder holding it>.

    That folder is the movie folder's parent, stepping out of the subfolders a lab workflow
    sorts movies into inside an experiment (fixed/, bad_signal/, ...). Deliberately not
    relative to the folder a run was pointed at, which may sit any number of levels higher."""
    found = recorded_experiment(movie_dir)
    if found:
        return found
    folder = os.path.dirname(os.path.abspath(movie_dir))
    while os.path.basename(folder) in WORKFLOW_DIRS and os.path.dirname(folder) != folder:
        folder = os.path.dirname(folder)
    return f"{LOCAL_ONLY}/{os.path.basename(folder) or 'unnamed'}"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(sources, dest_dir, root=None, flat=False, move=False, dry_run=False, stamp=None,
            keys=None):
    """Copy (or move) each source h5 into dest_dir; returns one dict per file.

    Each file goes under its experiment_key relative to root, unless keys (source -> experiment)
    already names it. status is 'new', 'replaced' (the previous version moved to
    superseded_<stamp>/), 'unchanged' (identical file already there) or 'renamed' (a same-name
    collision inside this run)."""
    stamp = stamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    taken = set()
    results = []
    for src in sources:
        if flat:
            key = ""
        elif keys and src in keys:
            key = keys[src]
        else:
            key = experiment_key(os.path.dirname(src), root)
        dest = os.path.join(dest_dir, *key.split("/"), os.path.basename(src)) if key \
            else os.path.join(dest_dir, os.path.basename(src))
        status = None
        if dest in taken:
            # the same file name twice in one run: never let the second archive the first
            stem, ext = os.path.splitext(os.path.basename(src))
            parent = os.path.basename(os.path.dirname(os.path.dirname(src)))
            dest = os.path.join(os.path.dirname(dest), f"{stem}__{parent}{ext}")
            n = 1
            while dest in taken:
                dest = os.path.join(os.path.dirname(dest), f"{stem}__{parent}_{n}{ext}")
                n += 1
            status = "renamed"
            print(f"WARNING: two movies named {os.path.basename(src)} in {key or dest_dir}; "
                  f"this one is collected as {os.path.basename(dest)}")
        taken.add(dest)
        if os.path.exists(dest):
            if sha256(dest) == sha256(src):
                results.append({"src": src, "dest": dest, "experiment": key, "status": "unchanged"})
                if move and not dry_run:
                    os.remove(src)
                continue
            status = status or "replaced"
            if not dry_run:
                archive = os.path.join(os.path.dirname(dest), f"superseded_{stamp}")
                os.makedirs(archive, exist_ok=True)
                shutil.move(dest, os.path.join(archive, os.path.basename(dest)))
        status = status or "new"
        if not dry_run:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if move:
                shutil.move(src, dest)
            else:
                shutil.copy2(src, dest)
        results.append({"src": src, "dest": dest, "experiment": key, "status": status})
    return results


def main():
    p = argparse.ArgumentParser(
        description=f"Collect every *{SUFFIX} under a directory tree, one folder per experiment.")
    p.add_argument("src_dir", help="root directory to search recursively")
    p.add_argument("dest_dir", help="directory to collect the matching files into")
    p.add_argument("--flat", action="store_true",
                   help="put every file directly in dest_dir instead of <experiment>/ subfolders")
    p.add_argument("--move", action="store_true",
                   help="move instead of copy (removes the originals)")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would happen without touching the filesystem")
    p.add_argument("--exclude", action="append", default=None, metavar="DIR",
                   help=f"directory name to skip (repeatable). "
                        f"Default: {' '.join(sorted(DEFAULT_EXCLUDE_DIRS))}")
    p.add_argument("--include-bad", action="store_true",
                   help="disable all exclusions and collect everything (DANGER: "
                        "pulls in bad_wings/bad_signal too)")
    args = p.parse_args()

    if not os.path.isdir(args.src_dir):
        sys.exit(f"src_dir is not a directory: {args.src_dir}")
    if os.path.abspath(args.src_dir) == os.path.abspath(args.dest_dir):
        sys.exit("src_dir and dest_dir must differ")

    if args.include_bad:
        exclude_dirs = set()
    else:
        exclude_dirs = set(args.exclude) if args.exclude else set(DEFAULT_EXCLUDE_DIRS)
    if exclude_dirs:
        print(f"Excluding directories named: {', '.join(sorted(exclude_dirs))}")

    dest_abs = os.path.abspath(args.dest_dir)
    matches = [m for m in find_files(args.src_dir, exclude_dirs)
               if not os.path.abspath(m).startswith(dest_abs + os.sep)]
    if not matches:
        print(f"No *{SUFFIX} files found under {args.src_dir}")
        return

    results = collect(matches, args.dest_dir, args.src_dir, flat=args.flat, move=args.move,
                      dry_run=args.dry_run)
    verb = "move" if args.move else "copy"
    for r in results:
        print(f"{r['status']:9s} {r['src']} -> {r['dest']}")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    experiments = sorted({r["experiment"] for r in results})
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    prefix = "[dry-run] would " if args.dry_run else ""
    print(f"\n{prefix}{verb} {len(results)} file(s) into {args.dest_dir} ({summary})")
    if not args.flat:
        print(f"{len(experiments)} experiment folder(s): {', '.join(experiments)}")


if __name__ == "__main__":
    main()
