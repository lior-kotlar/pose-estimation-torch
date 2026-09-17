"""
make_raw_movies.py
==================

Give every movie under one or more folders its raw movie: the camera views of
its *_sparse.mat files tiled into one mp4 by matlab/+VideoEditing/JoinSparses.m,
saved beside the mats as <movie>_raw_fr30_skip1.mp4.

Searches each folder and all its sub-folders for movie dirs (named mov<N> and
holding at least one *_sparse.mat), leaves the ones that already have a raw
movie alone and builds the rest, one MATLAB run per movie. This is the same
step process_experiment.py runs first on every experiment it prepares.

USAGE
-----
    .env/bin/python code/make_raw_movies.py <folder> [<folder> ...] [--dry-run]

    # a big tree (~6 min a movie) on a CPU job array:
    sbatch -J raw_tsory --array=0-19 --gres=gpu:0 --mem=16g --mail-type=FAIL \\
        sbatch_files/sbatch_configurable.sh code/make_raw_movies.py \\
        inference_datasets/Tsory

Inside a SLURM job array each task takes its own share of the movie dirs, so
the tasks never overlap. The split is over the full sorted list, so it does not
shift while other tasks are building. Submit a plain range (--array=0-19); to
fill in failures, resubmit the whole array -- finished movies are skipped.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_experiment import find_movie_dirs, find_raw_movie, run_raw_movies


def array_share() -> "tuple | None":
    """(i, n): this is task i of an n-task SLURM job array. None outside one."""
    env = os.environ
    if "SLURM_ARRAY_TASK_ID" not in env:
        return None
    first, last = int(env["SLURM_ARRAY_TASK_MIN"]), int(env["SLURM_ARRAY_TASK_MAX"])
    return int(env["SLURM_ARRAY_TASK_ID"]) - first, last - first + 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+",
                   help="folders to search, sub-folders included")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be built, build nothing")
    args = p.parse_args()

    movie_dirs = []
    for folder in args.folders:
        if not os.path.isdir(folder):
            sys.exit(f"not a directory: {folder}")
        movie_dirs += find_movie_dirs(folder, recursive=True)
    movie_dirs = list(dict.fromkeys(movie_dirs))    # nested folders overlap
    n_missing = sum(find_raw_movie(d) is None for d in movie_dirs)
    print(f"{len(movie_dirs)} movie dir(s) found, {n_missing} without a raw movie")
    share = array_share()
    if share:
        i, n = share
        movie_dirs = movie_dirs[i::n]
        print(f"array task {i + 1} of {n}: {len(movie_dirs)} movie dir(s)")

    failed = run_raw_movies(movie_dirs, args.dry_run)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
