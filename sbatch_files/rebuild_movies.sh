#!/bin/bash
#SBATCH --job-name=rebuild_movies
#SBATCH -o logs/%x_%J.out
#SBATCH -e logs/%x_%J.err
# glacier is the CPU partition (the GPU partitions -- salmon/catfish/goldfish/
# dogfish -- reject a job that requests no GRES). This is MATLAB I/O, not
# inference, so no GPU is wanted.
#SBATCH --partition=glacier
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=FAIL,END

# Rebuild specific movies' dataset h5 with the MATLAB builder.
#
# For movies whose original build died partway and left a truncated h5 (see
# code/process_experiment.py's INCOMPLETE check). Each movie is rebuilt with
# its exact original start_ind/end_ind through process_experiment's staged
# build: MATLAB writes into .build_tmp and the h5 is committed only on a zero
# exit, so an interrupted run leaves no partial file behind and the script is
# safe to re-run -- already-complete movies are skipped.
#
# No GPU: this is MATLAB I/O, not inference. ~10 frames/s per movie.
#
# usage:
#   sbatch sbatch_files/rebuild_movies.sh <rebuild_script.py> [args...]
#
# <rebuild_script.py> either holds the movie list itself (the original
# rebuild_truncated_movies form, no args) or takes a manifest, as
# code/rebuild_edge_cut_movies.py does. Anything after the script path is
# forwarded to it untouched.

set -eo pipefail
REBUILD_SCRIPT="${1:?rebuild script required (arg 1)}"
shift

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
source .env/bin/activate

echo "script: $REBUILD_SCRIPT $*"
echo "host  : $(hostname) | cpus: ${SLURM_CPUS_ON_NODE:-?} | job: ${SLURM_JOB_ID:-?}"
df -h /cs/labs/tsevi | tail -1

python -u "$REBUILD_SCRIPT" "$@"
