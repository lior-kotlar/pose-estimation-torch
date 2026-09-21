#!/bin/bash
#SBATCH --job-name=realign_ensemble
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# glacier is the CPU partition: this re-runs the ensemble step from each member's saved
# points_3D_all.npy and re-analyses, no GPU.
#SBATCH --partition=glacier
#SBATCH --mem=32g
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

# One movie per array task: code/realign_ensemble.py on line $SLURM_ARRAY_TASK_ID of the manifest
# (one movie dir per line; further tab-separated columns are ignored). Any further arguments are
# passed on to realign_ensemble.py. Submit from the repo root:
#   sbatch --array=1-$(wc -l < MANIFEST) sbatch_files/realign_ensemble_array.sh MANIFEST
#   sbatch --array=1-$(wc -l < MANIFEST) sbatch_files/realign_ensemble_array.sh MANIFEST --no-reanalyse
set -eo pipefail
MANIFEST="${1:?manifest required (arg 1)}"
shift
cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
MOVIE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST" | cut -f1)
echo "task $SLURM_ARRAY_TASK_ID: $MOVIE on $(hostname), $(nproc) cpus"
MPLBACKEND=Agg .env/bin/python code/realign_ensemble.py "$MOVIE" "$@"
