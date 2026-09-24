#!/bin/bash
#SBATCH --job-name=round
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# Defaults only: code/local_reanalysis_server.py passes what each kind of round actually needs
# on the sbatch command line, which overrides these. glacier is the CPU partition -- no GPU is
# used by either kind.
#SBATCH --partition=glacier
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00

# One movie per array task. Takes the manifest, then the python entry point to run on that movie,
# then any arguments for it -- so one script serves every kind of round rather than one script
# per kind. Submit from the repo root:
#   sbatch --array=1-$(wc -l < MANIFEST) sbatch_files/round_array.sh MANIFEST code/realign_ensemble.py --no-reanalyse
#   sbatch --array=1-$(wc -l < MANIFEST) sbatch_files/round_array.sh MANIFEST code/reanalyse_movies.py --only-video
set -eo pipefail
MANIFEST="${1:?manifest required (arg 1)}"
ENTRY="${2:?python entry point required (arg 2)}"
shift 2
cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
MOVIE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST" | cut -f1)
echo "task $SLURM_ARRAY_TASK_ID: $(basename "$ENTRY") on $MOVIE at $(hostname), $(nproc) cpus"
MPLBACKEND=Agg .env/bin/python "$ENTRY" "$MOVIE" "$@"
