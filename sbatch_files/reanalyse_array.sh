#!/bin/bash
# reanalyse_array.sh
#
# SLURM job array (CPU only): rewrite every derived product of ALREADY-PREDICTED
# movies from their cached 3D points -- analysis h5, CSV, both PNGs, the flight
# viewer, 'All body data.html', movie_html.html, source.json and (with
# --with-mp4 --force-mp4) the overlay mp4 -- using the CURRENT code and the
# experiment's live perturbation.json. No GPU, no re-prediction.
#
# Use it when the declaration or the product code changed after prediction
# (e.g. adding the lighting block), instead of re-running predict_array.sh.
#
# USAGE
#   # manifest = one predicted movie OUTPUT dir per line (predict_output/<exp>/<movie>)
#   N=$(wc -l < manifests/reanalyse_X.txt)
#   sbatch -J reanalyse_X --array=0-$((N-1))%40 sbatch_files/reanalyse_array.sh \
#       manifests/reanalyse_X.txt [extra reanalyse_movies.py args...]
#
# Default extra args: --with-mp4 --force-mp4. Previous products are moved into
# superseded_<timestamp>/ unless --no-archive is passed.
#
#SBATCH --job-name=reanalyse_array
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p glacier
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=FAIL

set -eo pipefail

MANIFEST="${1:-}"
if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
    echo "Manifest required (arg 1): $MANIFEST" >&2; exit 1
fi
shift
EXTRA=("$@")
if [ ${#EXTRA[@]} -eq 0 ]; then EXTRA=(--with-mp4 --force-mp4); fi

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
source .env/bin/activate

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MOVIE_DIR=$(sed -n "$((TASK_ID + 1))p" "$MANIFEST")
if [ -z "$MOVIE_DIR" ]; then echo "No movie at line $((TASK_ID + 1)) of $MANIFEST" >&2; exit 1; fi

echo "==========================================="
echo "task      : $TASK_ID  ($(basename "$MOVIE_DIR"))"
echo "host      : $(hostname)"
echo "movie_dir : $MOVIE_DIR"
echo "args      : ${EXTRA[*]}"
echo "==========================================="
python -u code/reanalyse_movies.py "$MOVIE_DIR" "${EXTRA[@]}"
