#!/bin/bash
#SBATCH --job-name=submit_predictions
#SBATCH -o logs/%x_%J.out
#SBATCH -e logs/%x_%J.err
#SBATCH --partition=glacier
#SBATCH --mem=8g
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
# END as well as FAIL: this job is the step that actually queues the GPU array,
# so "it finished" is the thing worth knowing -- typically nobody is watching a
# terminal by the time a multi-hour rebuild has chained into it.
#SBATCH --mail-type=FAIL,END

# Run a short CPU-side pipeline step as a SLURM job, so it can be chained with
# --dependency and does not depend on an interactive allocation staying alive
# for the hours the step it follows may take. Two users so far:
#
#   code/submit_edge_cut_predictions.py  chained off a rebuild, queues the array
#   code/check_run_complete.py           chained off the array, reconciles it
#
# Both only read h5 headers and call sbatch, so they are small and short; the
# GPU work is in the array, not here.
#
# usage:
#   sbatch --dependency=afterany:<jobid> \
#          sbatch_files/submit_predictions.sh <script.py> [args...]
#
# afterany rather than afterok on purpose: the rebuild script exits non-zero if
# ANY single movie failed to build, and that must not block predictions for the
# ones that did build. The submit script pre-flights every movie (build
# complete + calibration + reprojection verify) and skips whatever is not
# ready, so running it on a partial rebuild is safe.

set -eo pipefail
SUBMIT_SCRIPT="${1:?submit script required (arg 1)}"
shift

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
source .env/bin/activate

echo "script: $SUBMIT_SCRIPT $*"
echo "host  : $(hostname) | job: ${SLURM_JOB_ID:-?}"

python -u "$SUBMIT_SCRIPT" "$@"
