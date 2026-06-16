#!/bin/bash
# predict_array.sh
#
# SLURM job-array launcher: run prediction on one movie per array task,
# concurrent up to the array '%' cap. Each task writes its own log.
#
# USAGE
# -----
#   # 1. build a manifest (one movie directory per line, no trailing slash).
#   #    IMPORTANT: put it on a SHARED filesystem (project dir or $HOME), not
#   #    /tmp -- compute nodes have their own /tmp and won't see a login-node
#   #    file there.
#   mkdir -p manifests
#   ls -d inference_datasets/test/2023/mov* > manifests/movies_2023.txt
#
#   # 2. submit the array:
#   N=$(wc -l < manifests/movies_2023.txt)
#   sbatch --array=0-$((N-1))%32 sbatch_files/predict_array.sh manifests/movies_2023.txt predict_configurations/config1.json
#
# The %32 caps concurrent tasks at 32. Raise / lower as you like
# (e.g. %8 to be polite to other users, %80 to saturate salmon).
#
# To re-run only specific tasks after a partial failure:
#   sbatch --array=12,45,108%16 sbatch_files/predict_array.sh ...
#
#SBATCH --job-name=predict_array
#SBATCH -o logs/%x_%A_%a.out     # %A = array job id, %a = task id
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --mem=256g
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -p salmon
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=FAIL

# NOTE: deliberately NOT using `set -u` because SLURM env vars (like
# CUDA_VISIBLE_DEVICES) may legitimately be unset in some contexts and we
# don't want bash to abort silently before SLURM flushes the log file.
set -eo pipefail

MANIFEST="${1:-}"
BASE_CONFIG="${2:-}"
TIMINGS_PATH="${3:-}"   # optional: per-movie timing ledger (CSV)

if [ -z "$MANIFEST" ];       then echo "Manifest path required (arg 1)"     >&2; exit 1; fi
if [ -z "$BASE_CONFIG" ];    then echo "Base config path required (arg 2)"  >&2; exit 1; fi
if [ ! -f "$MANIFEST" ];     then echo "Manifest not found: $MANIFEST"      >&2; exit 1; fi
if [ ! -f "$BASE_CONFIG" ];  then echo "Base config not found: $BASE_CONFIG" >&2; exit 1; fi

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
source .env/bin/activate

# Pick this task's movie from the manifest (1-indexed via sed).
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-manual}"
TASK_LINE=$((TASK_ID + 1))
MOVIE_DIR=$(sed -n "${TASK_LINE}p" "$MANIFEST")
if [ -z "$MOVIE_DIR" ]; then
    echo "No movie at line $TASK_LINE in $MANIFEST" >&2
    exit 1
fi
MOV_NAME=$(basename "$MOVIE_DIR")

echo "==========================================="
echo "task        : $TASK_ID  ($MOV_NAME)"
echo "host        : $(hostname)"
echo "GPU(s)      : ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "movie_dir   : $MOVIE_DIR"
echo "base config : $BASE_CONFIG"
echo "==========================================="

# Generate a per-task config that points 'data directory' at just this movie.
# (The Predictor walks 'data directory' for *_ds_*tc_*tj.h5 files; restricting
# it to one mov dir ensures this task only processes its assigned movie.)
# Also stamp 'general run name' = SLURM job name so every task in this array
# lands under the same parent dir (override with `sbatch -J <name>`).
# Derive 'calibration path' from the movie dir's parent (the experiment dir),
# where the build step writes calibration.h5 -- so the calibration always
# follows the data instead of trusting a stale path baked into the base config.
TMP_CONFIG="/tmp/predict_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${MOV_NAME}.json"
RUN_NAME="${SLURM_JOB_NAME:-predict_array}"
python -c "
import json, os
cfg = json.load(open('$BASE_CONFIG'))
cfg['data directory'] = '$MOVIE_DIR'
cfg['general run name'] = '$RUN_NAME'
calib = os.path.join(os.path.dirname('$MOVIE_DIR'.rstrip('/')), 'calibration.h5')
if not os.path.isfile(calib):
    raise SystemExit('calibration.h5 not found next to movie dir: ' + calib)
cfg['calibration path'] = calib
timings = '$TIMINGS_PATH'
if timings:
    cfg['pipeline timings path'] = timings
json.dump(cfg, open('$TMP_CONFIG', 'w'), indent=2)
print('wrote', '$TMP_CONFIG  (run name: $RUN_NAME, calib:', calib, ', timings:', timings or '<none>', ')')
"

# Run prediction. Use --unbuffered so stdout/stderr stream live (useful for
# tailing the log while the array is running).
python -u code/prediction_code_lior/predict.py "$TMP_CONFIG"
rc=$?

rm -f "$TMP_CONFIG"
echo "==========================================="
echo "[$MOV_NAME] exit code $rc"
exit $rc
