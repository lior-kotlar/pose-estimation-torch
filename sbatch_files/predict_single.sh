#!/bin/bash
#SBATCH --job-name=predict_single
#SBATCH -o logs/%x_%J.out
#SBATCH -e logs/%x_%J.err
#SBATCH --partition=salmon
#SBATCH --mem=256g
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=FAIL,END

# Predict ONE movie with the current prediction_models/ ensemble.
#
# The ensemble is whatever is enabled under prediction_models/ (auto-discovered
# via the base config's "prediction models directory"). CPU-heavy: the selector
# grows ~2^M in the number of members, so the config caps it with
# "max ensemble models"; more --cpus-per-task scales the selection near-linearly
# (the worker pool respects the SLURM allocation). GPU is only for inference.
#
# usage:
#   sbatch sbatch_files/predict_single.sh <movie_dir> [predict_config] [calibration.h5]
#   sbatch -J <run_name> sbatch_files/predict_single.sh <movie_dir>
#
# <movie_dir> holds mov_*.h5. Calibration is resolved in this order:
#   1) the optional 3rd arg, 2) 'calibration path' in the config (if it exists),
#   3) calibration.h5 next to the movie dir. It must be an HDF5 calibration.h5
#   (K_matrices, rotation_matrices, ...), NOT a raw easyWand .mat.
# predict_config defaults to config1.json.

set -eo pipefail
MOVIE_DIR="${1:?movie dir required (arg 1)}"
BASE_CONFIG="${2:-predict_configurations/config1.json}"
CALIB_ARG="${3:-}"

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
source .env/bin/activate

MOV_NAME=$(basename "$MOVIE_DIR")
RUN_NAME="${SLURM_JOB_NAME:-predict_single}"
echo "movie : $MOVIE_DIR"
echo "config: $BASE_CONFIG | run: $RUN_NAME | host: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-<none>}"

# Per-run config: point 'data directory' at this movie and resolve calibration:
# explicit 3rd arg > config's 'calibration path' (if the file exists) > a
# calibration.h5 sitting next to the movie dir. Reject easyWand .mat with a hint.
TMP_CONFIG="/tmp/predict_single_${SLURM_JOB_ID:-$$}_${MOV_NAME}.json"
python -c "
import json, os
cfg = json.load(open('$BASE_CONFIG'))
cfg['data directory'] = '$MOVIE_DIR'
cfg['general run name'] = '$RUN_NAME'
arg_calib = '$CALIB_ARG'
cfg_calib = cfg.get('calibration path')
sibling = os.path.join(os.path.dirname('$MOVIE_DIR'.rstrip('/')), 'calibration.h5')
if arg_calib and os.path.isfile(arg_calib):
    calib = arg_calib
elif cfg_calib and os.path.isfile(cfg_calib):
    calib = cfg_calib
elif os.path.isfile(sibling):
    calib = sibling
else:
    raise SystemExit('no calibration found. tried: arg=%r config=%r sibling=%r' % (arg_calib or None, cfg_calib, sibling))
if calib.endswith('.mat'):
    raise SystemExit('calibration is an easyWand .mat (%s); the pipeline needs a built calibration.h5 (K_matrices, rotation_matrices, ...). Point the config or the 3rd arg at a calibration.h5.' % calib)
cfg['calibration path'] = calib
json.dump(cfg, open('$TMP_CONFIG', 'w'), indent=2)
print('wrote', '$TMP_CONFIG', '(calib:', calib, ')')
"

python -u code/prediction_code_lior/predict.py "$TMP_CONFIG"
rc=$?
rm -f "$TMP_CONFIG"
echo "[$MOV_NAME] exit code $rc"
exit $rc
