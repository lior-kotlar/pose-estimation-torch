#!/bin/bash
# pipeline.sh
#
# End-to-end orchestrator for one freshly-imported experiment:
#   1. run process_experiment.py here in this CPU job (clean, prescan, flip,
#      build, verify, write manifest, append per-step timings to
#      <input_dir>/pipeline_timings.csv).
#   2. if a non-empty manifest is produced, submit predict_array.sh as a
#      separate sbatch job, sized exactly to the manifest, named after this
#      job (so the array's per-task output directory and timing rows all
#      land under the same run name).
#
# This job exits as soon as the predict array is submitted; the array
# itself runs in parallel on GPU nodes.
#
# USAGE
# -----
#   sbatch -J <experiment_name> sbatch_files/pipeline.sh \
#       <input_dir> <easywand_path> <cam_name> <predict_config> \
#       [array_concurrency] [extra process_experiment.py args...]
#
# Example:
#   sbatch -J 2023_mov101to110 sbatch_files/pipeline.sh \
#       inference_datasets/test/2023/101to110 \
#       inference_datasets/.../10_8_23_allmovs_easyWandData.mat \
#       cam1 predict_configurations/config1.json
#
# The -J value becomes both the prep job name (used by logs/%x_%j.out) and
# the predict-array job name (used to derive the parent run directory).
#
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p glacier
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=FAIL

set -eo pipefail

INPUT_DIR="${1:-}"
EASYWAND="${2:-}"
CAM="${3:-}"
PRED_CONFIG="${4:-}"
ARRAY_CONCURRENCY="${5:-32}"   # optional: cap concurrent GPU tasks
# Anything after the 5th argument is forwarded verbatim to process_experiment.py
# (e.g. --perturbation --perturbation-type roll), so prep-stage options do not
# each need a positional slot here.
EXTRA_PREP_ARGS=("${@:6}")

usage() {
    echo "Usage: sbatch -J <name> $0 <input_dir> <easywand> <cam> <predict_config> [array_concurrency] [extra prep args...]" >&2
    echo "  <cam>: mirror cam to vertically flip (e.g. cam1), or 'none' to skip the flip" >&2
    exit 1
}

[ -z "$INPUT_DIR"    ] && { echo "input_dir required (arg 1)" >&2; usage; }
[ -z "$EASYWAND"     ] && { echo "easywand path required (arg 2)" >&2; usage; }
[ -z "$CAM"          ] && { echo "cam name required (arg 3), or 'none' to skip the flip" >&2; usage; }
[ -z "$PRED_CONFIG"  ] && { echo "predict config required (arg 4)" >&2; usage; }
[ ! -d "$INPUT_DIR"  ] && { echo "input_dir not a directory: $INPUT_DIR" >&2; exit 1; }
[ ! -f "$EASYWAND"   ] && { echo "easyWand mat not found: $EASYWAND" >&2; exit 1; }
[ ! -f "$PRED_CONFIG" ] && { echo "predict config not found: $PRED_CONFIG" >&2; exit 1; }

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
source .env/bin/activate

RUN_NAME="${SLURM_JOB_NAME:-pipeline}"
EXP_NAME=$(basename "$INPUT_DIR")
TIMINGS_PATH="$(realpath "$INPUT_DIR")/pipeline_timings.csv"

echo "==========================================="
echo "pipeline.sh  job=$SLURM_JOB_ID  run_name=$RUN_NAME"
echo "  input_dir   : $INPUT_DIR"
echo "  experiment  : $EXP_NAME"
echo "  easywand    : $EASYWAND"
echo "  cam         : $CAM"
echo "  predict cfg : $PRED_CONFIG"
echo "  timings     : $TIMINGS_PATH"
echo "==========================================="

# Step 1 — data prep. Per-movie timings get appended to TIMINGS_PATH.
# 'none'/'noflip'/'skip' for <cam> skips the (non-idempotent) mirror flip.
PREP_ARGS=(--easywand "$EASYWAND")
case "${CAM,,}" in
    none|noflip|skip|-) echo "  flip: SKIPPED (cam='$CAM')"; PREP_ARGS+=(--skip-flip) ;;
    *)                  PREP_ARGS+=(--cam "$CAM") ;;
esac
PREP_ARGS+=("${EXTRA_PREP_ARGS[@]}")
# An `if` rather than `[ ... ] && echo ...`: under `set -e` the one-liner form
# survives an empty array only by the &&-list exemption, and would abort the
# whole prep job if it ever became the last statement of a block.
if [ ${#EXTRA_PREP_ARGS[@]} -gt 0 ]; then
    echo "  extra prep args: ${EXTRA_PREP_ARGS[*]}"
fi
python -u code/process_experiment.py "$INPUT_DIR" "${PREP_ARGS[@]}"

# Step 2 — submit the predict array if a manifest with content exists.
MANIFEST="manifests/good_movies_${EXP_NAME}.txt"
if [ ! -s "$MANIFEST" ]; then
    echo "Manifest $MANIFEST missing or empty; no predict step to launch."
    exit 0
fi
N=$(wc -l < "$MANIFEST")
echo "Manifest $MANIFEST has $N movie(s); submitting predict array."

sbatch -J "$RUN_NAME" \
       --array=0-$((N-1))%${ARRAY_CONCURRENCY} \
       sbatch_files/predict_array.sh \
       "$MANIFEST" "$PRED_CONFIG" "$TIMINGS_PATH"
