#!/bin/bash
#SBATCH --job-name=default_training_job  # This will be overridden by the command line
#SBATCH -o logs/%x_%J.out                # %x automatically uses the new job name
#SBATCH -e logs/%x_%J.err
#SBATCH --mem=256g
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL

# usage: sbatch -J <JOB_NAME> <THIS_SBATCH_FILE_PATH> <PYTHON_SCRIPT_PATH> [ARGS...]
#
# Examples:
#   sbatch -J train_run sbatch_configurable.sh code/training_code/train.py configs/foo.json
#   sbatch -J process_exp sbatch_configurable.sh code/process_experiment.py \
#       inference_datasets/test/2023 \
#       --easywand inference_datasets/.../10_8_23_allmovs_easyWandData.mat \
#       --cam cam1 --verify
#
# Tip: for non-GPU jobs (e.g. process_experiment.py), override the gres line
# at submit time:  sbatch --gres=gpu:0 -J ... sbatch_configurable.sh ...
SCRIPT_PATH=$1
shift   # the rest of $@ is forwarded verbatim to python

if [ -z "$SCRIPT_PATH" ]; then
  echo "Error: No python script path provided (Argument 1)."
  exit 1
fi

echo "started"
cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
source .env/bin/activate

echo "Job started on $(hostname)"
echo "Job Name: $SLURM_JOB_NAME"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Running script: $SCRIPT_PATH"
echo "With args: $*"

python "$SCRIPT_PATH" "$@"

echo "finished working"