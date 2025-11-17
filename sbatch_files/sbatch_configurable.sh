#!/bin/bash
#SBATCH --job-name=default_training_job  # This will be overridden by the command line
#SBATCH -o logs/%x_%J.out                # %x automatically uses the new job name
#SBATCH -e logs/%x_%J.err
#SBATCH --mem=256g
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1

# usage: sbatch -J <YOUR_JOB_NAME> <THIS_SBATCH_FILE_PATH> <CONFIG_PATH>
CONFIG_PATH=$1

# Safety check
if [ -z "$CONFIG_PATH" ]; then
  echo "Error: No configuration file path provided."
  exit 1
fi

echo "started"
cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
source .env/bin/activate

echo "Job started on $(hostname)"
echo "Job Name: $SLURM_JOB_NAME"       # This will print the custom name you set
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Using configuration: $CONFIG_PATH"

python "code/training_code/train.py" "$CONFIG_PATH"

echo "finished working"