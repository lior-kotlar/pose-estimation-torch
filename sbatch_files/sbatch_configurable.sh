#!/bin/bash
#SBATCH --job-name=default_training_job  # This will be overridden by the command line
#SBATCH -o logs/%x_%J.out                # %x automatically uses the new job name
#SBATCH -e logs/%x_%J.err
#SBATCH --mem=256g
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL

# usage: sbatch -J <YOUR_JOB_NAME> <THIS_SBATCH_FILE_PATH> <PYTHON_SCRIPT_PATH> <CONFIG_PATH>
SCRIPT_PATH=$1
CONFIG_PATH=$2

# Safety check for Python script
if [ -z "$SCRIPT_PATH" ]; then
  echo "Error: No python script path provided (Argument 1)."
  exit 1
fi

# Safety check for Config path
if [ -z "$CONFIG_PATH" ]; then
  echo "Error: No configuration file path provided (Argument 2)."
  exit 1
fi

echo "started"
cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
source .env/bin/activate

echo "Job started on $(hostname)"
echo "Job Name: $SLURM_JOB_NAME"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Running script: $SCRIPT_PATH"
echo "Using configuration: $CONFIG_PATH"

# Now executes the variable script path with the variable config path
python "$SCRIPT_PATH" "$CONFIG_PATH"

echo "finished working"