#!/bin/bash
#SBATCH --job-name=train_try1
#SBATCH -o logs/%x_%J.out
#SBATCH -e logs/%x_%J.err
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

echo "started"

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
echo "activating environment"
source .env/bin/activate

echo "Job started on $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

echo "sbatch: running python"
python "code/training_code/train.py" "/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/train_configurations/config_debug.json"
echo "sbatch: finished working"