#!/bin/bash
#SBATCH --job-name=train_try1
#SBATCH -o logs/%x_%J.out
#SBATCH -e logs/%x_%J.err
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:2

echo "started"

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
echo "activating environment"
source .env/bin/activate

echo "Job started on $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

echo "running python"
python "code/training_code/train.py" "/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/train_configurations/config_debug.json"
echo "finished working"