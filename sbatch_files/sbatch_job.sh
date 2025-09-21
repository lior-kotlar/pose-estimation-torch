#!/bin/bash
#SBATCH --job-name=train_per_wing
#SBATCH -o logs/%x_%J.out
#SBATCH -e logs/%x_%J.err
#SBATCH --mem=256g
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:2

echo "started"

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
echo "activating environment"
source .env/bin/activate

echo "Job started on $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

echo "running python"
python "code/training_code/train.py" "/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/train_configurations/config1.json"
echo "finished working"