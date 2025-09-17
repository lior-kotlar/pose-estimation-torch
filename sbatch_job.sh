#!/bin/zsh
#SBATCH --job-name=train_try1
#SBATCH -o current_runs/%x_%J.out
#SBATCH -e current_runs/%x_%J.err
#SBATCH --mem=64g
#SBATCH -c16
#SBATCH --time=1:0:0
#SBATCH --gres=gpu:l40:s

echo "started"

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
echo "activating environment"
source /cs/labs/tsevi/lior.kotlar/pose-estimation-torch/.env/bin/activate

echo "run python"

python "/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/code/training_code/train.py" "/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/train_configurations/config1.json"

echo "finished working"