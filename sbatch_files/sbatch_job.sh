#!/bin/bash
#SBATCH --job-name=train_pwing_pcam
#SBATCH -o logs/%x_%J.out
#SBATCH -e logs/%x_%J.err
#SBATCH --partition=salmon
#SBATCH --mem=256g
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:l40s:1

# Trains on an L40S GPU (salmon partition): 48 GB, ~3x the FP16 throughput and
# ~2.8x the memory bandwidth of the catfish L4s, and idle nodes are usually
# available so queue time is short.
# To use a different GPU, override at submit time, e.g. H200:
#   sbatch -p goldfish --gres=gpu:h200:1 sbatch_files/sbatch_job.sh [CONFIG]
#
# usage:  sbatch -J <job_name> sbatch_files/sbatch_job.sh [TRAIN_CONFIG_JSON]
# CONFIG defaults to train_configurations/config1.json when omitted.

CONFIG="${1:-/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/train_configurations/config1.json}"

echo "started"

cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
echo "activating environment"
source .env/bin/activate

echo "Job started on $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

echo "running python on config: $CONFIG"
python "code/training_code/train.py" "$CONFIG"
echo "finished working"
