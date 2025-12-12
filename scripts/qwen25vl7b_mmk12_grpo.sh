#!/bin/bash
#SBATCH --job-name=mmk12_grpo
#SBATCH --output=./slurmlogs/%x_%j.out
#SBATCH --time=5-00:00:00
#SBATCH --partition=camera-inf
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=high

srun python3 -m areal.launcher.local trains/thinking_grpo.py --config configs/qwen25vl7b_mmk12_grpo.yaml trial_name=$(date +%Y%m%d_%H%M%S)