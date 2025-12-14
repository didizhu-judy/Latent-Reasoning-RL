#!/bin/bash
#SBATCH --job-name=qwen3vl8b_mmk12_grpo
#SBATCH --output=./slurmlogs/%x_%j.out
#SBATCH --time=5-00:00:00
#SBATCH --partition=lrc-xlong
#SBATCH --gres=gpu:h200:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=highest

source .venv/bin/activate


export HF_HOME=/home/d84417809/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"

# 强制使用个人 wandb 账户而不是组织账户
export WANDB_ENTITY="judyzhu"

srun python3 -m areal.launcher.local trains/grpo.py --config configs/qwen3vl8b_mmk12_grpo.yaml trial_name="run_$(date +%Y%m%d_%H%M%S)"