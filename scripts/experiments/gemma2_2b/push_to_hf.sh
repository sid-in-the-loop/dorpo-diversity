#!/bin/bash

#SBATCH --job-name=push_to_hf_gemma2_2b
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/push_to_hf_%j.out
#SBATCH --error=logs/push_to_hf_%j.err

cd /home/ssmurali/DiversityTuning

# Create log directories
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diversitytuning

export PYTHONPATH=$PWD:$PYTHONPATH

# HuggingFace token
export HF_TOKEN=${HF_TOKEN:-your_hf_token_here}

# Use /tmp for HuggingFace cache
export HF_HOME=/tmp/hf_cache_$SLURM_JOB_ID
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

# Push all models to HF Hub (adapter only, no base model loading)
python scripts/experiments/gemma2_2b/push_to_hf.py \
    --hf_token $HF_TOKEN \
    --model_type all

