#!/bin/bash

#SBATCH --job-name=eval_ddpo_both_gemma2_2b
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval/eval_ddpo_both_gemma2_2b_%j.out
#SBATCH --error=logs/eval/eval_ddpo_both_gemma2_2b_%j.err

cd /home/ssmurali/DiversityTuning

# Create log directories
mkdir -p logs/eval

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diversitytuning

export PYTHONPATH=$PWD:$PYTHONPATH

# HuggingFace authentication
export HF_TOKEN=${HF_TOKEN:-your_hf_token_here}

# Use /tmp for HuggingFace cache
export HF_HOME=/tmp/hf_cache_$SLURM_JOB_ID
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

# Create eval output directory if it doesn't exist
mkdir -p data/eval/ablation

CMD="""
python -m scripts_eval.generation_eval1_1 \
    --modelname google/gemma-2-2b-it \
    --lora_model_path checkpoints/gemma-2-2b-it/generation_models_DDPO_both/checkpoint-final \
    --val_path data/eval/ablation/gen_eval1_DDPO_both.csv \
    --device cuda:0 \
    --rewardmodelname google/gemma-2-2b \
    --reward_lora_model_path checkpoints/reward_models/google_gemma-2-2b_10.pt
"""

echo $CMD
eval "$CMD"

