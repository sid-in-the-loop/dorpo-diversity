#!/bin/bash

#SBATCH --job-name=ddpo_both_gemma2_2b
#SBATCH --gres=gpu:6
#SBATCH --partition=general
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train/ddpo_both_gemma2_2b_%j.out
#SBATCH --error=logs/train/ddpo_both_gemma2_2b_%j.err

cd /home/ssmurali/DiversityTuning

# Create log directories
mkdir -p logs/train

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diversitytuning

export PYTHONPATH=$PWD:$PYTHONPATH

# HuggingFace authentication (required for gated models)
export HF_TOKEN=${HF_TOKEN:-your_hf_token_here}

# Use /tmp for HuggingFace cache (more space on compute nodes)
export HF_HOME=/tmp/hf_cache_$SLURM_JOB_ID
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID
mkdir -p $HF_HOME
mkdir -p $TRITON_CACHE_DIR

CMD="""
accelerate launch \
    --gpu_ids 0,1,2,3,4,5 \
    `pwd`/scripts_dpo/generation_ddpo_model_train.py \
    --modelname google/gemma-2-2b-it \
    --init_lora_model_path checkpoints/gemma-2-2b-it/generation_models_SFT/google_gemma-2-2b-it_epoch_final.pt \
    --output_dir checkpoints/gemma-2-2b-it/generation_models_DDPO_both \
    --dataset data/writingPrompt_post_pair_sem_sty \
    --gradient_accumulation_steps 8
"""

echo $CMD
eval "$CMD"

