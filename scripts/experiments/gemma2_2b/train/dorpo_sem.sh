#!/bin/bash

#SBATCH --job-name=dorpo_sem_gemma2_2b
#SBATCH --gres=gpu:4
#SBATCH --partition=general
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train/dorpo_sem_gemma2_2b_%j.out
#SBATCH --error=logs/train/dorpo_sem_gemma2_2b_%j.err

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
# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD="""
accelerate launch \
    --gpu_ids 0,1,2,3 \
    `pwd`/scripts_orpo/generation_dorpo_model_train.py \
    --modelname google/gemma-2-2b-it \
    --output_dir checkpoints/gemma-2-2b-it/generation_models_DORPO_sem \
    --dataset data/writingPrompt_post_pair_sem
"""

echo $CMD
eval "$CMD"

