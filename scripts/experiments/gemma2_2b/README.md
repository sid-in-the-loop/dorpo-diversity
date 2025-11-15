# Gemma-2-2B Experiment Scripts

This directory contains all scripts needed to run SFT, DDPO, and DORPO experiments for Gemma-2-2B.

## Directory Structure

```
scripts/experiments/gemma2_2b/
├── data/
│   └── create_datasets.sh    # Create semantic-only and style-only datasets
├── train/
│   ├── sft.sh                 # Supervised Fine-Tuning
│   ├── ddpo_sem.sh           # DDPO with semantic diversity
│   ├── ddpo_sty.sh           # DDPO with style diversity
│   ├── ddpo_both.sh          # DDPO with both semantic + style diversity
│   ├── dorpo_sem.sh          # DORPO with semantic diversity
│   ├── dorpo_sty.sh          # DORPO with style diversity
│   └── dorpo_both.sh         # DORPO with both semantic + style diversity
└── eval/
    ├── sft.sh                 # Evaluate SFT model
    ├── ddpo_sem.sh           # Evaluate DDPO-sem model
    ├── ddpo_sty.sh           # Evaluate DDPO-sty model
    ├── ddpo_both.sh          # Evaluate DDPO-both model
    ├── dorpo_sem.sh          # Evaluate DORPO-sem model
    ├── dorpo_sty.sh          # Evaluate DORPO-sty model
    └── dorpo_both.sh         # Evaluate DORPO-both model
```

## Execution Order

### Step 1: Create Datasets (if needed)
If you don't have the semantic-only and style-only datasets yet:

```bash
cd scripts/experiments/gemma2_2b/data
sbatch create_datasets.sh
```

This creates:
- `data/writingPrompt_post_pair_sem` (semantic-only)
- `data/writingPrompt_post_pair_sty` (style-only)
- `data/writingPrompt_post_pair_sem_sty` (both - should already exist)

### Step 2: Train Models

**Prerequisites:**
- SFT model must be trained first (for DDPO variants)
- Reward model should be trained (for evaluation)

**Submit training jobs:**

```bash
cd scripts/experiments/gemma2_2b/train

# SFT (must run first)
sbatch sft.sh

# After SFT completes, submit DDPO/DORPO variants (can run in parallel)
sbatch ddpo_sem.sh
sbatch ddpo_sty.sh
sbatch ddpo_both.sh
sbatch dorpo_sem.sh
sbatch dorpo_sty.sh
sbatch dorpo_both.sh
```

**Note:** DDPO variants require the SFT checkpoint to exist. DORPO variants train from the base model.

### Step 3: Evaluate Models

After training completes, submit evaluation jobs:

```bash
cd scripts/experiments/gemma2_2b/eval

sbatch sft.sh
sbatch ddpo_sem.sh
sbatch ddpo_sty.sh
sbatch ddpo_both.sh
sbatch dorpo_sem.sh
sbatch dorpo_sty.sh
sbatch dorpo_both.sh
```

## Resource Requirements

**Training scripts:**
- GPUs: 4
- Memory: 128G
- Time: 24 hours
- CPUs: 48

**Evaluation scripts:**
- GPUs: 1
- Memory: 32G
- Time: 12 hours
- CPUs: 8

## Output Locations

### Checkpoints

- SFT: `checkpoints/gemma-2-2b-it/generation_models_SFT/google_gemma-2-2b-it_epoch_final.pt`
- DDPO-sem: `checkpoints/gemma-2-2b-it/generation_models_DDPO_sem/checkpoint-final/`
- DDPO-sty: `checkpoints/gemma-2-2b-it/generation_models_DDPO_sty/checkpoint-final/`
- DDPO-both: `checkpoints/gemma-2-2b-it/generation_models_DDPO_both/checkpoint-final/`
- DORPO-sem: `checkpoints/gemma-2-2b-it/generation_models_DORPO_sem/checkpoint-final/`
- DORPO-sty: `checkpoints/gemma-2-2b-it/generation_models_DORPO_sty/checkpoint-final/`
- DORPO-both: `checkpoints/gemma-2-2b-it/generation_models_DORPO_both/checkpoint-final/`

### Evaluation Results

All evaluation results are saved to `data/eval/ablation/`:
- `gen_eval1_SFT.csv`
- `gen_eval1_DDPO_sem.csv`
- `gen_eval1_DDPO_sty.csv`
- `gen_eval1_DDPO_both.csv`
- `gen_eval1_DORPO_sem.csv`
- `gen_eval1_DORPO_sty.csv`
- `gen_eval1_DORPO_both.csv`

### Logs

All logs are saved to `logs/`:
- Training: `logs/{method}_{variant}_gemma2_2b_{job_id}.out`
- Evaluation: `logs/eval_{method}_{variant}_gemma2_2b_{job_id}.out`

## Training Parameters

**SFT:**
- Model: `google/gemma-2-2b-it`
- LoRA: r=128, α=256
- Epochs: 1
- Scheduler: cosine warmup

**DDPO:**
- β=0.1
- Epochs: 3
- Starts from: SFT model
- Scheduler: linear

**DORPO:**
- λ=0.25 (beta in ORPOConfig)
- Epochs: 4
- Starts from: base model
- Scheduler: linear

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View logs:
```bash
tail -f logs/{script_name}_gemma2_2b_{job_id}.out
```

## Notes

- All scripts use 4 GPUs for training (changed from 6)
- Evaluation scripts use 1 GPU
- Make sure the reward model checkpoint exists before running evaluations
- DDPO variants depend on SFT completion
- DORPO variants can run independently (from base model)










