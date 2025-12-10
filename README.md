# SafeORPO for Safe and Creative Story Writing

This repo contains the full SafeORPO pipeline for creative writing: data prep, safety annotation, high-risk selection, reward modeling, SafeORPO training, evaluation, and plotting.

We use the WritingPrompts preferences dataset and module-style execution (e.g., `python -m scripts_dataset_cleaner.0_data_prep_scoring`).

## Environment
- Python env: use your preferred venv/conda; install dependencies (see `requirements.txt` if present).
- Many scripts expect OpenAI-compatible access via CMU gateway: set `OPENAI_API_KEY` and, when needed, `--openai_base_url https://ai-gateway.andrew.cmu.edu`.
- GPU training uses `accelerate` (see sbatch templates under `scripts/sbatch/`).

## Data Pipeline (module-style)
Source dataset: `euclaise/WritingPrompts_preferences`.

Run in order (adjust paths as needed):
1) Clean + score (for reward model):
   - `python -m scripts_dataset_cleaner.0_data_prep_scoring --source <raw> --data_path data/clean_scored`
2) Filter comments (remove mod/noise, length trim):
   - `python -m scripts_dataset_cleaner.1_data_prep_filtering --source data/clean_scored --data_path data/filtered --model_to_use <hf-embed-model>`
3) Pairwise preference (quality):
   - `python -m scripts_dataset_cleaner.2_data_prep_to_preference --source data/filtered --data_path_pair data/pair_pref --gap 1`
4) Diversity pairs (semantic/style deviation):
   - `python -m scripts_dataset_cleaner.3_data_prep_for_diversified_training --source_pair data/pair_pref --data_path_pair_div data/pair_pref_div --score_process semantic style --device cuda`
   - Optional knobs: `--max_cutoff`, `--min_score_non_zero`, `--sort_with_score`
5) DivPO pairs:
   - `python -m scripts_dataset_cleaner.4_data_prep_divpo --source data/filtered --data_path_pair data/pair_divpo --score_process semantic style --device cuda`

## Safety Annotations and High-Risk IDs
1) Annotate safety flags (JSONL):
   - `python scripts_orpo/annotate_safety_flags.py --dataset <hf_or_disk_path> --output_jsonl results/annotations/safety_flags.jsonl --model_name gpt-4o-mini-2024-07-18 --base_url https://ai-gateway.andrew.cmu.edu`
2) High-risk ID selection:
   - `python scripts_orpo/generate_high_risk_ids.py --annotations_jsonl results/annotations/safety_flags.jsonl --output_json data/annotations/high_risk_ids.json --num_samples 700 --safety_flags very_high high --max_length_tokens <optional>`
High-risk IDs trigger full SafeORPO (RaR + diversity); others use DORPO-style weighting.

## Reward Model
- Train (LoRA Gemma-2B example):
  - `accelerate launch scripts_reward_modeling/reward_model_train.py --model_name google/gemma-2-2b-it --output_dir checkpoints/reward_models/gemma2_2b --use_lora`
- Evaluate reward model:
  - `python -m scripts_reward_modeling.reward_model_perf_analysis --model_path checkpoints/reward_models/gemma2_2b --data_path data/filtered`

## Generation Training (SafeORPO focus)
- Main trainer: `scripts_orpo/generation_safeorpo_model_train.py`
- Key args:
  - `--modelname google/gemma-2-2b-it` (or other)
  - `--dataset data/pair_pref_div` (diversity pairs)
  - `--high_risk_id_path data/annotations/high_risk_ids.json`
  - `--diversity_model_names jinaai/jina-embeddings-v3`
  - Response gen params: `--response_max_new_tokens 256 --response_temperature 0.7 --response_top_p 0.95`
  - Safety/RaR: `--use_rar_implicit --rubric_generation_model gpt-4o-mini-2024-07-18 --openai_base_url https://ai-gateway.andrew.cmu.edu`
  - Resume: `--auto_resume`
- Sbatch helpers: see `scripts/sbatch/safeorpo/*.sbatch` (Gemma, Qwen variants).
- Other baselines:
  - SFT: `scripts_finetuning/generation_model_train.py`
  - DORPO: `scripts_orpo/generation_dorpo_model_train.py`

## Evaluation
- Main eval (reward + semantic/style diversity):
  - `python scripts_eval/generation_eval1_1.py --modelname <ckpt_or_hf> --output_csv results/gen_eval1_<tag>.csv`
  - Applies creative developer prompts; handles Gemma/Qwen chat templating.
- Aux metrics:
  - `python scripts_eval/generation_eval1_2.py --input_csv results/gen_eval1_<tag>.csv`
- Gold eval:
  - `python scripts_eval/generation_eval2.py --input_csv results/gen_eval1_<tag>.csv`

## Plotting
- Reward vs. diversity (normalized):
  - `python scripts_eval/plot_safeorpo_results.py` → `plots/reward_vs_diversity.png`
- Semantic vs. stylistic diversity:
  - `python scripts_eval/plot_diversity_comparison.py` → `plots/semantic_vs_stylistic_diversity.png`
- Both plots can consume `plots/summary_statistics.csv` (averaged metrics).

## Key Directories
- `data/`: cleaned/filtered/pair datasets; `data/annotations/high_risk_ids.json`
- `results/`: eval CSVs; `results/annotations/safety_flags.jsonl`
- `checkpoints/`: reward models under `checkpoints/reward_models/`; generation ckpts under model-specific dirs
- `plots/`: generated figures and `summary_statistics.csv`
- `logs/`: training/eval logs

## Quick Start (minimal path)
1) Data prep steps 1–4 to `data/pair_pref_div`.
2) Safety annotate → `results/annotations/safety_flags.jsonl`; generate high-risk → `data/annotations/high_risk_ids.json`.
3) Train reward model (optional if using provided) → `checkpoints/reward_models/...`.
4) Train SafeORPO: `generation_safeorpo_model_train.py` with high-risk IDs and diversity models.
5) Eval: `generation_eval1_1.py` (+ `generation_eval1_2.py`, `generation_eval2.py`).
6) Plot: `plot_safeorpo_results.py`, `plot_diversity_comparison.py`.