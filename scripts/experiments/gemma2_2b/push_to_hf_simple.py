#!/usr/bin/env python3
"""
Simplified script to push trained model checkpoints to HuggingFace Hub.
Uses GenerationModel to properly load and save models.
"""

import os
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models import GenerationModel
from transformers import AutoTokenizer
from huggingface_hub import login, create_repo, HfApi
from peft import PeftModel

# Model naming mapping
MODEL_NAMES = {
    "SFT": "gemma2-2b-sft",
    "DDPO_sem": "gemma2-2b-ddpo-sem",
    "DDPO_sty": "gemma2-2b-ddpo-sty", 
    "DDPO_both": "gemma2-2b-ddpo-both",
    "DORPO_sem": "gemma2-2b-dorpo-sem",
    "DORPO_sty": "gemma2-2b-dorpo-sty",
    "DORPO_both": "gemma2-2b-dorpo-both",
}

BASE_MODEL = "google/gemma-2-2b-it"
HF_USERNAME = "ssmurali-cmu"
CHECKPOINT_BASE = "checkpoints/gemma-2-2b-it"

def push_model_to_hub(model_path, repo_name, hf_token, is_pt_file=False):
    """Push a model checkpoint to HF Hub"""
    print(f"\n{'='*60}")
    print(f"📦 Processing: {repo_name}")
    print(f"   Path: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Create repo if it doesn't exist
        api = HfApi(token=hf_token)
        try:
            create_repo(f"{HF_USERNAME}/{repo_name}", token=hf_token, exist_ok=True, repo_type="model")
            print(f"✅ Created/verified repo: {HF_USERNAME}/{repo_name}")
        except Exception as e:
            print(f"⚠️  Repo creation note: {e}")
        
        # Load model using GenerationModel
        print("🔄 Loading model...")
        if is_pt_file:
            generation_model = GenerationModel(
                BASE_MODEL,
                lora_r=128,
                lora_alpha=256,
                lora_model_path=model_path,
                device="cpu"  # Use CPU to save memory
            )
        else:
            # For checkpoint-final directories, load via PeftModel
            from transformers import AutoModelForCausalLM
            import torch
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            )
            peft_model = PeftModel.from_pretrained(base_model, model_path)
            generation_model = type('obj', (object,), {
                'model': peft_model,
                'tokenizer': AutoTokenizer.from_pretrained(BASE_MODEL)
            })()
        
        # Save to temporary directory
        temp_dir = f"/tmp/{repo_name}"
        os.makedirs(temp_dir, exist_ok=True)
        print(f"💾 Saving to {temp_dir}...")
        
        if is_pt_file:
            generation_model.model.save_pretrained(temp_dir)
        else:
            generation_model.model.save_pretrained(temp_dir)
        
        generation_model.tokenizer.save_pretrained(temp_dir)
        
        # Push to hub
        print(f"🚀 Pushing to hub: {HF_USERNAME}/{repo_name}...")
        generation_model.model.push_to_hub(
            f"{HF_USERNAME}/{repo_name}",
            token=hf_token,
            use_auth_token=True
        )
        generation_model.tokenizer.push_to_hub(
            f"{HF_USERNAME}/{repo_name}",
            token=hf_token,
            use_auth_token=True
        )
        
        print(f"✅ Successfully pushed {repo_name}!")
        return True
        
    except Exception as e:
        print(f"❌ Error pushing {repo_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Push model checkpoints to HuggingFace Hub")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--model_type", type=str, choices=list(MODEL_NAMES.keys()) + ["all"], 
                       default="all", help="Which model to push (default: all)")
    parser.add_argument("--base_path", type=str, default=CHECKPOINT_BASE, 
                       help="Base path to checkpoints")
    
    args = parser.parse_args()
    
    # Login to HF
    print("🔐 Logging in to HuggingFace...")
    login(token=args.hf_token)
    
    # Determine which models to push
    if args.model_type == "all":
        models_to_push = list(MODEL_NAMES.keys())
    else:
        models_to_push = [args.model_type]
    
    base_path = Path(args.base_path)
    results = {}
    
    for model_key in models_to_push:
        repo_name = MODEL_NAMES[model_key]
        
        # Check for SFT (.pt file)
        if model_key == "SFT":
            pt_path = base_path / f"generation_models_{model_key}" / f"google_gemma-2-2b-it_epoch_final.pt"
            if pt_path.exists():
                success = push_model_to_hub(str(pt_path), repo_name, args.hf_token, is_pt_file=True)
                results[model_key] = "✅ Success" if success else "❌ Failed"
            else:
                print(f"⚠️  Checkpoint not found: {pt_path}")
                results[model_key] = "⚠️  Not found"
        
        # Check for DDPO/DORPO (checkpoint-final directory)
        else:
            checkpoint_dir = base_path / f"generation_models_{model_key}" / "checkpoint-final"
            if checkpoint_dir.exists() and checkpoint_dir.is_dir():
                success = push_model_to_hub(str(checkpoint_dir), repo_name, args.hf_token, is_pt_file=False)
                results[model_key] = "✅ Success" if success else "❌ Failed"
            else:
                print(f"⚠️  Checkpoint not found: {checkpoint_dir}")
                results[model_key] = "⚠️  Not found"
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 Summary:")
    print(f"{'='*60}")
    for model_key, status in results.items():
        print(f"  {model_key:15} -> {MODEL_NAMES[model_key]:25} {status}")

if __name__ == "__main__":
    main()






