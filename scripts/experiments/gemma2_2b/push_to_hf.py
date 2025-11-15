#!/usr/bin/env python3
"""
Script to push trained model checkpoints to HuggingFace Hub.
ONLY pushes adapter weights, does NOT load the full base model.
"""

import os
import argparse
from pathlib import Path
import shutil
import json
import torch
from huggingface_hub import HfApi, login, create_repo, upload_folder
from transformers import AutoTokenizer

# Model naming mapping
MODEL_NAMES = {
    "SFT": "gemma2-2b-sft",
    # "DDPO_sem": "gemma2-2b-ddpo-sem",
    # "DDPO_sty": "gemma2-2b-ddpo-sty", 
    # "DDPO_both": "gemma2-2b-ddpo-both",
    "DORPO_sem": "gemma2-2b-dorpo-sem",
    "DORPO_sty": "gemma2-2b-dorpo-sty",
    "DORPO_both": "gemma2-2b-dorpo-both",
}

BASE_MODEL = "google/gemma-2-2b-it"
HF_USERNAME = "ssmurali"
CHECKPOINT_BASE = "checkpoints/gemma-2-2b-it"

def convert_pt_to_peft_format(pt_path, output_dir):
    """Convert .pt file to PEFT adapter format without loading base model"""
    print(f"   Converting .pt to PEFT format...")
    
    # Load the .pt file
    lora_state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
    
    # Transform keys (handle the lora_A/lora_B key format)
    new_state_dict = {}
    for key, value in lora_state_dict.items():
        if 'lora_A.weight' in key:
            new_key = key.replace('lora_A.weight', 'lora_A.default.weight')
            new_state_dict[new_key] = value
        elif 'lora_B.weight' in key:
            new_key = key.replace('lora_B.weight', 'lora_B.default.weight')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Save adapter_model.bin
    torch.save(new_state_dict, os.path.join(output_dir, "adapter_model.bin"))
    
    # Create adapter_config.json
    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": 128,
        "lora_alpha": 256,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
    }
    
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    print(f"   ✅ Converted to PEFT format")

def push_checkpoint_dir_to_hub(checkpoint_dir, repo_name, hf_token, hf_username):
    """Push a checkpoint-final directory to HF Hub (just copies adapter files)"""
    print(f"\n{'='*60}")
    print(f"📦 Pushing adapter: {repo_name}")
    print(f"   Source: {checkpoint_dir}")
    print(f"{'='*60}")
    
    try:
        # Create repo if it doesn't exist
        api = HfApi(token=hf_token)
        try:
            create_repo(f"{hf_username}/{repo_name}", token=hf_token, exist_ok=True, repo_type="model")
            print(f"✅ Created/verified repo: {hf_username}/{repo_name}")
        except Exception as e:
            print(f"⚠️  Repo creation note: {e}")
        
        # Check if adapter files exist (handle both .bin and .safetensors formats)
        adapter_weight_file = None
        if os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")):
            adapter_weight_file = "adapter_model.safetensors"
        elif os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin")):
            adapter_weight_file = "adapter_model.bin"
        
        adapter_config_file = "adapter_config.json"
        
        if not adapter_weight_file:
            print(f"⚠️  Checking directory contents...")
            existing_files = os.listdir(checkpoint_dir)
            print(f"   Files found: {existing_files}")
            print(f"❌ Missing adapter weight file (adapter_model.bin or adapter_model.safetensors)")
            return False
        
        if not os.path.exists(os.path.join(checkpoint_dir, adapter_config_file)):
            print(f"❌ Missing adapter config file: {adapter_config_file}")
            return False
        
        adapter_files = [adapter_weight_file, adapter_config_file]
        
        # Load tokenizer (lightweight, no model needed)
        print("💾 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        # Save tokenizer to temp dir
        temp_dir = f"/tmp/{repo_name}"
        os.makedirs(temp_dir, exist_ok=True)
        tokenizer.save_pretrained(temp_dir)
        
        # Copy adapter files to temp dir
        print("📋 Copying adapter files...")
        for file in adapter_files:
            shutil.copy2(
                os.path.join(checkpoint_dir, file),
                os.path.join(temp_dir, file)
            )
        
        # Create README
        readme_content = f"""---
base_model: {BASE_MODEL}
library_name: peft
---
# {repo_name}

This is a PEFT LoRA adapter for {BASE_MODEL}.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}")
model = PeftModel.from_pretrained(base_model, "{hf_username}/{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}")
```
"""
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write(readme_content)
        
        # Push to hub
        print(f"🚀 Pushing to hub: {hf_username}/{repo_name}...")
        upload_folder(
            folder_path=temp_dir,
            repo_id=f"{hf_username}/{repo_name}",
            token=hf_token,
            repo_type="model"
        )
        
        print(f"✅ Successfully pushed {repo_name}!")
        return True
        
    except Exception as e:
        print(f"❌ Error pushing {repo_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def push_pt_file_to_hub(pt_path, repo_name, hf_token, hf_username):
    """Push a .pt LoRA checkpoint to HF Hub (converts to PEFT format first)"""
    print(f"\n{'='*60}")
    print(f"📦 Pushing adapter: {repo_name}")
    print(f"   Source: {pt_path}")
    print(f"{'='*60}")
    
    try:
        # Create repo if it doesn't exist
        api = HfApi(token=hf_token)
        try:
            create_repo(f"{hf_username}/{repo_name}", token=hf_token, exist_ok=True, repo_type="model")
            print(f"✅ Created/verified repo: {hf_username}/{repo_name}")
        except Exception as e:
            print(f"⚠️  Repo creation note: {e}")
        
        # Convert .pt to PEFT format
        temp_dir = f"/tmp/{repo_name}"
        os.makedirs(temp_dir, exist_ok=True)
        convert_pt_to_peft_format(pt_path, temp_dir)
        
        # Load tokenizer (lightweight, no model needed)
        print("💾 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.save_pretrained(temp_dir)
        
        # Create README
        readme_content = f"""---
base_model: {BASE_MODEL}
library_name: peft
---
# {repo_name}

This is a PEFT LoRA adapter for {BASE_MODEL}.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}")
model = PeftModel.from_pretrained(base_model, "{hf_username}/{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}")
```
"""
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write(readme_content)
        
        # Push to hub
        print(f"🚀 Pushing to hub: {hf_username}/{repo_name}...")
        upload_folder(
            folder_path=temp_dir,
            repo_id=f"{hf_username}/{repo_name}",
            token=hf_token,
            repo_type="model"
        )
        
        print(f"✅ Successfully pushed {repo_name}!")
        return True
        
    except Exception as e:
        print(f"❌ Error pushing {repo_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Push model checkpoints to HuggingFace Hub (adapter only)")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--model_type", type=str, choices=list(MODEL_NAMES.keys()) + ["all"], 
                       default="all", help="Which model to push (default: all)")
    parser.add_argument("--base_path", type=str, default=CHECKPOINT_BASE, 
                       help="Base path to checkpoints")
    parser.add_argument("--hf_username", type=str, default=HF_USERNAME,
                       help="HuggingFace username/org")
    
    args = parser.parse_args()
    hf_username = args.hf_username
    
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
                success = push_pt_file_to_hub(str(pt_path), repo_name, args.hf_token, hf_username)
                results[model_key] = "✅ Success" if success else "❌ Failed"
            else:
                print(f"⚠️  Checkpoint not found: {pt_path}")
                results[model_key] = "⚠️  Not found"
        
        # Check for DDPO/DORPO (checkpoint-final directory)
        else:
            checkpoint_dir = base_path / f"generation_models_{model_key}" / "checkpoint-final"
            if checkpoint_dir.exists() and checkpoint_dir.is_dir():
                success = push_checkpoint_dir_to_hub(str(checkpoint_dir), repo_name, args.hf_token, hf_username)
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
