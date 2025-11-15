import datasets
from datasets import DatasetDict
import numpy as np
import random
from argparse import ArgumentParser
from tqdm import tqdm
import os

# Set arguments
parser = ArgumentParser()
parser.add_argument("--data_dir", default="data", type=str, help="Base data directory")
parser.add_argument("--train_size", default=8000, type=int, help="Number of training samples")
parser.add_argument("--test_size", default=1000, type=int, help="Number of test samples")
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
parser.add_argument("--output_suffix", default="_filtered", type=str, help="Suffix to add to output directory names")

args = parser.parse_args()

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)

# List of datasets to filter
dataset_names = [
    "writingPrompt_cleaned",
    "writingPrompt_filtered",
    "writingPrompt_post",
    "writingPrompt_post_pair",
    "writingPrompt_post_pair_sem_sty",
]

print("=" * 60)
print("Dataset Filtering Script")
print("=" * 60)
print(f"Train size: {args.train_size}")
print(f"Test size: {args.test_size}")
print(f"Seed: {args.seed}")
print(f"Output suffix: {args.output_suffix}")
print()

# First, check which datasets exist and get their sizes
available_datasets = {}
dataset_sizes = {}

for dataset_name in dataset_names:
    dataset_path = os.path.join(args.data_dir, dataset_name)
    if os.path.exists(dataset_path):
        try:
            print(f"Loading {dataset_name} to check sizes...")
            ds = datasets.load_from_disk(dataset_path, keep_in_memory=False)
            train_size = len(ds.get("train", []))
            test_size = len(ds.get("test", []))
            dataset_sizes[dataset_name] = {"train": train_size, "test": test_size}
            available_datasets[dataset_name] = dataset_path
            print(f"  ✓ {dataset_name}: train={train_size}, test={test_size}")
        except Exception as e:
            print(f"  ✗ {dataset_name}: Error loading - {e}")
    else:
        print(f"  ✗ {dataset_name}: Path does not exist")

print()

# Check if we have enough samples
if not available_datasets:
    print("No datasets found! Exiting.")
    exit(1)

# Find the minimum sizes across all datasets to ensure we can use the same indices
min_train_size = min([sizes["train"] for sizes in dataset_sizes.values()])
min_test_size = min([sizes["test"] for sizes in dataset_sizes.values()])

print(f"Minimum train size across all datasets: {min_train_size}")
print(f"Minimum test size across all datasets: {min_test_size}")
print()

if args.train_size > min_train_size:
    print(f"Warning: Requested train size ({args.train_size}) > minimum available ({min_train_size})")
    print(f"Using {min_train_size} samples instead.")
    args.train_size = min_train_size

if args.test_size > min_test_size:
    print(f"Warning: Requested test size ({args.test_size}) > minimum available ({min_test_size})")
    print(f"Using {min_test_size} samples instead.")
    args.test_size = min_test_size

# First, we need to get the selected prompts from the base dataset
# This ensures pair datasets are filtered based on the same prompts
base_dataset_name = "writingPrompt_cleaned"
if base_dataset_name not in available_datasets:
    # Try alternative base datasets
    for alt_base in ["writingPrompt_filtered", "writingPrompt_post"]:
        if alt_base in available_datasets:
            base_dataset_name = alt_base
            break

if base_dataset_name not in available_datasets:
    print("ERROR: No base dataset found to extract prompts from!")
    exit(1)

print(f"Using {base_dataset_name} as base dataset to extract prompts...")
base_ds = datasets.load_from_disk(available_datasets[base_dataset_name], keep_in_memory=False)

# Generate random indices that will be used for base datasets
train_indices = sorted(random.sample(range(min_train_size), args.train_size))
test_indices = sorted(random.sample(range(min_test_size), args.test_size))

print(f"Selected {len(train_indices)} train indices and {len(test_indices)} test indices")
print(f"Train index range: {min(train_indices)} to {max(train_indices)}")
print(f"Test index range: {min(test_indices)} to {max(test_indices)}")
print()

# Extract selected prompts from base dataset
selected_prompts_train = set()
selected_prompts_test = set()

if "train" in base_ds:
    print("Extracting selected prompts from train split...")
    for idx in tqdm(train_indices):
        example = base_ds["train"][idx]
        prompt = f"{example.get('post_title', '')}\n{example.get('post_text', '')}"
        selected_prompts_train.add(prompt.strip())
    print(f"  ✓ Extracted {len(selected_prompts_train)} unique prompts")

if "test" in base_ds:
    print("Extracting selected prompts from test split...")
    for idx in tqdm(test_indices):
        example = base_ds["test"][idx]
        prompt = f"{example.get('post_title', '')}\n{example.get('post_text', '')}"
        selected_prompts_test.add(prompt.strip())
    print(f"  ✓ Extracted {len(selected_prompts_test)} unique prompts")

print()

# Process each dataset
for dataset_name, dataset_path in available_datasets.items():
    print("=" * 60)
    print(f"Processing: {dataset_name}")
    print("=" * 60)
    
    try:
        # Load dataset
        print("Loading dataset...")
        ds = datasets.load_from_disk(dataset_path, keep_in_memory=False)
        
        # Create filtered dataset
        filtered_ds = {}
        
        # Check if this is a pair dataset (has 'chosen' and 'rejected' columns)
        is_pair_dataset = "train" in ds and len(ds["train"]) > 0 and "chosen" in ds["train"].column_names
        
        if is_pair_dataset:
            print("Detected pair dataset - filtering by prompt content...")
            
            # Filter train split by prompt
            if "train" in ds:
                print(f"Filtering train split ({len(ds['train'])} -> ?)...")
                def filter_train_pairs(example):
                    prompt = example["chosen"][0]["content"]
                    return prompt.strip() in selected_prompts_train
                
                filtered_train = ds["train"].filter(filter_train_pairs)
                filtered_ds["train"] = filtered_train
                print(f"  ✓ Train: {len(filtered_ds['train'])} pairs")
            
            # Filter test split by prompt
            if "test" in ds:
                print(f"Filtering test split ({len(ds['test'])} -> ?)...")
                def filter_test_pairs(example):
                    prompt = example["chosen"][0]["content"]
                    return prompt.strip() in selected_prompts_test
                
                filtered_test = ds["test"].filter(filter_test_pairs)
                filtered_ds["test"] = filtered_test
                print(f"  ✓ Test: {len(filtered_ds['test'])} pairs")
        else:
            # Regular dataset - filter by index
            # Filter train split
            if "train" in ds:
                print(f"Filtering train split ({len(ds['train'])} -> {args.train_size})...")
                filtered_train = ds["train"].select(train_indices)
                filtered_ds["train"] = filtered_train
                print(f"  ✓ Train: {len(filtered_ds['train'])} samples")
            
            # Filter test split
            if "test" in ds:
                print(f"Filtering test split ({len(ds['test'])} -> {args.test_size})...")
                filtered_test = ds["test"].select(test_indices)
                filtered_ds["test"] = filtered_test
                print(f"  ✓ Test: {len(filtered_ds['test'])} samples")
        
        # Create DatasetDict and save
        filtered_dataset = DatasetDict(filtered_ds)
        
        output_path = os.path.join(args.data_dir, f"{dataset_name}{args.output_suffix}")
        print(f"Saving to: {output_path}")
        filtered_dataset.save_to_disk(output_path)
        print(f"  ✓ Saved successfully!")
        
    except Exception as e:
        print(f"  ✗ Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
    
    print()

print("=" * 60)
print("Filtering complete!")
print("=" * 60)
print("Filtered datasets saved with suffix:", args.output_suffix)
print()
print("Output directories:")
for dataset_name in available_datasets.keys():
    output_path = os.path.join(args.data_dir, f"{dataset_name}{args.output_suffix}")
    print(f"  - {output_path}")
