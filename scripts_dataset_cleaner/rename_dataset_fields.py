#!/usr/bin/env python3
"""
Rename fields in writingPrompt_cleaned to add 'filtered_' prefix
for SFT and Reward training compatibility
"""
import datasets
import pyarrow as pa
import glob
import os
from datasets import Dataset, DatasetDict

def load_dataset_from_arrow(dataset_path):
    """Load dataset directly from arrow files"""
    ds_dict = {}
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            arrow_files = sorted(glob.glob(os.path.join(split_path, '*.arrow')))
            if arrow_files:
                tables = []
                for arrow_file in arrow_files:
                    try:
                        with open(arrow_file, 'rb') as f:
                            reader = pa.ipc.open_stream(f)
                            table = reader.read_all()
                            tables.append(table)
                    except Exception as e:
                        print(f"Error loading {arrow_file}: {e}")
                        continue
                
                if tables:
                    combined_table = pa.concat_tables(tables)
                    # Create dataset without metadata to avoid issues
                    ds_dict[split] = Dataset.from_dict({
                        col: combined_table[col].to_pylist() 
                        for col in combined_table.column_names
                    })
    
    return DatasetDict(ds_dict) if ds_dict else None

def rename_fields(dataset_path, output_path=None):
    """Rename fields to add filtered_ prefix"""
    if output_path is None:
        output_path = dataset_path
    
    print(f"Loading dataset from {dataset_path}...")
    ds = load_dataset_from_arrow(dataset_path)
    
    if ds is None:
        print("Failed to load dataset!")
        return False
    
    print(f"Original columns: {ds['train'].column_names}")
    
    # Rename fields
    def rename_example(example):
        # Rename comment_texts -> filtered_comment_texts
        if 'comment_texts' in example and 'filtered_comment_texts' not in example:
            example['filtered_comment_texts'] = example['comment_texts']
        
        # Rename transformed_score -> filtered_transformed_scores
        if 'transformed_score' in example and 'filtered_transformed_scores' not in example:
            example['filtered_transformed_scores'] = example['transformed_score']
        
        return example
    
    print("Renaming fields...")
    ds_renamed = ds.map(rename_example)
    
    print(f"New columns: {ds_renamed['train'].column_names}")
    
    # Save the renamed dataset
    print(f"Saving to {output_path}...")
    ds_renamed.save_to_disk(output_path)
    
    print("✓ Done!")
    return True

if __name__ == "__main__":
    dataset_path = "data/writingPrompt_cleaned"
    rename_fields(dataset_path)













