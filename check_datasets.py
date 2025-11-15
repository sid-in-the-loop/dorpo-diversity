#!/usr/bin/env python3
"""
Check all datasets for SFT, Reward, DDPO, DORPO training
Verify required fields exist by reading arrow files directly
"""
import pyarrow as pa
import os
import sys
import glob

datasets_to_check = {
    'SFT': {
        'path': 'data/writingPrompt_cleaned',
        'required_fields': ['post_title', 'post_text', 'filtered_comment_texts', 'filtered_transformed_scores']
    },
    'Reward': {
        'path': 'data/writingPrompt_cleaned',
        'required_fields': ['post_title', 'post_text', 'filtered_comment_texts', 'filtered_transformed_scores']
    },
    'DDPO': {
        'path': 'data/writingPrompt_post_pair_sem_sty',
        'required_fields': ['chosen', 'rejected', 'score_chosen', 'score_rejected']
    },
    'DORPO': {
        'path': 'data/writingPrompt_post_pair_sem_sty',
        'required_fields': ['chosen', 'rejected', 'score_chosen', 'score_rejected']
    }
}

def check_dataset_from_arrow(dataset_path):
    """Check dataset by reading arrow files directly"""
    results = {}
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            arrow_files = sorted(glob.glob(os.path.join(split_path, '*.arrow')))
            if arrow_files:
                try:
                    # Load first arrow file to check structure
                    with open(arrow_files[0], 'rb') as f:
                        reader = pa.ipc.open_stream(f)
                        table = reader.read_all()
                    
                    columns = table.column_names
                    num_rows = len(table)
                    
                    # Get sample data
                    sample = {}
                    for col in columns[:5]:  # Check first 5 columns
                        col_data = table[col].to_pylist()
                        if len(col_data) > 0:
                            sample[col] = col_data[0]
                    
                    results[split] = {
                        'columns': columns,
                        'num_rows': num_rows,
                        'sample': sample
                    }
                except Exception as e:
                    results[split] = {'error': str(e)}
        else:
            results[split] = {'error': 'Split path does not exist'}
    
    return results

print('=' * 80)
print('Dataset Verification Report')
print('=' * 80)
print()

all_ok = True

for method, info in datasets_to_check.items():
    print(f'[{method}]')
    print(f'  Dataset path: {info["path"]}')
    
    if not os.path.exists(info['path']):
        print(f'  ❌ Dataset does NOT exist!')
        all_ok = False
        print()
        continue
    
    try:
        results = check_dataset_from_arrow(info['path'])
        
        if not results:
            print(f'  ❌ Could not read dataset files')
            all_ok = False
            print()
            continue
        
        print(f'  ✓ Dataset files found')
        
        # Check train split
        if 'train' in results:
            train_info = results['train']
            if 'error' in train_info:
                print(f'  ❌ Train split error: {train_info["error"]}')
                all_ok = False
            else:
                train_cols = train_info['columns']
                print(f'  Train columns: {train_cols}')
                print(f'  Train size: {train_info["num_rows"]}')
                
                missing_fields = [f for f in info['required_fields'] if f not in train_cols]
                if missing_fields:
                    print(f'  ❌ Missing required fields: {missing_fields}')
                    print(f'  Available fields: {train_cols}')
                    all_ok = False
                else:
                    print(f'  ✓ All required fields present')
                    
                    # Show sample structure
                    if train_info['sample']:
                        print(f'  Sample data (first row):')
                        for field in info['required_fields']:
                            if field in train_info['sample']:
                                val = train_info['sample'][field]
                                if isinstance(val, list):
                                    print(f'    - {field}: list with {len(val)} items')
                                    if len(val) > 0:
                                        print(f'      First item type: {type(val[0]).__name__}')
                                else:
                                    print(f'    - {field}: {type(val).__name__}')
        
        # Check test split
        if 'test' in results:
            test_info = results['test']
            if 'error' in test_info:
                print(f'  ❌ Test split error: {test_info["error"]}')
                all_ok = False
            else:
                test_cols = test_info['columns']
                print(f'  Test columns: {test_cols}')
                print(f'  Test size: {test_info["num_rows"]}')
                
                missing_fields = [f for f in info['required_fields'] if f not in test_cols]
                if missing_fields:
                    print(f'  ❌ Missing required fields in test: {missing_fields}')
                    all_ok = False
                else:
                    print(f'  ✓ All required fields present in test')
        
    except Exception as e:
        print(f'  ❌ Error checking dataset: {e}')
        import traceback
        traceback.print_exc()
        all_ok = False
    
    print()

print('=' * 80)
if all_ok:
    print('✓ All datasets verified successfully!')
else:
    print('❌ Some datasets have issues. See above for details.')
    print()
    print('NOTE: If datasets are missing required fields, you may need to:')
    print('  - Run the data processing pipeline (steps 0-4)')
    print('  - Check that filtered_comment_texts exists (requires step 1)')
    print('  - Check that writingPrompt_post_pair_sem_sty exists (requires step 4)')
print('=' * 80)

sys.exit(0 if all_ok else 1)
