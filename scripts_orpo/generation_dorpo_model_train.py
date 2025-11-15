from datasets import load_from_disk
from trl import ORPOConfig
from models import GenerationModel, DORPOTrainer

from argparse import ArgumentParser

from accelerate import Accelerator
import pyarrow as pa
import glob
import os
from datasets import Dataset, DatasetDict

def load_dataset_from_arrow(dataset_path):
    """Load dataset directly from arrow files, bypassing corrupted metadata"""
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
                    except:
                        continue
                
                if tables:
                    combined_table = pa.concat_tables(tables)
                    ds_dict[split] = Dataset.from_dict({
                        col: combined_table[col].to_pylist() 
                        for col in combined_table.column_names
                    })
    
    return DatasetDict(ds_dict) if ds_dict else None

parser = ArgumentParser()

parser.add_argument("--modelname", default="meta-llama/Llama-3.1-8B", type=str, help="Model to use for training")
parser.add_argument("--output_dir", default="checkpoints/Llama-3.1-8B/generation_models_DORPO_sem_sty", type=str, help="Path to the model")

parser.add_argument("--dataset", default="data/writingPrompt_post_pair_sem_sty", type=str, help="Path to the dataset")

parser.add_argument("--learning_rate", default=5e-6, type=float, help="Learning rate for the model")
parser.add_argument("--num_train_epochs", default=4, type=int, help="Number of training epochs")

parser.add_argument("--batch_size", default=2, type=int, help="Batch size for training")

args = parser.parse_args()

generation_model = GenerationModel(args.modelname, lora_r = 128,
  lora_alpha = 256, 
)

# Try loading dataset normally first, fallback to arrow loading if metadata is corrupted
try:
dataset = load_from_disk(args.dataset)
except Exception as e:
    print(f"Failed to load dataset normally: {e}")
    print("Attempting to load from arrow files directly...")
    dataset = load_dataset_from_arrow(args.dataset)
    if dataset is None:
        raise RuntimeError(f"Failed to load dataset from {args.dataset}")

train_dataset = dataset["train"]
test_dataset = dataset["test"]

accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(
  project_name = "CWTuning_DORPO_Gemma2_2B",
)

training_args = ORPOConfig(
  output_dir=args.output_dir, 
  logging_steps=10, 
  learning_rate=args.learning_rate, 
  num_train_epochs=args.num_train_epochs, 
  per_device_train_batch_size=args.batch_size,
  per_device_eval_batch_size=args.batch_size,
  report_to = "wandb",
  bf16=True,
  bf16_full_eval=True,
  eval_strategy = "epoch",
  save_strategy = "epoch",
  beta = 0.25, # parameter to care about
)
trainer = DORPOTrainer(
  model=generation_model.model, 
  args=training_args, 
  processing_class=generation_model.tokenizer, 
  train_dataset=train_dataset,
  eval_dataset=test_dataset,
)

trainer.train()

# save the model
trainer.save_model(args.output_dir+"/checkpoint-final")