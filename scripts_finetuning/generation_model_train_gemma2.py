from models import GenerationModel
import datasets

generation_model = GenerationModel(
  "google/gemma-2-2b-it", 
  lora_r = 128,
  lora_alpha = 256,
  save_path = "checkpoints/gemma-2-2b-it/generation_models_SFT",
)

ds = datasets.load_from_disk("data/writingPrompt_cleaned")

generation_model.train_with_finetuning(ds["train"], ds["test"], epoch=1, batch_size=1, 
  lr=3e-5, 
  gradient_accumulation_steps=4,
  accelerate_config={
    "project_name": "CWTuning_SFT_Gemma2_2B",
    "config": {},
  }
)

