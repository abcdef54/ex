import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "./models/generals/"
adaper_path = "./results/checkpoint-590/"
output_model_path = "./models/SFT/"

print("Loading base model")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(adaper_path)

print("Merging adapter with model")
model = PeftModel.from_pretrained(base_model, adaper_path)
merged_model = model.merge_and_unload()

print(f"Saving merged model to {output_model_path}...")
merged_model.save_pretrained(output_model_path, safe_serialization=False)
tokenizer.save_pretrained(output_model_path)

print("DONE")