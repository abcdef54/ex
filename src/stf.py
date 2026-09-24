import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig
from trl import SFTTrainer, SFTConfig
MODEL_PATH = "./models/generals/"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1, required=False, help="Number of epochs to train")
parser.add_argument("--eval-step", type=int, default=100, required=False, help="Evaluate every N steps")

QLORA_SETTINGS = {
    "rank": 32,
    "alpha": 64,
    "dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

SFT_CONFIGS = {
    "epochs": 1,
    "lr": 2e-4,
    "optimizer": "paged_adamw_8bit",
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "bf16": True,
    "gradient_accumulation_steps": 2,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_checkpointing": True,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}


def get_model():
    quantization_config = get_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)
    return model

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def get_quantization_config():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    return quantization_config

def get_peft_config():
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        lora_alpha=64,
        r=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=QLORA_SETTINGS["target_modules"]
    )
    return peft_config

def get_training_args():
    training_args = SFTConfig(
        output_dir="./results/",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        weight_decay=0.1,
        max_grad_norm=0.3,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        gradient_checkpointing=True,
        max_length=16384
    )
    return training_args

def format_prompt(sample):
    return f"Question: {sample['question']}\n\nAnswer: {sample['answer']}"

def get_trainer(train_dataset=None, eval_dataset=None):
    model = get_model()
    tokenizer = get_tokenizer()
    peft_config = get_peft_config()
    training_args = get_training_args()
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        peft_config=peft_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=format_prompt
    )
    return trainer

def get_splits():
    train = load_dataset("json", data_files="./data/SFT/open_thought_10k_train.jsonl", split="train")
    val = load_dataset("json", data_files="./data/SFT/open_thought_1k_val.jsonl", split="train")

    original_train_len = len(train)
    original_val_len = len(val)

    tokenizer = get_tokenizer()
    max_len = 16384

    def is_within_limit(sample):
        text = f"Question: {sample['question']}\n\nAnswer: {sample['answer']}"
        return len(tokenizer.encode(text)) < max_len

    train = train.filter(is_within_limit)
    val = val.filter(is_within_limit)

    final_train_len = len(train)
    final_val_len = len(val)

    print(f"Original train length: {original_train_len}")
    print(f"Original val length: {original_val_len}")
    print(f"Final train length: {final_train_len}")
    print(f"Final val length: {final_val_len}")

    print(f"Train removed: {original_train_len - final_train_len}")
    print(f"Val removed: {original_val_len - final_val_len}")
    

    return train, val


if __name__ == "__main__":
    train, val = get_splits()

    trainer = get_trainer(train_dataset=train, eval_dataset=val)

    trainer.train()



