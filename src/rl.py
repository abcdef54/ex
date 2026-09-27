import re
import argparse
import math
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

MODEL_PATH = "./models/SFT/"
DATASET_PATH = "./data/RL/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=False, default=8, help="Batch size")
parser.add_argument("--grad-accumulation-steps", type=int, required=False, default=2, help="Gradient accumulation steps")
parser.add_argument("--max-completion-length", type=int, required=False, default=1024, help="Max completion length")
parser.add_argument("--num-generation", type=int, required=False, default=4, help="Number of generations")
parser.add_argument("--lr", type=float, required=False, default=1e-6, help="Learning rate")
parser.add_argument("--epochs", type=int, required=False, default=1, help="Number of epochs")
parser.add_argument("--output-dir", type=str, required=False, default="./results/SFT_RL/", help="Output directory")
parser.add_argument("--lora-rank", type=int, required=False, default=64, help="LoRA rank")
parser.add_argument("--lora-alpha", type=int, required=False, default=128, help="LoRA alpha")
parser.add_argument("--lora-dropout", type=float, required=False, default=0.0, help="LoRA dropout")
parser.add_argument("--loss-type", type=str, required=False, default="dapo", choices=["dapo", "sapo"], help="Loss type")

def reasoning_format_reward(completions, **kwargs):
    rewards = []

    for completion in completions:
        if isinstance(completion, list):
            completion_text = completion[0]["content"]
        else:
            completion_text = completion

        match = re.search(
            r"<think>(.*?)</think>",
            completion_text,
            flags=re.DOTALL,
        )

        if match is None:
            rewards.append(0.0)
            continue

        reasoning = match.group(1).strip()
        reasoning_token_length = len(tokenizer.encode(reasoning, add_special_tokens=False))

        reasoning_reward = (1.0 - math.exp(-reasoning_token_length / 256.0)) * 0.3

        rewards.append(reasoning_reward)

    return rewards


def get_train_eval_set() -> tuple[Dataset, Dataset]:
    train = load_dataset("json", data_files="./data/RL/deep_math_5k_train.jsonl", split="train")
    val = load_dataset("json", data_files="./data/RL/deep_math_500_val.jsonl", split="train")

    return train, val


def get_trainer(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    max_completion_length: int = 1024,
    num_generation: int = 4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.0,
    loss_type: str = "dapo",
    lr: float = 1e-6,
    epochs: int = 1,
    output_dir: str = "./results/SFT_RL/"
):
    bnb_configs = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )

    grpo_configs = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        num_generations=num_generation,
        num_completions_to_print=num_generation,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        max_completion_length=max_completion_length,
        optim="paged_adamw_8bit",
        loss_type=loss_type,
        beta=0.0,
        scale_rewards="batch",
        logging_steps=15,
        entropy_coef=0.0,
        use_adaptive_entropy=False,
        use_transformers_continuous_batching=True,
        transformers_continuous_batching_config={
            "use_cuda_graph": False,
            "max_memory_percent": 0.3
        },
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=False,
        greater_is_better=False,
        gradient_checkpointing=True
    )

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        reward_funcs=[accuracy_reward, reasoning_format_reward],
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        quantization_config=bnb_configs,
        peft_config=lora_config,
        args=grpo_configs,
    )

    return trainer


if __name__ == "__main__":
    args = parser.parse_args()

    train, val = get_train_eval_set()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = get_trainer(
        train_dataset=train,
        val_dataset=val,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_completion_length,
        num_generation=args.num_generation,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        loss_type=args.loss_type,
        lr=args.lr,
        epochs=args.epochs,
        output_dir=args.output_dir
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    
    