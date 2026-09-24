import re
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import Counter

LOCAL_MODEL = "./models/base/"

FEW_SHOT_COT_PROMPT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

"""

FEW_SHOT_DIRECT_PROMPT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: The answer is 6.
Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: The answer is 5.
Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: The answer is 39.
Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: The answer is 8.

"""

parser = argparse.ArgumentParser(description="Benchmark the base model")
parser.add_argument("--num-samples", type=int, required=False, default=-1, help="number of samples to test on")
parser.add_argument("--batch-size", type=int, required=False, default=8, help="batch size for evaluation")
parser.add_argument("--mode", type=str, required=True, choices=["cot-few", "cot-zero", "direct"], help="which prompting mode to use")
parser.add_argument("--self-consistency", type=int, required=False, default=1, help="self-consistency sampling")

def get_model_and_tokenizer(path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()

    return model, tokenizer

def get_gsm8k_dataset_splits() -> tuple[Dataset, Dataset]:
    ds = load_dataset("openai/gsm8k", "main")
    train = ds['train']
    test = ds['test']
    return train, test

def build_direct_prompt(question: str) -> str:
    return f"{FEW_SHOT_DIRECT_PROMPT}Question: {question}\nAnswer:"

def build_cot_few_shot_prompt(question: str) -> str:
    return f"{FEW_SHOT_COT_PROMPT}Question: {question}\nAnswer:"

def build_cot_zero_shot_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer: Let's think step by step.\n"

def extract_ground_truth(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")

def extract_predicted_answer(completion: str) -> str:
    # Truncate at next question if generated
    completion = completion.split("Question:")[0].strip()

    matched = re.search(r"[Tt]he answer is (?:roughly )?\$?(-?[0-9]+(?:\.[0-9]+)?)", completion)
    if matched:
        return matched.group(1).replace(",", "")

    matched = re.search(r"####?\s*(-?[0-9]+(?:\.[0-9]+)?)", completion)
    if matched:
        return matched.group(1).replace(",", "")

    numbers = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", completion)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return ""


def bench(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    prompt_builder: callable,
    self_consistency: int = 1,
    num_samples: int = None,
    batch_size: int = 8,
) -> tuple[float, int, int]:
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))
    
    dataset = dataset.map(
        lambda x: {"prompt": prompt_builder(x["question"])},
    )

    correct = 0
    total = len(dataset)
    processed = 0
    print(f"Benchmarking on {total} samples with batch size {batch_size}...")

    if self_consistency > 1:
        print(f"Using Self Consistency with N={self_consistency}")

    # Iterate through dataset in batches
    for start_idx in tqdm(range(0, total, batch_size), desc="Benchmarking"):
        end_idx = min(start_idx + batch_size, total)
        batch_slice = dataset.select(range(start_idx, end_idx))
        prompts = [s["prompt"] for s in batch_slice]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=["\nQuestion:", "\n\nQuestion:"],
                tokenizer=tokenizer,
                num_return_sequences=self_consistency,
            )
        
        # Left-padding means prompt tokens occupy [:, :input_len]
        input_len = inputs.input_ids.shape[1]
        completions = tokenizer.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True
        )

        completions = [completions[i:i+self_consistency] for i in range(0, len(completions), self_consistency)]

        for sample, completions in zip(batch_slice, completions):
            processed += 1
            pred = Counter([extract_predicted_answer(c) for c in completions]).most_common(1)[0][0]
            truth = extract_ground_truth(sample["answer"])

            if pred == truth:
                correct += 1

            if processed <= 5 or processed % 50 == 0 or processed == total:
                print(f"Sample {processed}: Pred='{pred}' | Truth='{truth}' | Match={pred == truth}")
                print(f"Running Accuracy: {correct / processed * 100:.2f}% ({correct}/{processed})\n")
        
    accuracy = correct / total
    print(f"Final Accuracy: {accuracy * 100:.2f}%")

    return accuracy, correct, total


if __name__ == "__main__":
    train, test = get_gsm8k_dataset_splits()
    model, tokenizer = get_model_and_tokenizer(LOCAL_MODEL)

    args = parser.parse_args()

    num_samples = args.num_samples
    if num_samples <= 0:
        num_samples = None

    if args.mode == "cot-few":
        prompt_builder = build_cot_few_shot_prompt
    elif args.mode == "cot-zero":
        prompt_builder = build_cot_zero_shot_prompt
    elif args.mode == "direct":
        prompt_builder = build_direct_prompt
    else:
        prompt_builder = build_direct_prompt

    self_consistency = args.self_consistency

    accuracy, correct, total = bench(
        model, 
        tokenizer, 
        test,
        prompt_builder,
        num_samples=num_samples, 
        batch_size=args.batch_size,
        self_consistency=self_consistency,
    )
    title = ""
    if self_consistency > 1:
        title += f" + SELF CONSISTENCY {self_consistency}"
    if args.mode == "cot-few":
        title += " + COT"
    elif args.mode == "cot-zero":
        title += " + COT-Zero-Shot"
    elif args.mode == "direct":
        title += " + DIRECT"
    
    print(f"\n\n==================FINAL BASE MODEL BENCHMARK RESULTS ({title})==================\n")
    print(f"Correct: {correct}/{total}")
    print(f"Error: {(total-correct)/total:.2f}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\n======================================================================\n")