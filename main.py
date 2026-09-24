from datasets import load_dataset

if __name__ == "__main__":
    train = load_dataset("openai/gsm8k", split="train")
    test = load_dataset("openai/gsm8k", split="test")

    