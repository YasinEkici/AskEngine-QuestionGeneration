import os
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

def prepare_squad(model_name: str, output_dir: str):
    """
    Load SQuAD dataset, preprocess it for Question Generation, and save to disk.
    
    Args:
        model_name: Name of the model to use for tokenization (e.g., 't5-base').
        output_dir: Directory to save the processed dataset.
    """
    print(f"Loading dataset and tokenizer for {model_name}...")
    dataset = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        inputs = [f"answer: {a['text'][0]} context: {c}" for a, c in zip(examples["answers"], examples["context"])]
        targets = [q for q in examples["question"]]
        
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing dataset...")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    print(f"Saving processed dataset to {output_dir}...")
    tokenized_datasets.save_to_disk(output_dir)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="t5-base", help="Model name for tokenizer")
    parser.add_argument("--output_dir", type=str, default="./data/processed_squad", help="Output directory")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    prepare_squad(args.model_name, args.output_dir)
