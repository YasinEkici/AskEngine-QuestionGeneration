import os
import argparse
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_from_disk

def train(
    model_name: str,
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int = 1,
    max_steps: int = -1,
):
    """
    Fine-tune a T5 model on the processed SQuAD dataset.
    """
    print(f"Loading data from {data_dir}...")
    dataset = load_from_disk(data_dir)
    
    print(f"Loading model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True, # Save memory by recomputing activations
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs if max_steps == -1 else 3.0, # Default to 3 epochs if max_steps is set, but max_steps will override
        max_steps=max_steps,
        predict_with_generate=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none", # Disable wandb/mlflow for now unless requested
        fp16=False,
        bf16=True, # Use BF16 for RTX 5080 (Ampere+) for better stability
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving best model to {output_dir}...")
    trainer.save_model(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="t5-large", help="Model name")
    parser.add_argument("--data_dir", type=str, default="./data/processed_squad", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./models/quillgen-t5-large", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max number of training steps (overrides epochs)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    train(
        args.model_name,
        args.data_dir,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.gradient_accumulation_steps,
        args.max_steps,
    )
