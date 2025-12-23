import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import evaluate
from tqdm import tqdm

def evaluate_model(model_path: str, data_dir: str, batch_size: int = 16):
    """
    Evaluate the fine-tuned model on the test set using BLEU, ROUGE, and METEOR.
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Loading data from {data_dir}...")
    dataset = load_from_disk(data_dir)
    test_dataset = dataset["validation"] # SQuAD only has train/validation

    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bert_score = evaluate.load("bertscore")

    print("Generating predictions...")
    predictions = []
    references = []

    # Iterate in batches
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i : i + batch_size]
        
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64, 
                num_beams=4, 
                early_stopping=True
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Labels are already tokenized in the dataset, we need to decode them or use the original text if available
        # The processed dataset has 'labels' which are token ids.
        # However, for metric calculation, we need string references.
        # Let's decode the labels from the batch.
        decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        predictions.extend(decoded_preds)
        references.extend(decoded_labels)

    print("Computing metrics...")
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    meteor_score = meteor.compute(predictions=predictions, references=references)
    bert_score_res = bert_score.compute(predictions=predictions, references=references, lang="en")

    print("\nResults:")
    print(f"BLEU: {bleu_score['bleu']:.4f}")
    print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
    print(f"METEOR: {meteor_score['meteor']:.4f}")
    print(f"BERTScore (Precision): {sum(bert_score_res['precision']) / len(bert_score_res['precision']):.4f}")
    print(f"BERTScore (Recall): {sum(bert_score_res['recall']) / len(bert_score_res['recall']):.4f}")
    print(f"BERTScore (F1): {sum(bert_score_res['f1']) / len(bert_score_res['f1']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/quillgen-t5-base", help="Path to fine-tuned model")
    parser.add_argument("--data_dir", type=str, default="./data/processed_squad", help="Data directory")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_dir)
