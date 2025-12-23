import pytest
from transformers import AutoTokenizer
from datasets import Dataset

def test_tokenizer_loading():
    """Test if tokenizer loads correctly."""
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    assert tokenizer is not None

def test_data_formatting():
    """Test if data is formatted correctly for T5."""
    example = {
        "context": "My name is Sarah and I live in London.",
        "answers": {"text": ["London"], "answer_start": [31]},
        "question": "Where does Sarah live?"
    }
    
    formatted_input = f"answer: {example['answers']['text'][0]} context: {example['context']}"
    expected_input = "answer: London context: My name is Sarah and I live in London."
    
    assert formatted_input == expected_input

def test_tokenization_shape():
    """Test if tokenization produces expected output keys."""
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    inputs = ["answer: London context: My name is Sarah."]
    targets = ["Where does Sarah live?"]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length", return_tensors="pt")
    
    assert "input_ids" in model_inputs
    assert "attention_mask" in model_inputs
    assert labels["input_ids"].shape == (1, 64)
