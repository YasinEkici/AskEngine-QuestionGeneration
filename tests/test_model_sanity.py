import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pytest

# Mark as slow or requiring GPU/Internet
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_real_model_generation_sanity():
    """
    Smoke test: Loads a small real model to verify the generation pipeline 
    (torch, cuda, transformers) works without crashing.
    """
    model_name = "t5-small"
    device = "cuda"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    except Exception as e:
        pytest.skip(f"Failed to load model (network issue?): {e}")

    context = "The quick brown fox jumps over the lazy dog."
    answer = "fox"
    input_text = f"answer: {answer} context: {context}"
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    ).to(device)

    # Test with custom parameters to ensure no runtime errors
    outputs = model.generate(
        **inputs, 
        max_length=32, 
        num_beams=2, 
        early_stopping=True
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0
    print(f"\nGenerated Sanity Output: {generated_text}")
