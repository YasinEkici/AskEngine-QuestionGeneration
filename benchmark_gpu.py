import torch
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def benchmark_gpu():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("No GPU found! Exiting.")
        return

    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Measure memory before
    print(f"Initial Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("Loading t5-large model...")
    start_load = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
    print(f"Model loaded in {time.time() - start_load:.2f}s")
    
    print("Moving model to GPU (BF16)...")
    model = model.bfloat16().to(device)
    print(f"Model moved to GPU. Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    input_text = "translate English to German: Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    print("Running warmup pass...")
    with torch.no_grad():
        model.generate(**inputs)
    
    print("Running 10 forward passes...")
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            model.generate(**inputs)
    
    duration = time.time() - start_time
    print(f"10 passes took {duration:.2f}s ({duration/10:.2f}s per pass)")
    
    print("Done!")

if __name__ == "__main__":
    benchmark_gpu()
