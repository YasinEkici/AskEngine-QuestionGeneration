from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from .schemas import ContextAnswerPair, GeneratedQuestion, SuggestionRequest, SuggestionResponse
from inference.ner_utils import extract_key_phrases
import os
import logging
import time
import uuid
import json
from logging.handlers import RotatingFileHandler

# --- Observability Setup ---
os.makedirs("data/logs", exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("askengine_metrics")
# File Handler (Rotate at 10MB, keep 5 backups)
handler = RotatingFileHandler("data/logs/metrics.log", maxBytes=10*1024*1024, backupCount=5)
# JSON Formatter
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, 'props'):
            log_obj.update(record.props)
        return json.dumps(log_obj)

handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# Prevent propagation to root logger to avoid dups in console if root has handlers
logger.propagate = False 

# Also log to console for dev
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console)

# Global variables for model and tokenizer
model = None
tokenizer = None
LOADED_MODEL_NAME = "Unknown"
MODEL_PATH = "./models/quillgen-t5-large" # Default path, can be env var
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, tokenizer, LOADED_MODEL_NAME
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    try:
        # Check if local model exists, otherwise load base model for demo purposes if training hasn't run
        path_to_load = MODEL_PATH if os.path.exists(MODEL_PATH) else "t5-base"
        print(f"Actually loading from: {path_to_load}")
        
        tokenizer = AutoTokenizer.from_pretrained(path_to_load)
        model = AutoModelForSeq2SeqLM.from_pretrained(path_to_load).to(DEVICE)
        model.eval()
        LOADED_MODEL_NAME = path_to_load
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Don't crash, just log. API calls will fail if model is None.
    
    yield
    
    # Clean up resources
    print("Shutting down...")

app = FastAPI(title="AskEngine API", lifespan=lifespan)

@app.post("/generate_question", response_model=GeneratedQuestion)
async def generate_question(payload: ContextAnswerPair):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # --- Observability Start ---
    req_id = str(uuid.uuid4())
    start_time = time.time()
    
    # 1. Generation
    # Ensure inputs are secure (basic check done by Pydantic)
    input_text = f"answer: {payload.answer} context: {payload.context}"
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True,
        padding="longest" # Dynamic padding
    ).to(DEVICE)
    
    # Approx token count for metrics
    input_token_count = inputs.input_ids.shape[1]

    # ... Generation Logic ...
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=payload.max_length, 
            num_beams=max(payload.beam_size, payload.num_candidates), # Ensure enough beams for candidates
            early_stopping=True,
            num_return_sequences=payload.num_candidates, # Generate K candidates
            no_repeat_ngram_size=2
        )
    
    # Decode all candidates
    candidates_raw = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    
    # Deduplicate (preserve order)
    unique_candidates = []
    seen = set()
    for c in candidates_raw:
        c = c.strip()
        if c and c not in seen:
            unique_candidates.append(c)
            seen.add(c)
            
    # 2. Quality Filtering & Scoring
    scored_candidates = []
    from inference.quality_utils import evaluate_quality # Lazy import or move to top
    
    for c in unique_candidates:
        report = evaluate_quality(c, payload.answer)
        if report["is_accepted"]:
             scored_candidates.append(c)
        else:
            # Optional: Log rejected candidates for debugging
            pass # Keep logs clean for now

    # Fallback
    if not scored_candidates:
        final_candidates = unique_candidates if unique_candidates else ["Could not generate valid question."]
    else:
        final_candidates = scored_candidates
    
    best_question = final_candidates[0] if final_candidates else "Error generating question"
    
    # --- Observability End ---
    latency_ms = (time.time() - start_time) * 1000
    
    meta_data = {
        "request_id": req_id,
        "latency_ms": round(latency_ms, 2),
        "device": DEVICE,
        "input_tokens": input_token_count,
        "beam_size": payload.beam_size,
        "model": LOADED_MODEL_NAME
    }
    
    # Structured Log
    logger.info(f"Req {req_id} completed in {meta_data['latency_ms']}ms", extra={'props': meta_data})

    return GeneratedQuestion(
        question=best_question, 
        candidates=final_candidates,
        model_name=LOADED_MODEL_NAME,
        meta=meta_data
    )

@app.post("/suggest_answers", response_model=SuggestionResponse)
async def suggest_answers(payload: SuggestionRequest):
    candidates = extract_key_phrases(payload.context, top_k=payload.max_suggestions)
    return SuggestionResponse(candidates=candidates)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": DEVICE, "model_loaded": model is not None}

# Feedback Endpoint
from pydantic import BaseModel
from .data_logging import feedback_logger

class FeedbackRequest(BaseModel):
    context: str
    answer: str
    generated_question: str
    corrected_question: str
    rating: int = None
    model_name: str = "t5-large"

@app.post("/feedback")
async def log_feedback(feedback: FeedbackRequest):
    result = feedback_logger.log_feedback(
        context=feedback.context,
        answer=feedback.answer,
        generated_question=feedback.generated_question,
        corrected_question=feedback.corrected_question,
        rating=feedback.rating,
        model_name=feedback.model_name
    )
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result
