import json
import os
import time
from typing import Dict, Any
from pathlib import Path
import threading

class FeedbackLogger:
    """
    Handles thread-safe logging of user feedback to a JSONL file.
    Designed for Human-in-the-Loop data collection.
    """
    def __init__(self, log_file: str = "data/feedback.jsonl"):
        self.log_file = log_file
        self.lock = threading.Lock()
        self._ensure_directory()

    def _ensure_directory(self):
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def log_feedback(self, 
                     context: str, 
                     answer: str, 
                     generated_question: str, 
                     corrected_question: str, 
                     rating: int = None,
                     model_name: str = "unknown") -> Dict[str, Any]:
        
        entry = {
            "timestamp": time.time(),
            "model_name": model_name,
            "context": context,
            "answer": answer,
            "generated_question": generated_question,
            "corrected_question": corrected_question,
            "rating": rating
        }

        with self.lock:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                return {"status": "success", "message": "Feedback logged successfully"}
            except Exception as e:
                print(f"Error logging feedback: {e}")
                return {"status": "error", "message": str(e)}

# Singleton instance for easy import
feedback_logger = FeedbackLogger()
