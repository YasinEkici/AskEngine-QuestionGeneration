import os
import json
import threading
from inference.data_logging import FeedbackLogger
import tempfile
import pytest

def test_logger_creates_directory_and_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "test_data", "feedback.jsonl")
        logger = FeedbackLogger(log_file=log_file)
        
        logger.log_feedback("ctx", "ans", "gen", "corr")
        
        assert os.path.exists(log_file)
        
def test_logger_writes_correct_json_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "test.jsonl")
        logger = FeedbackLogger(log_file=log_file)
        
        logger.log_feedback(
            context="Jupiter is big.",
            answer="Jupiter",
            generated_question="What is big?",
            corrected_question="Which planet is big?",
            rating=5,
            model_name="test-model"
        )
        
        with open(log_file, "r") as f:
            line = f.readline()
            data = json.loads(line)
            
            assert data["context"] == "Jupiter is big."
            assert data["corrected_question"] == "Which planet is big?"
            assert data["rating"] == 5
            assert "timestamp" in data

def test_thread_safety():
    """Simulate concurrent writes to ensure no data corruption"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "concurrent.jsonl")
        logger = FeedbackLogger(log_file=log_file)
        
        def write_entry(i):
            logger.log_feedback("ctx", "ans", "gen", f"corr_{i}")

        threads = [threading.Thread(target=write_entry, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        
        # Verify 10 lines written
        with open(log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 10
