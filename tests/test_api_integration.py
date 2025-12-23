from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from inference.model_server import app
import pytest

client = TestClient(app)

# Mock the global variables in model_server
@pytest.fixture(autouse=True)
def mock_model_components():
    with patch("inference.model_server.model") as mock_model, \
         patch("inference.model_server.tokenizer") as mock_tokenizer, \
         patch("inference.model_server.LOADED_MODEL_NAME", "mock-t5-large"):
        
        # Setup mock behavior
        mock_tokenizer.return_value.to.return_value = {"input_ids": "fake_tensor"}
        mock_tokenizer.decode.return_value = "What is the capital of France?"
        
        # Mock generate output
        mock_model.generate.return_value = ["fake_output_tensor"]
        
        yield mock_model, mock_tokenizer

def test_health_check_endpoint():
    # We patch the health check global check effectively by mocking the globals above
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_generate_question_success():
    payload = {
        "context": "Paris is the capital of France.",
        "answer": "Paris",
        "beam_size": 2,
        "max_length": 32
    }
    
    response = client.post("/generate_question", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert data["question"] == "What is the capital of France?"
    assert "candidates" in data
    assert len(data["candidates"]) >= 1
    assert data["candidates"][0] == "What is the capital of France?"
    assert data["model_name"] == "mock-t5-large"

def test_generate_question_validation_error():
    # Missing 'answer'
    payload = {
        "context": "Paris is the capital of France."
    }
    response = client.post("/generate_question", json=payload)
    assert response.status_code == 422

@patch("inference.model_server.model", None)
def test_generate_question_service_unavailable():
    # Test when model is not initialized
    payload = {
        "context": "Test",
        "answer": "Test"
    }
    response = client.post("/generate_question", json=payload)
    assert response.status_code == 503
