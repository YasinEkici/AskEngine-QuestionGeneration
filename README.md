# 📝 AskEngine: Automatic Question Generation System

AskEngine is a production-grade Machine Learning project that generates intelligent questions from a given text context and answer. It leverages state-of-the-art Transformer models (T5/BART) fine-tuned on the Stanford Question Answering Dataset (SQuAD).

![Python](https://img.shields.io/badge/Python-3.13.3-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Hugging Face](https://img.shields.io/badge/🤗-Transformers-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Features

- **Advanced NLP**: T5-Large model fine-tuned for high-quality question generation.
- **Smart Chunking**: Automatically processes long documents (PDF/TXT) by finding relevant context windows.
- **High Performance**: Optimized with Dynamic Padding and BF16 for fast inference on modern GPUs.
- **Production-Ready API**: Robust FastAPI backend with validation and error handling.
- **Interactive Interface**: Modern Streamlit web app with session history and file upload.
- **Answer Assistant**: "Magic Button" to auto-suggest potential answers (dates, names, key phrases) from the context.
- **Refined UX**: Clean, modern 2-column layout with progressive disclosure of advanced settings.
- **Observability**: Structured JSON logging and metrics (Latency, Tokens, Request ID) for monitoring.
- **Active Learning**: Built-in feedback loop to collect user corrections for future training.
- **Multi-Candidate Generation**: Generates multiple question variations and intelligently reranks them.
- **Quality Guardrails**: Automatic filtering of low-quality, repetitive, or leaked outputs.
- **Smart Chunking**: Token-aware context windowing with visual **Span Highlighting** for answer verification.
- **Enhanced PDF Support**: Layout-aware text extraction that repairs broken words and line breaks.
- **Reliable Testing**: Comprehensive Unit, Integration, and Model Sanity tests.
- **CI/CD Integrated**: GitHub Actions workflow for automated testing.
- **Code Quality**: Modular architecture with type hinting and reproducible pipelines.

## 🛠️ Tech Stack

- **Language**: Python 3.13.3
- **ML Framework**: PyTorch, Hugging Face Transformers
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit (with Custom CSS)
- **Utilities**: PyPDF (PDF Parsing)
- **Testing**: Pytest (Mocking + Integration), Benchmark Scripts
- **DevOps**: GitHub Actions

## 📂 Project Structure

```
/QuillGen
|-- /data_processing   # Data preparation scripts
|-- /training          # Model training and evaluation
|-- /inference         # API server and schemas
|-- /app               # Streamlit demo application
|-- /tests             # Unit tests
|-- .github/workflows  # CI/CD configuration
```

## ⚡ Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-enabled GPU (recommended for training)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/AskEngine.git
   cd AskEngine
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### 🏃‍♂️ Running the Demo

1. **Start the API Server:**

   ```bash
   uvicorn inference.model_server:app --reload
   ```

   The API will be available at `http://127.0.0.1:8000`.
2. **Launch the Streamlit App:**

   ```bash
   streamlit run app/demo.py
   ```

   Open your browser at `http://localhost:8501`.

## 🧠 Model Training

To train the model from scratch:

1. **Prepare the Data:**

   ```bash
   python data_processing/prepare_squad.py
   ```
2. **Train the Model:**

   ```bash
   python training/train.py --epochs 3 --batch_size 8
   ```

   The best model will be saved to `models/quillgen-t5-large`.
3. **Evaluate:**

   ```bash
   python training/evaluate.py
   ```

## 🧪 Testing

The project includes a professional test suite covering three layers:

1. **Unit Tests**: Check data processing logic (`tests/test_data_processing.py`).
2. **API Integration Tests**: Verify API endpoints and logic using Mocks (`tests/test_api_integration.py`).
3. **Model Sanity Tests**: Validate GPU access and real model generation (`tests/test_model_sanity.py`).

Run the full suite:

```bash
pytest -v
```

## 🔮 Roadmap

We have ambitious plans to scale AskEngine into a fully cloud-native solution:

- [ ] **Dockerization**: Containerize the API and Frontend for reproducible deployments.
- [ ] **Cloud Deployment**: Deploy on AWS/GCP using Kubernetes or Serverless container services.
- [ ] **Model Distillation**: Compress T5-Large into a smaller, faster student model for mobile-friendly inference.
- [ ] **Multi-Language Support**: Extend training data to support question generation in other languages (e.g., Turkish, Spanish).

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

MIT License
