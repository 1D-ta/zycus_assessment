# Zycus AI Support Suite

An intelligent triage and Technical Account Management (TAM) platform powered by Hybrid Retrieval, PII sanitization, and deterministic two-stage prompt chaining.

## Quickstart

1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Environment Setup:** Copy `.env.example` to `.env` and add your API key.
3. **Launch API:** `python -m app.main`
4. **Launch UI:** `streamlit run ui/streamlit_app.py`

## Architecture Highlights
* **Hybrid RAG:** Fuses BM25Okapi and ONNX dense embeddings via Reciprocal Rank Fusion (RRF).
* **Zero-Hallucination TAM Briefs:** Two-stage generation pipeline with code-level quote validation.
* **Streaming Delivery:** Server-Sent Events (SSE) for minimal latency on client UIs.