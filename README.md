# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system built from scratch in Python — without LangChain or LlamaIndex — implementing document chunking, embeddings, vector similarity search, and LLM-based answer generation manually, along with a Streamlit web interface.

## Why built from scratch

Most RAG tutorials rely on frameworks that abstract away the core mechanics. This project implements each component manually — chunking strategy, embedding generation, cosine similarity search, and prompt construction — to build a genuine understanding of how retrieval-augmented generation actually works.

## Features

- Custom document chunking with overlapping windows to preserve context across boundaries
- Local embedding generation (Sentence-Transformers) — no API cost for embeddings
- Disk-cached embeddings — avoids re-computation on repeated runs
- Cosine-similarity-based vector search implemented from scratch
- Agentic retry — reformulates and re-searches when retrieval confidence is low
- Cloud LLM integration (Gemini API) for answer generation, with streaming responses
- Source citations (filename + chunk index) on every answer
- Confidence-threshold gating — avoids unnecessary LLM calls when no relevant content is found
- Streamlit web UI with document upload and chat interface

## Architecture

loader.py            -> reads .txt/.pdf files from a folder
chunker.py           -> splits text into overlapping word-based chunks
embedder.py          -> generates local embeddings for each chunk
vector_store.py      -> stores embeddings, disk caching, cosine similarity search
retriever.py         -> finds top-5 most relevant chunks for a query
agent_controller.py  -> checks retrieval confidence, retries with reformulated query
generator.py         -> builds prompt, streams LLM response, adds citations
main.py              -> CLI entry point
app.py               -> Streamlit web interface

## How it works

Documents -> Chunking -> Embeddings -> Vector Search -> Context -> LLM -> Answer

1. Documents are loaded and split into ~300-word overlapping chunks
2. Each chunk is embedded locally into a 384-dimensional vector
3. Embeddings are cached to disk to avoid redundant computation
4. On a query, the same embedding process runs, and the top-5 most similar chunks are retrieved via cosine similarity
5. If retrieval confidence is low, the query is reformulated and retried once
6. If still low confidence, the system reports no relevant information found — avoiding hallucinated answers
7. Otherwise, retrieved chunks + the query are sent to Gemini, which streams back an answer with source citations

## Tech stack

- Python
- Sentence-Transformers — local embeddings
- NumPy — vector operations, cosine similarity
- PyPDF — PDF text extraction
- Google Gemini API — LLM-based answer generation
- Streamlit — web interface

## Setup

pip install -r requirements.txt

Set your Gemini API key as an environment variable:

setx GEMINI_API_KEY "your-key-here"

## Usage

CLI:
python main.py

Web UI:
streamlit run app.py

## Known limitations

- Scanned/image-only PDFs are not supported (no OCR) — only extractable text content is processed
- Query reformulation is currently rule-based; an LLM-based reformulation would improve retry quality

## Roadmap

- Tool/function calling — allow the LLM to use external tools (e.g., web search) when document context is insufficient

## Challenges & Learnings

- Initially attempted local LLM inference via Ollama (gemma4:e4b) for a fully offline pipeline. Hit VRAM allocation failures on 6GB GPU hardware under real-world memory contention (background apps competing for GPU memory). Diagnosed the issue using nvidia-smi and Task Manager, isolated it to environment-specific memory pressure rather than a code defect.
- Pivoted to Google Gemini API for generation to ensure reliability and reproducibility across environments, while keeping embeddings local (no cost, no external dependency for the retrieval half of the pipeline).
- This tradeoff reflects a common real-world engineering decision: balancing full local control against operational reliability.
