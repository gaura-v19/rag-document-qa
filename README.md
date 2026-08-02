# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system built from scratch in Python — no LangChain or LlamaIndex — to understand the raw mechanics of document chunking, embeddings, similarity search, and LLM-based answer generation.

## Why build this from scratch

Most RAG tutorials wrap everything in LangChain/LlamaIndex, hiding how retrieval actually works. This project implements each core mechanism manually to understand the real problem each component solves — chunking strategy, embedding generation, cosine similarity search, and prompt construction — rather than treating them as black-box library calls.

## Features

- **Local embeddings** — uses `sentence-transformers` (no API cost, no external calls)
- **Local LLM** — runs entirely offline via Ollama (`gemma4:e4b`)
- **Disk-cached embeddings** — documents aren't re-embedded on every run
- **Streamed responses** — answers stream token-by-token instead of waiting for full generation
- **Source citations** — every answer cites the file name + chunk index it came from
- **Agentic retry** — if retrieval confidence is low, the query is reformulated and retried once before generating an answer

## Architecture
loader.py → reads .txt/.pdf files from a folder
chunker.py → splits text into overlapping word-based chunks
embedder.py → generates local embeddings for each chunk
vector_store.py → stores embeddings, supports disk caching, cosine similarity search
retriever.py → finds top-5 most relevant chunks for a query
agent_controller.py → checks retrieval confidence, retries with reformulated query if low
generator.py → builds prompt, streams LLM response, adds citations
main.py → CLI entry point, ties everything together

## How it works

1. Documents are loaded and split into ~300-word overlapping chunks
2. Each chunk is converted into a vector embedding (local model, no API)
3. Embeddings are cached to disk — subsequent runs skip re-embedding
4. On a user query, the query is embedded the same way, and the top-5 most similar chunks are retrieved via cosine similarity
5. If the best match's similarity score is too low, the query is reformulated and retrieval is retried once
6. Retrieved chunks + the query are sent to a local LLM (Ollama), which streams back an answer with source citations

## Tech stack

- Python
- `sentence-transformers` — local embedding model
- `pypdf` — PDF text extraction
- `numpy` — vector operations, cosine similarity
- Ollama (`gemma4:e4b`) — local LLM for answer generation

## Setup

```bash
pip install -r requirements.txt
ollama pull gemma4:e4b
```

## Usage

```bash
python main.py
```

Place `.txt`/`.pdf` documents in a folder (default: `test_docs/`), then ask questions in the CLI loop.

## Status

Work in progress .
