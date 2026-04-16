# Qdrant Edge RAG (Legal Assistant)

A local Retrieval-Augmented Generation (RAG) app using:
- `qdrant_edge` for local vector storage
- `fastembed` for embeddings
- `FastAPI` for API endpoints
- OpenRouter-compatible chat models (default: `google/gemma-4`)

## 1) Prerequisites

- Python 3.10+ installed
- Windows PowerShell or Command Prompt

## 2) Setup

From project root:

```powershell
cd "C:\Users\hp\Desktop\superteams.ai\qdrant_Edge_rag"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Configure API key (for LLM answers)

Set your OpenRouter key in the same terminal where you run Uvicorn:

```cmd
set OPENROUTER_API_KEY=your_openrouter_key_here
```

Optional: choose model explicitly:

```cmd
set OPENROUTER_MODEL=meta-llama/llama-3-8b-instruct
```

If no valid key is available, the app returns an extractive fallback answer from retrieved text.

## 4) Ingest data into Qdrant Edge

Make sure your content is in `data.txt`, then run:

```cmd
venv\Scripts\python ingest.py
```

Expected output:

```text
Inserted <N> chunks into Qdrant Edge
```

## 5) Start API server

```cmd
uvicorn main:app --reload
```

Server URL:
- `http://127.0.0.1:8000`

Browser UI:
- `http://127.0.0.1:8000/`

## 6) Test endpoints

Health:

```text
GET http://127.0.0.1:8000/health
```

Ready:

```text
GET http://127.0.0.1:8000/ready
```

Ask (example):

```text
GET http://127.0.0.1:8000/ask?q=what%20happens%20if%20payment%20is%20delayed
```

Swagger UI:
- `http://127.0.0.1:8000/docs`

## 7) Typical run order

1. `venv\Scripts\activate`
2. `venv\Scripts\python ingest.py`
3. `set OPENROUTER_API_KEY=...`
4. `uvicorn main:app --reload`
5. Call `/ask`

## 8) Troubleshooting

- **Fallback response appears (`Based on the stored legal context...`)**
  - Usually means LLM call failed (missing key / model issue). Re-check `OPENROUTER_API_KEY`.
- **`Shard not found. Run ingest.py first.`**
  - Run `venv\Scripts\python ingest.py`.
- **Changed `data.txt` but old answers still appear**
  - Re-run ingestion. Current `ingest.py` rebuilds the shard cleanly each run.

