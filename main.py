import logging
import os
import re
from typing import Optional

from fastapi import FastAPI, HTTPException, Query as FastAPIQuery
from pydantic import BaseModel
from qdrant_edge import EdgeShard, Query, QueryRequest
from fastembed import TextEmbedding
from openai import OpenAI
from openai import OpenAIError

# ---------------- CONFIG ----------------
MODEL_NAME = "BAAI/bge-small-en"
VECTOR_NAME = "text"
COLLECTION = "legal_docs"
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "500"))
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4")

BASE_PATH = "./qdrant_data"
SHARD_PATH = os.path.join(BASE_PATH, COLLECTION)

logger = logging.getLogger("rag_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------- EMBEDDING ----------------
text_model = TextEmbedding(
    model_name=MODEL_NAME,
    cache_dir="./models"
)

def embed(text: str):
    return list(text_model.embed([text]))[0].tolist()

# ---------------- QDRANT EDGE ----------------
edge_shard: Optional[EdgeShard] = None

def get_edge_shard() -> EdgeShard:
    global edge_shard

    if edge_shard is not None:
        return edge_shard

    if not os.path.exists(SHARD_PATH):
        raise RuntimeError(" Qdrant shard not found. Run ingest.py first.")

    edge_shard = EdgeShard.load(SHARD_PATH)
    return edge_shard

# ---------------- GEMMA (OpenRouter) ----------------
llm_client: Optional[OpenAI] = None

def get_llm_client() -> OpenAI:
    global llm_client

    if llm_client is not None:
        return llm_client

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) and restart server."
        )

    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return llm_client

def generate_answer(prompt: str):
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except OpenAIError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

def build_fallback_answer(query: str, docs: list[str]) -> str:
    """Return an extractive answer when LLM is unavailable."""
    query_terms = {word.lower() for word in query.split() if len(word) > 2}
    sentences: list[str] = []

    for doc in docs:
        normalized_doc = doc.replace("\n", " ").strip()
        # Split sentence endings but preserve decimal numbers like 1.5%.
        parts = re.split(r"(?<!\d)\.(?!\d)|[!?]", normalized_doc)
        sentences.extend([f"{part.strip()}." for part in parts if part.strip()])

    scored_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for term in query_terms if term in sentence_lower)
        scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    top_sentences = []
    for score, text in scored_sentences:
        if score <= 0:
            continue
        if text in top_sentences:
            continue
        top_sentences.append(text)
        if len(top_sentences) == 3:
            break

    if top_sentences:
        return "Based on the stored legal context: " + " ".join(top_sentences)

    return "Relevant context was found, but no direct extractive match is available."

# ---------------- SEARCH ----------------
def search(query: str):
    shard = get_edge_shard()

    query_vector = embed(query)

    results = shard.query(
        QueryRequest(
            query=Query.Nearest(query_vector, using=VECTOR_NAME),
            limit=TOP_K,
            with_payload=True
        )
    )

    return [r.payload.get("content", "") for r in results if r.payload]

# ---------------- RAG PIPELINE ----------------
def rag(query: str):
    docs = search(query)

    if not docs:
        return "No relevant legal information found."

    context = "\n".join(docs)

    prompt = f"""You are a legal assistant. Answer ONLY using the context below.

Context:
{context}

Question: {query}"""

    try:
        return generate_answer(prompt)
    except Exception as exc:
        logger.warning("Falling back to extractive answer: %s", exc)
        return build_fallback_answer(query, docs)

# ---------------- FASTAPI ----------------
app = FastAPI()


class AskResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/ready", response_model=HealthResponse)
def ready():
    if not os.path.exists(SHARD_PATH):
        raise HTTPException(status_code=503, detail="Shard not found. Run ingest.py first.")
    return HealthResponse(status="ready")

@app.get("/ask", response_model=AskResponse)
def ask(q: str = FastAPIQuery(..., min_length=3, max_length=MAX_QUERY_CHARS)):
    try:
        answer = rag(q)
        return AskResponse(answer=answer)
    except Exception as e:
        logger.exception("Ask request failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}") from e