import logging
import os
import re
from typing import Optional

from fastapi import FastAPI, HTTPException, Query as FastAPIQuery
from fastapi.responses import HTMLResponse
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
PROMPT_TEMPLATE = """You are a legal assistant responding to a question with information drawn only from the provided context.

Instructions:
- Answer using ONLY the context below.
- Do not invent facts or cite information not present in the context.
- If the context does not contain a clear answer, say: "No relevant legal information found in the provided context."
- Keep the response concise, professional, and directly relevant to the question.
- Avoid extraneous commentary, opinions, or speculation.

Context:
{context}

Question: {query}

Answer:"""


def rag(query: str):
    docs = search(query)

    if not docs:
        return "No relevant legal information found."

    context = "\n".join(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

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


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Legal RAG Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f7fb; margin: 0; padding: 0; }
        .container { max-width: 760px; margin: 40px auto; background: #ffffff; padding: 24px; border-radius: 12px; box-shadow: 0 16px 32px rgba(0,0,0,0.08); }
        h1 { margin-top: 0; }
        textarea { width: 100%; min-height: 120px; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; font-size: 15px; resize: vertical; }
        button { background: #2563eb; color: white; border: none; border-radius: 8px; padding: 12px 22px; font-size: 15px; cursor: pointer; }
        button:disabled { background: #9ca3af; cursor: default; }
        .result { margin-top: 20px; padding: 18px; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px; white-space: pre-wrap; }
        .status { margin-top: 10px; font-size: 14px; color: #4b5563; }
    </style>
</head>
<body>
    <div class=\"container\">
        <h1>Legal RAG Assistant</h1>
        <p>Enter a legal question and get a response from the local RAG system.</p>
        <textarea id=\"queryInput\" placeholder=\"Ask a legal question...\"></textarea>
        <div style=\"margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap;\">
            <button id=\"askButton\">Ask</button>
            <span class=\"status\" id=\"statusText\">Ready to ask.</span>
        </div>
        <div class=\"result\" id=\"answerBox\">Your answer will appear here.</div>
    </div>
    <script>
        const askButton = document.getElementById('askButton');
        const queryInput = document.getElementById('queryInput');
        const statusText = document.getElementById('statusText');
        const answerBox = document.getElementById('answerBox');

        askButton.addEventListener('click', async () => {
            const query = queryInput.value.trim();
            if (!query) {
                statusText.textContent = 'Please enter a question first.';
                return;
            }

            askButton.disabled = true;
            statusText.textContent = 'Requesting answer...';
            answerBox.textContent = '';

            try {
                const response = await fetch(`/ask?q=${encodeURIComponent(query)}`);
                if (!response.ok) {
                    const error = await response.text();
                    throw new Error(error || 'Request failed');
                }
                const data = await response.json();
                answerBox.textContent = data.answer;
                statusText.textContent = 'Answer received.';
            } catch (err) {
                answerBox.textContent = `Error: ${err.message}`;
                statusText.textContent = 'Unable to fetch answer.';
            } finally {
                askButton.disabled = false;
            }
        });
    </script>
</body>
</html>"""