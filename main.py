import logging
import os
import re
from typing import Optional

from fastapi import FastAPI, Query as FastAPIQuery
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from qdrant_edge import EdgeShard, Query, QueryRequest, ScrollRequest
from fastembed import TextEmbedding
from openai import OpenAI

# ---------------- CONFIG ----------------
MODEL_NAME = "BAAI/bge-small-en"
VECTOR_NAME = "text"
COLLECTION = "legal_docs"

TOP_K = 10
OPENROUTER_MODEL = "meta-llama/llama-3-8b-instruct"

BASE_PATH = "./qdrant_data"
SHARD_PATH = os.path.join(BASE_PATH, COLLECTION)

logging.basicConfig(level="INFO")

# ---------------- EMBEDDING ----------------
text_model = TextEmbedding(model_name=MODEL_NAME, cache_dir="./models")

def embed(text: str):
    return list(text_model.embed([text]))[0].tolist()

# ---------------- QDRANT ----------------
edge_shard: Optional[EdgeShard] = None

def get_edge_shard():
    global edge_shard
    if edge_shard:
        return edge_shard

    if not os.path.exists(SHARD_PATH):
        raise RuntimeError("Run ingest.py first.")

    edge_shard = EdgeShard.load(SHARD_PATH)
    return edge_shard

# ---------------- LLM ----------------
llm_client: Optional[OpenAI] = None

def get_llm_client():
    global llm_client
    if llm_client:
        return llm_client

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key")

    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return llm_client

def generate_answer(messages):
    client = get_llm_client()
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages
    )
    return response.choices[0].message.content

# ---------------- PATTERNS ----------------
GREETING_PATTERN = re.compile(r"^(hi|hello|hey)\b", re.I)
SUMMARY_PATTERN = re.compile(r"\b(summary|summarize|overview)\b", re.I)

# ---------------- SEARCH ----------------
def search(query: str):
    shard = get_edge_shard()
    vector = embed(query)

    results = shard.query(
        QueryRequest(
            query=Query.Nearest(vector, using=VECTOR_NAME),
            limit=TOP_K,
            with_payload=True
        )
    )

    return [r.payload["content"] for r in results if r.payload]


def retrieve_all_docs() -> list[str]:
    shard = get_edge_shard()
    docs: list[str] = []
    offset = 0
    page_size = 50

    while True:
        results = shard.scroll(
            ScrollRequest(
                offset=offset,
                limit=page_size,
                with_payload=True,
            )
        )

        if not results:
            break

        batch = results[0] if isinstance(results, tuple) and results else results
        if not batch:
            break

        docs.extend([r.payload["content"] for r in batch if r.payload])
        offset += len(batch)

        if len(batch) < page_size:
            break

    return docs

# ---------------- PROMPT ----------------
PROMPT_TEMPLATE = """You are a legal assistant.

Use ONLY the provided context.

If the user greets you or says hello:
- Reply politely as a legal assistant.
- Tell them to ask a legal question related to the provided context.

If asked to summarize:
- Provide a direct summary of the retrieved context only.
- Do NOT include an introductory phrase such as "Here is a summary..." or "Below is a summary...".
- Do NOT name the document or state its title unless the question explicitly asks for it.
- Cover ALL major sections of the document.
- Each sentence should represent a DIFFERENT section.
- Do NOT skip important sections.
- Use exact details (e.g., "Indian law", "Bengaluru").
- Keep it clear, structured, and concise.

If asked for 5–6 lines:
- Write EXACTLY 5–6 sentences.
- Each sentence = one section.

If the question is unrelated to the legal context, say exactly: "I can only answer legal questions based on the provided context. Please ask a question about the document."

Context:
{context}

Question: {query}

Answer:
"""

def build_prompt(query, context):
    return [
        {"role": "system", "content": PROMPT_TEMPLATE.format(context=context, query=query)},
        {"role": "user", "content": query}
    ]

SUMMARY_CLEANUP_REGEX = re.compile(
    r"^(?:\s*(?:here is|below is|following is|summary of|a summary of|the summary of)\s*(?:the\s*)?[^:]*:?\s*)",
    re.I
)

SECTION_HEADER_PATTERN = re.compile(r"^\s*(\d+)\.\s+(.+)$", re.M)


def extract_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    matches = list(SECTION_HEADER_PATTERN.finditer(text))
    if not matches:
        return sections

    for idx, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip().replace("\n", " ")
        sections.append((title, body))
    return sections


def summarize_sections(docs: list[str]) -> str | None:
    all_sections: list[tuple[str, str]] = []
    for doc in docs:
        all_sections.extend(extract_sections(doc))

    if not all_sections:
        return None

    sentence_summary: list[str] = []
    for title, body in all_sections:
        parts = re.split(r"(?<!\d)\.(?!\d)|[!?]", body)
        first_sentence = next((p.strip() for p in parts if p.strip()), "").strip()
        if not first_sentence:
            continue
        sentence = f"{title}: {first_sentence}."
        sentence_summary.append(sentence)

    return " ".join(sentence_summary) if sentence_summary else None


def clean_summary_answer(answer: str) -> str:
    cleaned = SUMMARY_CLEANUP_REGEX.sub("", answer.strip())
    return cleaned or answer.strip()


# ---------------- RAG ----------------
def rag(query: str):
    query = query.strip()

    try:
        if SUMMARY_PATTERN.search(query):
            docs = retrieve_all_docs()
        else:
            docs = search(query)
    except Exception as e:
        return f"Error processing query: {e}"

    if not docs:
        return "No relevant legal information found."

    context = "\n\n--- SECTION ---\n\n".join(docs)

    if SUMMARY_PATTERN.search(query):
        section_summary = summarize_sections(docs)
        if section_summary:
            return section_summary

    try:
        answer = generate_answer(build_prompt(query, context))
        if SUMMARY_PATTERN.search(query):
            return clean_summary_answer(answer)
        return answer
    except Exception as e:
        logging.warning("LLM generation failed: %s", e)
        return f"Error: {str(e)}"

# ---------------- FASTAPI ----------------
app = FastAPI()

class AskResponse(BaseModel):
    answer: str

@app.get("/ask", response_model=AskResponse)
def ask(q: str = FastAPIQuery(...)):
    return AskResponse(answer=rag(q))


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
    <div class="container">
        <h1>Legal RAG Assistant</h1>
        <p>Enter a legal question and get a response from the local RAG system.</p>
        <textarea id="queryInput" placeholder="Ask a legal question..."></textarea>
        <div style="margin-top: 14px; display: flex; gap: 10px;">
            <button id="askButton">Ask</button>
            <span class="status" id="statusText">Ready to ask.</span>
        </div>
        <div class="result" id="answerBox">Your answer will appear here.</div>
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
                    throw new Error("Request failed");
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
</html>
"""