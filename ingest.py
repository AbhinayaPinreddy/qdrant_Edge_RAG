import os
import shutil
from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    Point,
    UpdateOperation,
)
import hashlib
from fastembed import TextEmbedding

# -------- CONFIG --------
MODEL_NAME = "BAAI/bge-small-en"
VECTOR_NAME = "text"
COLLECTION = "legal_docs"
DATA_FILE = "data.txt"

# -------- EMBEDDING --------
text_model = TextEmbedding(
    model_name=MODEL_NAME,
    cache_dir="./models"
)

def embed(text: str):
    return list(text_model.embed([text]))[0].tolist()

# -------- QDRANT EDGE --------
BASE_PATH = "./qdrant_data"
SHARD_PATH = os.path.join(BASE_PATH, COLLECTION)

def create_fresh_shard() -> EdgeShard:
    os.makedirs(BASE_PATH, exist_ok=True)
    if os.path.exists(SHARD_PATH):
        shutil.rmtree(SHARD_PATH)
    os.makedirs(SHARD_PATH, exist_ok=True)
    return EdgeShard.create(
        SHARD_PATH,
        EdgeConfig(
            vectors={
                VECTOR_NAME: EdgeVectorParams(size=384, distance=Distance.Cosine)
            }
        ),
    )

# -------- CHUNKING --------
def chunk_text(text: str, chunk_size: int = 90, overlap: int = 20):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks

# -------- LOAD FILE --------
def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return f.read()

# -------- INSERT --------
def insert_data():
    edge_shard = create_fresh_shard()
    text = load_data()
    chunks = chunk_text(text)

    points = []

    for index, chunk in enumerate(chunks):
        stable_id = hashlib.md5(f"{COLLECTION}:{index}:{chunk}".encode("utf-8")).hexdigest()
        point = Point(
            id=stable_id,
            vector={VECTOR_NAME: embed(chunk)},
            payload={"content": chunk, "chunk_index": index, "source": DATA_FILE}
        )
        points.append(point)

    edge_shard.update(
        UpdateOperation.upsert_points(points)
    )

    print(f"Inserted {len(points)} chunks into Qdrant Edge")


# -------- RUN --------
if __name__ == "__main__":
    insert_data()