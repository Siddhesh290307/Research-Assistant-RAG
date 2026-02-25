import os
import uuid
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-peft"
EMBEDDING_DIMENSION = 384

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found.")

pc = Pinecone(api_key=PINECONE_API_KEY)


# -----------------------------
# Create index if not exists
# -----------------------------
def create_index_if_not_exists():
    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )


def get_index():
    return pc.Index(INDEX_NAME)


# -----------------------------
# Upsert vectors
# -----------------------------
def upsert_vectors(chunks, embed_fn, batch_size=100):
    index = get_index()
    vectors_batch = []

    for chunk in chunks:
        embedding = embed_fn(chunk["text"])

        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        vectors_batch.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                **chunk["metadata"],
                "text": chunk["text"]
            }
        })

        if len(vectors_batch) >= batch_size:
            index.upsert(vectors=vectors_batch)
            vectors_batch = []

    if vectors_batch:
        index.upsert(vectors=vectors_batch)


# -----------------------------
# Dense query
# -----------------------------
def query_dense(query_embedding, top_k=10):
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    index = get_index()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    cleaned = []

    for match in results.get("matches", []):
        cleaned.append({
            "text": match["metadata"].get("text", ""),
            "metadata": {
                k: v for k, v in match["metadata"].items() if k != "text"
            },
            "score": match["score"]
        })

    return cleaned