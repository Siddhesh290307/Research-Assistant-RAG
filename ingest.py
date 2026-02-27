import os
from services.parser import parse_pdf
from services.chunking import fixed_chunking, semantic_chunking
from services.embeddings import embed_text
from services.pinecone_service import create_index_if_not_exists, upsert_vectors

PDF_FOLDER = "data"

SLIDING_INDEX = "rag-sliding"
SEMANTIC_INDEX = "rag-semantic"

# Create both indexes
create_index_if_not_exists(SLIDING_INDEX)
create_index_if_not_exists(SEMANTIC_INDEX)

all_docs = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, file)
        docs = parse_pdf(path)
        all_docs.extend(docs)

# -----------------------------
# Sliding Chunking
# -----------------------------
sliding_chunks = fixed_chunking(all_docs)
print(f"Sliding chunks: {len(sliding_chunks)}")

upsert_vectors(SLIDING_INDEX, sliding_chunks, embed_text)
print("Sliding ingestion complete.")


# -----------------------------
# Semantic Chunking
# -----------------------------
semantic_chunks = semantic_chunking(all_docs)
print(f"Semantic chunks: {len(semantic_chunks)}")

upsert_vectors(SEMANTIC_INDEX, semantic_chunks, embed_text)
print("Semantic ingestion complete.")