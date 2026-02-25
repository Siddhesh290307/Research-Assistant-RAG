import os
from services.parser import parse_pdf
from services.chunking import fixed_chunking
from services.embeddings import embed_text
from services.pinecone_service import create_index_if_not_exists, upsert_vectors

PDF_FOLDER = "data"

create_index_if_not_exists()

all_docs = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, file)
        docs = parse_pdf(path)
        all_docs.extend(docs)

chunks = fixed_chunking(all_docs)

print(f"Total chunks: {len(chunks)}")

upsert_vectors(chunks, embed_text)

print("Ingestion complete.")