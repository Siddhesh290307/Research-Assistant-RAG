import os
from services.parser import parse_pdf
from services.chunking import fixed_chunking
from services.embeddings import embed_query
from services.pinecone_service import upsert_vectors, get_index

# -----------------------------
# 1️⃣ Load PDFs from data folder
# -----------------------------
pdf_folder = "./data"
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
print("Found PDFs:", pdf_files)

all_docs = []

for pdf_file in pdf_files:
    elements = parse_pdf(pdf_file)  # returns list of elements
    # convert elements to text dicts
    docs = [
        {
            "text": str(e), 
            "metadata": e["metadata"]
        } 
        for e in elements
    ]
    all_docs.extend(docs)

print(f"Total documents loaded from PDFs: {len(all_docs)}")

# -----------------------------
# 2️⃣ Chunk all documents
# -----------------------------
chunks = fixed_chunking(all_docs, chunk_size=800, overlap=150)
print(f"Total chunks created: {len(chunks)}")

# -----------------------------
# 3️⃣ Upsert all chunks to Pinecone
# -----------------------------
upsert_vectors(chunks, embed_query, batch_size=10)

# -----------------------------
# 4️⃣ Test a query
# -----------------------------
query = "Explain sparse attention in transformers"  # example query
query_embedding = embed_query(query)

index = get_index()
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

print("Query Results:")
for match in results.get("matches", []):
    print("Score:", match["score"])
    print("Text snippet:", match["metadata"].get("text")[:300])  # print first 300 chars
    print("Source:", match["metadata"].get("source"))
    print("-"*50)

# Optional: verify index stats
stats = index.describe_index_stats()
print("Index stats:", stats)