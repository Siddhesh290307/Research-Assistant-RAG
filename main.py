from fastapi import FastAPI
from services.embeddings import embed_query
from services.pinecone_service import query_dense
from services.generator import generate_answer

app = FastAPI()

@app.post("/query")
def query_rag(query: str):

    query_embedding = embed_query(query)
    results = query_dense(query_embedding, top_k=5)

    if not results:
        return {"answer": "No relevant context found."}

    context = "\n\n".join([r["text"] for r in results if r["text"]])

    if not context.strip():
        return {"answer": "Retrieved context is empty."}

    answer = generate_answer(query, context)

    return {"answer": answer}