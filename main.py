# main.py
from fastapi import FastAPI
from services.embeddings import embed_query
from services.pinecone_service import query_dense
from services.generator import generate_answer
import traceback

app = FastAPI(title="RAG Research Assistant")

MAX_CONTEXT_CHARS = 4000  # hard cap to prevent overflow

@app.post("/query")
def query_rag(query: str, mode: str = "sliding"):

    try:
        # 1️⃣ Embed the query
        query_embedding = embed_query(query)

        # 2️⃣ Retrieve top-k relevant chunks from Pinecone
        index_name = "rag-semantic" if mode == "semantic" else "rag-sliding"
        results = query_dense(
            index_name=index_name,
            query_embedding=query_embedding,
            top_k=5
        )

        if not results:
            return {
                "answer": "No relevant context found.",
                "context": "",
                "retrieved_chunks": []
            }

        # 3️⃣ Combine retrieved text
        retrieved_chunks = [r["text"] for r in results if r.get("text")]
        context = "\n\n".join(retrieved_chunks)

        # HARD CONTEXT LIMIT
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]

        # 4️⃣ Generate answer using Ollama
        try:
            answer = generate_answer(query, context)
        except Exception:
            print("LLM GENERATION ERROR:")
            traceback.print_exc()
            answer = "Generation failed."

        return {
            "answer": answer,
            "context": context,
            "retrieved_chunks": retrieved_chunks
        }

    except Exception:
        print("QUERY PIPELINE ERROR:")
        traceback.print_exc()
        return {
            "answer": "Internal error.",
            "context": "",
            "retrieved_chunks": []
        }