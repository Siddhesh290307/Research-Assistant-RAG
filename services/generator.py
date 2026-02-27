from ollama import chat
import traceback

def generate_answer(query: str, context: str) -> str:
    MAX_CONTEXT_CHARS = 1500
    safe_context = context[:MAX_CONTEXT_CHARS]

    messages = [
        {"role": "system", "content": "You are a helpful AI research assistant."},
        {
            "role": "user",
            "content": (
                "Answer the question ONLY using the context below. "
                "If the answer is not present, respond with 'I don't know.'\n\n"
                f"Context:\n{safe_context}\n\nQuestion:\n{query}\n\nAnswer:"
            )
        }
    ]

    try:
        # Correct call without unsupported arguments
        response = chat(model="llama2:latest", messages=messages)
        return response.message.content.strip() if response.message else "Generation failed."
    except Exception:
        print("LLM GENERATION ERROR:")
        traceback.print_exc()
        return "Generation failed."