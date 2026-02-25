import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found.")

client = genai.Client(api_key=GEMINI_API_KEY)

def generate_answer(query: str, context: str) -> str:
    prompt = f"""
You are an AI research assistant.

Use ONLY the provided context to answer.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
    )

    return response.text