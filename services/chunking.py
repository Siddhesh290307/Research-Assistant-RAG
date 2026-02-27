import re
from sklearn.metrics.pairwise import cosine_similarity
from services.embedding_model import model


#fixed size sliding window chunking
def fixed_chunking(docs, chunk_size=800, overlap=150):
    chunks = []

    for doc in docs:
        text = doc["text"]
        metadata = doc["metadata"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })

            start += chunk_size - overlap

    return chunks

#sematic chunking
def semantic_chunking(docs, similarity_threshold=0.75, max_chunk_chars=1200):
    chunks = []

    for doc in docs:
        text = doc["text"]
        metadata = doc["metadata"]

        # Sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        embeddings = model.encode(sentences)

        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity(
                [embeddings[i - 1]],
                [embeddings[i]]
            )[0][0]

            if sim < similarity_threshold or \
               len(" ".join(current_chunk)) > max_chunk_chars:

                chunks.append({
                    "text": " ".join(current_chunk),
                    "metadata": metadata
                })

                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "metadata": metadata
            })

    return chunks