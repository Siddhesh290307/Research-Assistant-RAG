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