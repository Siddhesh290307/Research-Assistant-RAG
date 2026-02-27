from services.embedding_model import model

def embed_text(text: str):
    return model.encode(text).tolist()

def embed_query(text: str):
    return model.encode(text).tolist()