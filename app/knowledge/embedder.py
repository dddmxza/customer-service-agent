from sentence_transformers import SentenceTransformer
from typing import List
from app.config import config


_model = None


def get_embed_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_path = str(config.BASE_DIR / "models" / "bge-small-zh-v1.5")
        print(f"Loading embedding model from: {model_path}")
        _model = SentenceTransformer(model_path)
        print("Model loaded")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embed_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


if __name__ == "__main__":
    from app.knowledge.loader import load_documents
    from app.knowledge.splitter import split_documents

    docs = load_documents()
    chunks = split_documents(docs)
    texts = [c["content"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    vectors = embed_texts(texts)

    print(f"Done. {len(vectors)} vectors.")
    print(f"Vector dimension: {len(vectors[0])}")
    print(f"First 5 numbers: {vectors[0][:5]}")