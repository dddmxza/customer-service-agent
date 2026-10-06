import chromadb
from typing import List, Dict
from app.config import config
from app.knowledge.embedder import embed_texts


_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name="customer_service",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def add_chunks(chunks: List[Dict]) -> None:
    collection = get_collection()
    texts = [c["content"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = embed_texts(texts)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=vectors,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    print(f"Added {len(chunks)} chunks to Chroma.")


if __name__ == "__main__":
    from app.knowledge.loader import load_documents
    from app.knowledge.splitter import split_documents

    docs = load_documents()
    chunks = split_documents(docs)
    add_chunks(chunks)