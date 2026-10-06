import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.knowledge.store import get_collection
from app.knowledge.embedder import embed_texts


def search(query: str, top_k: int = 3):
    collection = get_collection()
    q_vec = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
    )

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        distance = results["distances"][0][i]
        print(f"\n--- Top {i+1} (source: {source}, distance: {distance:.4f}) ---")
        print(doc)


if __name__ == "__main__":
    search("退款要几天到账？")
    search("新疆包邮吗？")
    search("黄金会员打几折？")
    search("发票怎么开？")