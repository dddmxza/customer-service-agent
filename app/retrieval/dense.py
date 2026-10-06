from typing import List, Dict
from app.knowledge.store import get_collection
from app.knowledge.embedder import embed_texts


def search_dense(query: str, top_k: int = 10) -> List[Dict]:
    """向量检索，返回统一格式的结果。"""
    collection = get_collection()
    q_vec = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
    )

    output = []
    for i, doc in enumerate(results["documents"][0]):
        output.append({
            "chunk_id": results["ids"][0][i],
            "content": doc,
            "source": results["metadatas"][0][i]["source"],
            "distance": float(results["distances"][0][i]),
        })
    return output


if __name__ == "__main__":
    for q in ["退款要几天到账？", "黄金会员打几折？"]:
        print(f"\n{'='*60}\nQuery: {q}\n{'='*60}")
        for i, r in enumerate(search_dense(q, top_k=3)):
            print(f"\n--- Top {i+1} (source: {r['source']}, distance: {r['distance']:.4f}) ---")
            print(r["content"])