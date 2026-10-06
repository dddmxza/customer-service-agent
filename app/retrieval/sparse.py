import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict
from app.knowledge.store import get_collection


_bm25 = None
_docs = None
_ids = None
_metadatas = None


def build_bm25():
    """从 Chroma 取出所有 chunk，构建 BM25 索引。"""
    global _bm25, _docs, _ids, _metadatas

    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])

    _docs = result["documents"]
    _ids = result["ids"]
    _metadatas = result["metadatas"]

    tokenized = [list(jieba.cut(doc)) for doc in _docs]
    _bm25 = BM25Okapi(tokenized)
    print(f"BM25 index built with {len(_docs)} chunks.")


def get_bm25():
    if _bm25 is None:
        build_bm25()
    return _bm25


def search_bm25(query: str, top_k: int = 10) -> List[Dict]:
    """用 BM25 检索，返回 top_k 结果。"""
    bm25 = get_bm25()
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)

    ranked_idx = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []
    for i in ranked_idx:
        if scores[i] <= 0:
            continue
        results.append({
            "chunk_id": _ids[i],
            "content": _docs[i],
            "source": _metadatas[i]["source"],
            "score": float(scores[i]),
        })
    return results


if __name__ == "__main__":
    queries = [
        "退款要几天到账？",
        "新疆包邮吗？",
        "黄金会员打几折？",
        "发票怎么开？",
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        results = search_bm25(q, top_k=3)
        for i, r in enumerate(results):
            print(f"\n--- Top {i+1} (source: {r['source']}, score: {r['score']:.4f}) ---")
            print(r["content"])