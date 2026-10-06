from typing import List, Dict
from app.retrieval.hybrid import search_hybrid
from app.retrieval.reranker import rerank
from app.config import config


# 拒答阈值：rerank 分数低于此值，认为知识库中没有相关内容
RERANK_THRESHOLD = 0.2


def retrieve(query: str, top_k: int = None) -> List[Dict]:
    """
    完整检索链路：混合检索 → RRF 融合 → Reranker 精排。
    返回 top_k 结果。如果最高分低于阈值，返回空列表。
    """
    if top_k is None:
        top_k = config.RERANK_TOP_K

    candidates = search_hybrid(query, top_k=10)
    results = rerank(query, candidates, top_k=top_k)

    if not results:
        return []

    if results[0]["rerank_score"] < RERANK_THRESHOLD:
        return []

    return results


if __name__ == "__main__":
    queries = [
        "退款要几天到账？",
        "新疆包邮吗？",
        "黄金会员打几折？",
        "发票怎么开？",
    ]
    for q in queries:
        print(f"\n{'='*60}\nQuery: {q}\n{'='*60}")
        results = retrieve(q)
        if not results:
            print(">>> 知识库中无相关内容，应转人工。")
            continue
        for i, r in enumerate(results):
            print(f"\n--- Top {i+1} (source: {r['source']}, score: {r['rerank_score']:.4f}) ---")
            print(r["content"])