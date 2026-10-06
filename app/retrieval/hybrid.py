from typing import List, Dict
from app.retrieval.dense import search_dense
from app.retrieval.sparse import search_bm25


def rrf_fusion(results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    倒数排名融合。
    results_list: 多条检索链的结果，每条链是一个按排名排序的列表。
    """
    scores = {}
    doc_info = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
            if cid not in doc_info:
                doc_info[cid] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    output = []
    for cid, score in ranked:
        item = dict(doc_info[cid])
        item["rrf_score"] = score
        output.append(item)
    return output


def search_hybrid(query: str, top_k: int = 5) -> List[Dict]:
    dense_results = search_dense(query, top_k=10)
    sparse_results = search_bm25(query, top_k=10)
    fused = rrf_fusion([dense_results, sparse_results])
    return fused[:top_k]


if __name__ == "__main__":
    queries = [
        "退款要几天到账？",
        "新疆包邮吗？",
        "黄金会员打几折？",
        "发票怎么开？",
    ]
    for q in queries:
        print(f"\n{'='*60}\nQuery: {q}\n{'='*60}")
        results = search_hybrid(q, top_k=3)
        for i, r in enumerate(results):
            print(f"\n--- Top {i+1} (source: {r['source']}, rrf: {r['rrf_score']:.5f}) ---")
            print(r["content"])