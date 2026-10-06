from sentence_transformers import CrossEncoder
from typing import List, Dict
from app.config import config


_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        model_path = str(config.BASE_DIR / "models" / "bge-reranker-base")
        print(f"Loading reranker from: {model_path}")
        _model = CrossEncoder(model_path)
        print("Reranker loaded")
    return _model


def rerank(query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
    """对候选文档重排，返回 top_k。"""
    if not candidates:
        return []

    model = get_reranker()
    pairs = [[query, c["content"]] for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    output = []
    for doc, score in ranked[:top_k]:
        item = dict(doc)
        item["rerank_score"] = float(score)
        output.append(item)
    return output


if __name__ == "__main__":
    from app.retrieval.hybrid import search_hybrid

    queries = [
        "退款要几天到账？",
        "新疆包邮吗？",
        "黄金会员打几折？",
        "发票怎么开？",
    ]
    for q in queries:
        print(f"\n{'='*60}\nQuery: {q}\n{'='*60}")
        candidates = search_hybrid(q, top_k=10)
        results = rerank(q, candidates, top_k=3)
        for i, r in enumerate(results):
            print(f"\n--- Top {i+1} (source: {r['source']}, rerank: {r['rerank_score']:.4f}) ---")
            print(r["content"])