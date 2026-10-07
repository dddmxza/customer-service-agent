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

