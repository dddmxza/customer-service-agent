import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.retrieval.service import retrieve


def load_eval_set():
    path = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate(top_k: int = 3):
    eval_set = load_eval_set()

    total = len(eval_set)
    hit = 0
    reject_correct = 0
    reject_total = 0
    false_answer = 0  # 应该拒答但没拒答
    miss = 0          # 应该有答案但没找到

    print(f"Evaluating {total} questions, top_k={top_k}\n")
    print(f"{'='*70}")

    for item in eval_set:
        q = item["question"]
        expected_source = item["expected_source"]
        expected_keyword = item["expected_keyword"]

        results = retrieve(q, top_k=top_k)

        # 情况1：应该有答案
        if expected_source is not None:
            found = False
            for r in results:
                if r["source"] == expected_source and expected_keyword in r["content"]:
                    found = True
                    break
            if found:
                hit += 1
                print(f"✅ {q}")
            else:
                miss += 1
                print(f"❌ {q}  (期望: {expected_source} / {expected_keyword})")
                for r in results[:2]:
                    print(f"     实际: {r['source']} | {r['content'][:30]}...")

        # 情况2：应该拒答
        else:
            reject_total += 1
            if not results:
                reject_correct += 1
                print(f"✅ {q}  (正确拒答)")
            else:
                false_answer += 1
                print(f"⚠️  {q}  (应拒答但返回了结果)")
                print(f"     Top1: {results[0]['content'][:50]}...")

        print("-" * 70)

    print(f"\n{'='*70}")
    print(f"总问题数: {total}")
    print(f"应有答案的问题: {total - reject_total}")
    print(f"  命中: {hit}")
    print(f"  未命中: {miss}")
    print(f"  Recall@{top_k}: {hit / (total - reject_total) * 100:.1f}%")
    print(f"\n应拒答的问题: {reject_total}")
    print(f"  正确拒答: {reject_correct}")
    print(f"  错误回答: {false_answer}")
    print(f"  拒答准确率: {reject_correct / reject_total * 100:.1f}%")


if __name__ == "__main__":
    evaluate(top_k=3)