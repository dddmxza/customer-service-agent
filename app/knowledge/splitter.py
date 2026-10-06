from typing import List, Dict
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.config import config


def split_documents(docs: List[Dict]) -> List[Dict]:
    """
    先用 MarkdownHeaderTextSplitter 按标题切，
    再用 RecursiveCharacterTextSplitter 保证每块不超过 chunk_size。
    """
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in docs:
        md_chunks = md_splitter.split_text(doc["content"])
        for i, mc in enumerate(md_chunks):
            piece = mc.page_content
            # 如果这块还是太长，再用字符切分器二次切
            sub_pieces = char_splitter.split_text(piece)
            for j, sp in enumerate(sub_pieces):
                chunks.append({
                    "content": sp,
                    "source": doc["source"],
                    "chunk_id": f"{doc['source']}_{i}_{j}",
                })
    return chunks


if __name__ == "__main__":
    from app.knowledge.loader import load_documents

    docs = load_documents()
    chunks = split_documents(docs)
    print(f"共 {len(chunks)} 个 chunk\n")
    for c in chunks:
        print(f"[{c['chunk_id']}] 来源: {c['source']}")
        print(c["content"])
        print("-" * 50)