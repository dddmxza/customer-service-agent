from pathlib import Path
from typing import List, Dict
from app.config import config

def load_documents() -> List[Dict]:
    """
    加载 data/raw 下所有 markdown 和 txt 文档。
    返回列表，每个元素包含 content（内容）和 source（来源文件名）。
    """
    docs = []
    for path in config.RAW_DIR.glob("**/*"):
        if path.suffix in [".md", ".txt"]:
            content = path.read_text(encoding="utf-8")
            docs.append({
                "content": content,
                "source": path.name,
                "path": str(path)
            })
    return docs
