import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    # 路径
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    CHROMA_DIR = DATA_DIR / "chroma"
    BM25_INDEX_PATH = DATA_DIR / "bm25.pkl"

    # 模型
    EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
    RERANK_MODEL = "BAAI/bge-reranker-base"

    # 检索参数
    CHUNK_SIZE = 150
    CHUNK_OVERLAP = 50
    DENSE_TOP_K = 10
    SPARSE_TOP_K = 10
    RERANK_TOP_K = 5

    # LLM - DeepSeek
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DEEPSEEK_MODEL = "deepseek-flash"
    TEMPERATURE = 0

config = Config()