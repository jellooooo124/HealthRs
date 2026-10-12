"""
Vector store for the diet-guideline knowledge base (Chroma + qwen3.7-text-embedding-flash).

Recipes are NOT embedded: the recommender works on structured constraints. Only the small guideline
corpus in data/guidelines/ (and documents uploaded in the demo) is indexed.
Supported formats: .md / .txt / .pdf. Re-ingesting identical content is skipped (MD5 hash).
"""
import hashlib
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.models.cloud_model import get_embeddings
from src.config import get_agent_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger("healthrec.rag")

CHROMA_DIR = os.path.join(get_project_root(), "data", "chroma_db")
GUIDELINE_DIR = os.path.join(get_project_root(), "data", "guidelines")
_HASHES = os.path.join(CHROMA_DIR, ".ingested_hashes")
_store = None


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _ingested() -> set:
    return set(open(_HASHES).read().split()) if os.path.exists(_HASHES) else set()


def get_vector_store() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings(),
                        collection_name=get_agent_config()["rag"]["collection_name"])
    return _store


def is_document_ingested(path: str) -> bool:
    return os.path.exists(path) and _file_hash(path) in _ingested()


def ingest_document(path: str) -> int:
    """Split and index one file; returns the number of chunks added (0 if already indexed)."""
    h = _file_hash(path)
    if h in _ingested():
        logger.info(f"[rag] already indexed: {os.path.basename(path)}")
        return 0
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".txt"):
        docs = TextLoader(path, autodetect_encoding=True).load()
    elif ext == ".pdf":
        docs = PyPDFLoader(path).load()
    else:
        raise ValueError(f"unsupported file type: {ext}")
    cfg = get_agent_config()["rag"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"],
                                              separators=["\n## ", "\n\n", "\n", ". ", " "])
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = os.path.basename(path)
    get_vector_store().add_documents(chunks)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(_HASHES, "a") as f:
        f.write(h + "\n")
    logger.info(f"[rag] indexed {os.path.basename(path)}: {len(chunks)} chunks")
    return len(chunks)


def search(query: str, k: int | None = None):
    k = k or get_agent_config()["rag"]["retrieval_top_k"]
    return get_vector_store().similarity_search(query, k=k)
