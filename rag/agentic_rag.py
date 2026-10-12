"""
Agentic RAG over the diet-guideline corpus (3 steps):
  1. query rewriting      (LLM; switchable via rag.query_rewrite — off in evaluation to save tokens)
  2. multi-path retrieval (original + rewritten queries, de-duplicated)
  3. self-reflection      (LLM keeps only passages that answer the question; rag.self_reflection switch)
"""
from langchain_core.prompts import ChatPromptTemplate

from agent.models.cloud_model import CloudChatModel
from rag.vector_stores import search
from src.config import get_agent_config, load_config
from src.utils.logger import get_logger

logger = get_logger("healthrec.agentic_rag")
_llm = None


def _helper_llm():
    global _llm
    if _llm is None:
        _llm = CloudChatModel("agent", temperature=0.0).llm_safe
    return _llm


def search_guidelines(query: str) -> str:
    cfg, prompts = get_agent_config()["rag"], load_config("prompts")
    queries = [query]
    if cfg.get("query_rewrite", True):
        try:
            out = (ChatPromptTemplate.from_template(prompts["rag_query_rewrite"]) | _helper_llm()).invoke({"query": query})
            queries += [q.strip("-• ").strip() for q in out.content.splitlines() if q.strip()][:3]
        except Exception as e:   # retrieval still works with the original query
            logger.warning(f"[rag] rewrite failed: {e}")
    seen, docs = set(), []
    for q in queries:
        for d in search(q):
            key = d.page_content[:120]
            if key not in seen:
                seen.add(key)
                docs.append(d)
    if not docs:
        return "No guideline passage found."
    context = "\n\n".join(f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs)
    if not cfg.get("self_reflection", True):
        return context
    out = (ChatPromptTemplate.from_template(prompts["rag_reflection"]) | _helper_llm()).invoke({"query": query, "context": context})
    text = out.content.strip()
    return "No relevant guideline passage found." if "NOT_FOUND" in text else text
