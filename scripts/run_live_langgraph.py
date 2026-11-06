"""Real local-LLM, three-agent LangGraph multi-turn smoke test, using live ES/MCP.

No LLM response is mocked. The OpenAI-compatible endpoint is served by Ollama
with an openly licensed *local* model. No DashScope/OpenAI account is required.

Run (after starting ES and Ollama and indexing the bundled sample):
    export HEALTHREC_RETRIEVAL_BACKEND=elasticsearch HEALTHREC_USE_BPR=0
    export HEALTHREC_MCP_PORT=18767
    export HEALTHREC_PROFILE_DB=/tmp/healthrec_llm_profile.sqlite
    export HEALTHREC_ES_INDEX=healthrec-langgraph-llm
    export HEALTHREC_OLLAMA_MODEL=qwen2.5:3b
    python -m scripts.run_live_langgraph --out outputs/live_langgraph
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from langchain_openai import ChatOpenAI

from agent.graph import HealthRecGraph
from agent.mcp_client import load_tools
from agent.runtime import HealthRecAgent
from agent.state_manager import StateManager
from prototype_elasticsearch.index_recipes import index_catalog
from prototype_elasticsearch.profiles import make_store
from prototype_elasticsearch.settings import es_client
from src.config import load_config


async def run_dialogs(out: Path, *, model: str, index: str) -> dict:
    es = es_client()
    assert es.ping(), "Real Elasticsearch service is required"
    indexed = index_catalog(source="sample", index=index, recreate=True)
    assert indexed["indexed"] > 0, indexed

    # The CLI must not load a cloud moderation LLM; the three LangGraph
    # agents themselves DO use the real local model.
    cfg = dict(load_config("agent_config")["pipeline"])
    cfg.update({"recommender_tool": True, "clarification": True,
                "safety_reviewer": True, "reviewer_llm_check": True,
                "reviewer_output_method": "json_schema"})
    load_config("agent_config")["guardrails"]["llm_check"] = False

    store = make_store()
    user_id = "langgraph_llm_e2e"
    store.create(user_id)
    # Baseline scenario keeps the already-confirmed profile simple. Allergy-
    # specific, real-ES exclusions are tested separately in test_elasticsearch_mcp_e2e.py.
    store.update(user_id, conditions=[], allergies=[],
                 taste_prefs=["chicken", "soup"], confirmed=True)

    llm = ChatOpenAI(
        model=model, base_url=os.getenv("HEALTHREC_LOCAL_LLM_URL", "http://127.0.0.1:11434/v1"),
        api_key="local-ollama-no-key", temperature=0, timeout=180, max_retries=1,
        max_tokens=380,
    )
    tools = await load_tools()
    graph = HealthRecGraph(tools, llm, cfg=cfg)
    # Preserve review/revision behaviour but limit local CPU smoke-test runtime.
    graph.max_revisions = 1
    runtime = HealthRecAgent(graph, tools, model, StateManager(str(out / "memory")))
    interactions = [
        "Recommend two chicken dinners under 45 minutes.",
        "Can you suggest two more chicken meals, different from the first ones?",
        # A third user turn is built below using an actual returned recipe ID.
    ]
    transcript = []
    for turn in range(3):
        if turn == 2:
            ids = [r["recipe_id"] for part in transcript for r in part["candidates"]]
            text = (f"What ingredients are in recipe #{ids[0]}?" if ids else
                    "Can you explain why you chose the recipes?")
        else:
            text = interactions[turn]
        start = time.perf_counter()
        response = await runtime.chat(user_id, text)
        observed_nodes = [entry.get("node") for entry in response.get("trace", [])]
        review = response.get("review") or {}
        transcript.append({
            "turn": turn + 1,
            "user": text,
            "assistant": response.get("reply", ""),
            "route": response.get("route"),
            "nodes": observed_nodes,
            "model": response.get("model"),
            "candidate_ids": [r["recipe_id"] for r in response.get("candidates", [])],
            "candidates": response.get("candidates", []),
            "review": review,
            "draft": response.get("debug_draft", ""),
            "profile_conditions": response.get("profile", {}).get("conditions", []),
            "profile_allergies": response.get("profile", {}).get("allergies", []),
            "tool_names": [entry.get("tool") for block in response.get("trace", [])
                           for entry in block.get("tools", [])],
            "latency_s": round(time.perf_counter() - start, 2),
            "llm_usage": response.get("usage"),
        })
        print(f"turn {turn+1}: route={response.get('route')}, nodes={observed_nodes}, "
              f"tool_names={transcript[-1]['tool_names']}, "
              f"candidate_ids={transcript[-1]['candidate_ids']}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.json").write_text(
        json.dumps({"model": model, "index": index, "turns": transcript}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # All nodes and actual tool calls must be seen, not merely constructed.
    for item in transcript:
        assert "profile_agent" in item["nodes"], item
        assert "recommender_agent" in item["nodes"], item
        assert "safety_reviewer" in item["nodes"], item
        assert item["review"].get("llm_passed") is not None, (
            "A REAL LLM safety reviewer must have returned a verdict", item["review"])
        assert item["assistant"].strip(), "Missing assistant reply"
        assert item["profile_conditions"] == [] and item["profile_allergies"] == [], (
            "The profile agent may not invent medical conditions or allergies", item)
    assert sum("recommend_recipes" in item["tool_names"] for item in transcript) >= 1, (
        "No real MCP recommend_recipes call was observed")
    assert any(item["candidate_ids"] for item in transcript), "No grounded recipe recommendations produced"
    assert any(item["route"] == "reviewed" for item in transcript), (
        "All turns fell back; model did not produce a reviewed output")
    return {"model": model, "turns": len(transcript),
            "reviewed": sum(x["route"] == "reviewed" for x in transcript),
            "all_safety_llm_checked": True,
            "grounded_recommendations": sum(bool(x["candidate_ids"]) for x in transcript)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("HEALTHREC_OLLAMA_MODEL", "qwen2.5:3b"))
    ap.add_argument("--out", type=Path, default=Path("outputs/live_langgraph"))
    args = ap.parse_args()
    if os.getenv("HEALTHREC_RETRIEVAL_BACKEND") != "elasticsearch":
        raise RuntimeError("Set HEALTHREC_RETRIEVAL_BACKEND=elasticsearch before starting")
    if os.getenv("HEALTHREC_USE_BPR") != "0":
        raise RuntimeError("Set HEALTHREC_USE_BPR=0 for sample-only run")
    index = os.getenv("HEALTHREC_ES_INDEX", "healthrec-langgraph-llm")
    print(json.dumps(asyncio.run(run_dialogs(args.out, model=args.model, index=index)), indent=2))


if __name__ == "__main__":
    main()
