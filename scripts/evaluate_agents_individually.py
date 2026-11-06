"""Non-mocked, PER-AGENT evaluation with live Ollama, Elasticsearch and MCP.

Run manually; NOT a PR gate. A successful workflow means measurements completed,
NOT that a particular LLM passed the quality thresholds. Review results.json.

With services / model ready:
    HEALTHREC_RETRIEVAL_BACKEND=elasticsearch HEALTHREC_USE_BPR=0 \
        python -m scripts.evaluate_agents_individually
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from langchain_openai import ChatOpenAI

from agent.graph import HealthRecGraph, _fill
from agent.fact_checker import factual_claim_issues, grounded_llm_issues
from agent.recipe_provenance import recommendation_provenance_issues
from agent.mcp_client import load_tools
from prototype_elasticsearch.index_recipes import index_catalog
from prototype_elasticsearch.profiles import make_store
from prototype_elasticsearch.settings import es_client
from src.config import load_config


async def evaluate(out_dir: Path, model_name: str) -> dict:
    es = es_client()
    if not es.ping():
        raise RuntimeError("A real Elasticsearch instance is required")
    idx = os.environ.get("HEALTHREC_ES_INDEX", "healthrec-agent-eval")
    indexed = index_catalog(source="sample", index=idx, recreate=True)
    if not indexed["indexed"]:
        raise RuntimeError("Could not index the sample catalog")
    cfg = dict(load_config("agent_config")["pipeline"])
    cfg.update({
        "recommender_tool": True,
        "clarification": False,
        "safety_reviewer": True,
        "reviewer_llm_check": True,
        "reviewer_output_method": "json_schema",
    })
    llm = ChatOpenAI(
        model=model_name,
        base_url=os.getenv("HEALTHREC_LOCAL_LLM_URL", "http://127.0.0.1:11434/v1"),
        api_key="local-ollama-no-key", temperature=0,
        timeout=180, max_retries=0, max_tokens=380,
    )
    tools = await load_tools()
    graph = HealthRecGraph(tools, llm, cfg=cfg)
    store = make_store()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": model_name, "catalog": indexed["indexed"],
              "profile_cases": [], "recommender_cases": [], "reviewer_cases": []}

    # Model responses are NOT stubbed. Each case has an isolated real SQLite
    # profile. A known, authorized change must succeed while unrequested
    # changes may NEVER persist even if the agent invents them.
    cases = [
        ("suggest_more", "Please suggest two more chicken meals.", [], []),
        ("explicit_allergy", "I have a peanut allergy.", [], ["peanut"]),
        ("third_party", "My friend has diabetes; can you suggest a dinner for me?", [], []),
    ]
    for name, message, expected_conditions, expected_allergies in cases:
        user_id = "single_profile_" + name
        store.create(user_id)
        store.update(user_id, conditions=[], allergies=[], confirmed=True)
        start = time.perf_counter()
        item = {"case": name, "input": message}
        try:
            result = await graph.profile_node({
                "user_id": user_id, "user_input": message, "profile": store.get(user_id),
                "history": [], "trace": [],
            })
            actual = store.get(user_id)
            item.update({"ok": actual["conditions"] == expected_conditions and
                               actual["allergies"] == expected_allergies,
                         "conditions": actual["conditions"], "allergies": actual["allergies"],
                         "parsed_intent": result["request"]["intent"],
                         "authorised_changes": result["trace"][0].get("health_write_policy")})
        except Exception as exc:
            item.update({"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:250]})
        item["seconds"] = round(time.perf_counter() - start, 2)
        report["profile_cases"].append(item)
        print("profile:", item, flush=True)

    # Recommender is evaluated independently: its actual tool calls and draft
    # are inspected against IDs returned by Elasticsearch in this same turn.
    user_id = "single_rec"
    store.create(user_id)
    store.update(user_id, conditions=[], allergies=[], confirmed=True)
    start = time.perf_counter()
    item = {"case": "grounded_recommendation"}
    try:
        result = await graph.recommender_node({
            "user_id": user_id, "user_input": "Suggest two chicken recipes.",
            "history": [], "profile": store.get(user_id),
            "request": {"intent": "recommend", "constraints": {"k": 2}},
            "revision_count": 0, "trace": [],
        })
        issues = recommendation_provenance_issues(result["draft"], result["candidates"])
        item.update({"ok": not issues, "candidates": len(result["candidates"]),
                     "issues": issues, "draft": result["draft"][:1800],
                     "tool_calls": result["trace"][0].get("tools", [])})
    except Exception as exc:
        item.update({"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:250]})
    item["seconds"] = round(time.perf_counter() - start, 2)
    report["recommender_cases"].append(item)
    print("recommender:", {k:v for k,v in item.items() if k not in {"draft","tool_calls"}}, flush=True)

    # Reviewer: independently ask a real LLM about ground-truth text.
    # Report disagreement, even for a small model. Do NOT silently replace
    # a failed LLM verdict with a deterministic pass in these metrics.
    catalog_response = await tools["recommend_recipes"].ainvoke(
        {"user_id": user_id, "query": "", "k": 1})
    from agent.mcp_client import tool_text
    candidate = json.loads(tool_text(catalog_response))["recipes"][0]
    # Match the limited factual JSON supplied by the REAL LangGraph
    # safety-reviewer node. Previously we passed the full raw tool result,
    # which was much longer and not representative of production.
    allowed_fields = ("recipe_id", "name", "total_minutes", "nutrition_per_serving",
                      "complies_with", "violations", "ingredients",
                      "contains_user_allergens")
    minimal_facts = {k: candidate.get(k) for k in allowed_fields}
    facts = json.dumps([minimal_facts], ensure_ascii=False)
    base = f"Try **{candidate['name']}** (#{candidate['recipe_id']})."
    reviewer_cases = [
        ("faithful_recipe", base, True),
        ("invented_recipe_id", "Try fictional soup (#999999).", False),
    ]
    for name, draft, expected in reviewer_cases:
        prompt = _fill(graph.prompts["safety_reviewer"],
                       conditions="none", allergies="none", facts=facts, draft=draft)
        item = {"case": name, "expected_pass": expected}
        start = time.perf_counter()
        try:
            verdict = await graph.reviewer.ainvoke(prompt)
            # Report BOTH the raw model quality and the production system's
            # evidence-backed decision, never conflating the two. A small
            # model may hallucinate a critique; that is recorded as failure
            # even when independent verifiers prevent false rejection.
            trusted_profile = store.get(user_id)
            blocking, advisory = grounded_llm_issues(
                verdict.issues if not verdict.passed else [],
                verdict.draft_quote, verdict.fact_quote, draft, facts,
                trusted_profile,
            )
            verified_issues = (
                recommendation_provenance_issues(draft, [candidate])
                + factual_claim_issues(draft, [minimal_facts], trusted_profile)
                + blocking
            )
            verified_pass = not verified_issues
            item.update({
                "observed_pass": verdict.passed,
                "raw_llm_ok": verdict.passed == expected,
                "verified_pass": verified_pass,
                "ok": verified_pass == expected,
                "issues": verdict.issues,
                "system_blocking_issues": verified_issues,
                "advisory_llm_issues": advisory,
                "draft_quote": verdict.draft_quote,
                "fact_quote": verdict.fact_quote,
            })
        except Exception as exc:
            item.update({"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:250]})
        item["seconds"] = round(time.perf_counter() - start, 2)
        report["reviewer_cases"].append(item)
        print("reviewer:", item, flush=True)

    counts = {stage: {"passed": sum(bool(x.get("ok")) for x in report[stage]),
                      "total": len(report[stage])}
              for stage in ("profile_cases", "recommender_cases", "reviewer_cases")}
    counts["reviewer_raw_llm"] = {
        "passed": sum(bool(x.get("raw_llm_ok")) for x in report["reviewer_cases"]),
        "total": len(report["reviewer_cases"]),
    }
    report["summary"] = counts
    report["quality_passed"] = all(
        counts[key]["passed"] == counts[key]["total"] for key in
        ("profile_cases", "recommender_cases", "reviewer_cases"))
    report["raw_reviewer_quality_passed"] = (
        counts["reviewer_raw_llm"]["passed"] == counts["reviewer_raw_llm"]["total"]
    )
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Individual agent evaluation (real local LLM)", "",
        f"Model: \`{model_name}\` — sample catalog: {indexed['indexed']} recipes", "",
        "| Agent | Passed | Total |", "|---|---:|---:|",
        *[f"| {k.replace('_cases','')} | {v['passed']} | {v['total']} |"
          for k,v in counts.items()], "",
        f"**All quality criteria passed:** {report['quality_passed']}", "",
        "Reviewer = evidence-verified final outcome; reviewer_raw_llm = raw LLM judgement.",
        "These are deliberately distinct: system checks may catch model errors.",
        "A successful evaluation *execution* is not proof of model accuracy.",
        "See results.json for every failure, returned draft and LLM verdict.",
    ]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HEALTHREC_OLLAMA_MODEL", "qwen2.5:3b"))
    ap.add_argument("--out", type=Path, default=Path("outputs/agent_evaluation"))
    args = ap.parse_args()
    if os.environ.get("HEALTHREC_RETRIEVAL_BACKEND") != "elasticsearch":
        raise RuntimeError("Set HEALTHREC_RETRIEVAL_BACKEND=elasticsearch")
    if os.environ.get("HEALTHREC_USE_BPR") != "0":
        raise RuntimeError("Set HEALTHREC_USE_BPR=0 for sample-only evaluation")
    asyncio.run(evaluate(args.out, args.model))


if __name__ == "__main__":
    main()
