"""A REAL Elasticsearch + FastAPI + streamable MCP end-to-end test (no LLM key).

Run with: pytest -q -m live_es
Start Elasticsearch first, e.g. docker compose -f prototype_elasticsearch/compose.yaml up -d
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from prototype_elasticsearch.index_recipes import index_catalog
from prototype_elasticsearch.settings import es_client


@pytest.mark.live_es
def test_real_elasticsearch_index_fastapi_and_mcp(tmp_path, monkeypatch):
    es = es_client()
    for _ in range(45):
        if es.ping():
            break
        time.sleep(2)
    assert es.ping(), "Elasticsearch must actually be running for E2E tests"
    index = "healthrec-mcp-e2e"
    indexed = index_catalog(source="sample", index=index, recreate=True, limit=400)
    assert indexed["indexed"] > 0
    assert indexed["count_on_es"] == indexed["indexed"]

    db = str(tmp_path / "profiles.sqlite")
    monkeypatch.setenv("HEALTHREC_PROFILE_DB", db)
    monkeypatch.setenv("HEALTHREC_ES_INDEX", index)
    monkeypatch.setenv("HEALTHREC_RETRIEVAL_BACKEND", "elasticsearch")
    monkeypatch.setenv("HEALTHREC_USE_BPR", "0")
    monkeypatch.setenv("HEALTHREC_MCP_PORT", "18765")
    monkeypatch.setenv("HEALTHREC_PROFILE_DB", db)

    # Test the actual FastAPI route against the live index, not a stubbed client.
    from fastapi.testclient import TestClient
    from prototype_elasticsearch.server import app
    with TestClient(app) as http:
        health = http.get("/health")
        assert health.status_code == 200
        assert health.json()["count"] > 0
        p = http.put("/profiles/e2e_user", json={
            "conditions": [], "allergies": ["peanut"], "taste_prefs": ["chicken"], "confirmed": True,
        })
        assert p.status_code == 200
        rsp = http.post("/recommendations", json={
            "user_id": "e2e_user", "query": "", "k": 5,
        })
        assert rsp.status_code == 200
        assert rsp.json()["recipes"], "Expected indexed safe recipes"
        assert all("peanut" not in rec["allergens"] for rec in rsp.json()["recipes"])

    # Start actual MCP server in ES mode and use the SAME tool loader the LangGraph
    # agent uses. No external AI API or mock is involved in this transport test.
    proc_env = os.environ.copy()
    log_path = tmp_path / "mcp.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agent.mcp_server"],
            env=proc_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    try:
        for _ in range(45):
            if proc.poll() is not None:
                pytest.fail("MCP server exited: " + log_path.read_text()[-3000:])
            try:
                with socket.create_connection(("127.0.0.1", 18765), timeout=0.4):
                    break
            except OSError:
                time.sleep(1)
        else:
            pytest.fail("MCP server did not start: " + log_path.read_text()[-3000:])

        async def run_tools():
            from agent.mcp_client import load_tools, tool_text
            tools = await load_tools()
            assert {"get_user_profile", "recommend_recipes", "get_recipe_details",
                    "record_feedback"}.issubset(tools.keys())
            profile = json.loads(tool_text(await tools["get_user_profile"].ainvoke({"user_id": "e2e_user"})))
            assert profile["allergies"] == ["peanut"]
            first = json.loads(tool_text(await tools["recommend_recipes"].ainvoke({
                "user_id": "e2e_user", "query": "", "k": 5,
            })))
            assert first["backend"] == "elasticsearch"
            assert first["recipes"]
            assert all(not r["violations"] for r in first["recipes"])
            rid = first["recipes"][0]["recipe_id"]
            details = json.loads(tool_text(await tools["get_recipe_details"].ainvoke({
                "recipe_id": rid, "user_id": "e2e_user",
            })))
            assert details["recipe_id"] == rid
            feedback = json.loads(tool_text(await tools["record_feedback"].ainvoke({
                "user_id": "e2e_user", "recipe_id": rid, "signal": "dislike",
            })))
            assert feedback["status"] == "recorded"
            again = json.loads(tool_text(await tools["recommend_recipes"].ainvoke({
                "user_id": "e2e_user", "query": "", "k": 5, "exclude_recipe_ids": [rid],
            })))
            assert rid not in [r["recipe_id"] for r in again["recipes"]]

            # Real LangGraph graph, real running Elasticsearch, and real HTTP
            # MCP tools; ONLY the LLM replies are deliberately deterministic.
            # This test validates the actual graph routing and hard safety gates.
            from langchain_core.messages import AIMessage, ToolMessage
            from agent.graph import HealthRecGraph, ProfileDecision
            from src.config import load_config

            class FixedProfileAgent:
                async def ainvoke(self, *args, **kwargs):
                    return {"structured_response": ProfileDecision(intent="recommend")}

            class FixedRecAgent:
                def __init__(self, invent=False):
                    self.invent = invent

                async def ainvoke(self, *args, **kwargs):
                    real_result = tool_text(await tools["recommend_recipes"].ainvoke({
                        "user_id": "e2e_user", "query": "", "k": 2,
                    }))
                    candidates = json.loads(real_result)["recipes"]
                    assert candidates
                    correct = candidates[0]
                    cited_id = 130001 if self.invent else correct["recipe_id"]
                    text = f"Try **{correct['name']}** (#{cited_id})."
                    return {"messages": [
                        ToolMessage(content=real_result, name="recommend_recipes", tool_call_id="verified_1"),
                        AIMessage(content=text),
                    ]}

            g = object.__new__(HealthRecGraph)
            g.cfg = {"recommender_tool": True, "clarification": False,
                     "safety_reviewer": True, "reviewer_llm_check": False,
                     "history_turns": 4}
            g.tools = tools
            g.prompts = load_config("prompts")
            g.profile_agent = FixedProfileAgent()
            g.max_revisions = 0
            g.rec_agent = FixedRecAgent()
            g.graph = g._build()

            valid = await g.graph.ainvoke({
                "user_id": "e2e_user",
                "user_input": "Please recommend two chicken recipes.",
                "history": [], "trace": [],
            })
            assert valid["route"] == "reviewed", valid.get("review")
            assert valid["review"]["rule_issues"] == []
            assert f"#{valid['candidates'][0]['recipe_id']}" in valid["reply"]
            assert valid["candidates"]
            assert valid["profile"]["allergies"] == ["peanut"]

            g.rec_agent = FixedRecAgent(invent=True)
            invalid = await g.graph.ainvoke({
                "user_id": "e2e_user",
                "user_input": "Please show me two more chicken dishes.",
                "history": [], "trace": [],
            })
            assert invalid["route"] == "fallback", invalid
            assert any("#130001" in issue for issue in invalid["review"]["rule_issues"])
            assert "#130001" not in invalid["reply"]
            assert invalid["profile"]["allergies"] == ["peanut"]

            # A genuine first-person statement, unlike the "more recipes"
            # request, is allowed to modify the real SQLite profile via MCP.
            prior = json.loads(tool_text(await tools["get_user_profile"].ainvoke(
                {"user_id": "e2e_user"})))
            explicit = await g.profile_node({
                "user_id": "e2e_user", "user_input": "I have a milk allergy.",
                "profile": prior, "history": [], "trace": [],
            })
            assert explicit["profile"]["allergies"] == ["milk", "peanut"]
            assert explicit["trace"][0]["health_write_policy"] == {
                "allergies": ["milk", "peanut"],
            }
            return first, again
        asyncio.run(run_tools())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        es.indices.delete(index=index, ignore_unavailable=True)
