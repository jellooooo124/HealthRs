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
