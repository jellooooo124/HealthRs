"""
Entry point used by the demo and the evaluation scripts.

    agent = await HealthRecAgent.create()                 # role "agent" (dev) or "eval" (pinned eval model)
    out = await agent.chat("1634", "Something quick with chicken tonight?")
    out["reply"], out["trace"], out["usage"]

For sync callers (Streamlit) use SyncHealthRecAgent, which runs a private event loop in a thread.
"""
import asyncio
import re
import threading
import time
import uuid

from agent.graph import HealthRecGraph
from agent.guardrails import check_input, redact_pii
from agent.mcp_client import load_tools
from agent.models.cloud_model import CloudChatModel, UsageTracker
from agent.state_manager import StateManager
from src.config import load_config


class HealthRecAgent:
    def __init__(self, graph: HealthRecGraph, tools: dict, model_name: str, memory: StateManager,
                 fallback_names: list[str] | None = None):
        self.graph, self.tools, self.model_name, self.memory = graph, tools, model_name, memory
        self.fallback_names = fallback_names or []

    @classmethod
    async def create(cls, role: str = "agent", pipeline_overrides: dict | None = None,
                     memory_dir: str | None = None) -> "HealthRecAgent":
        tools = await load_tools()
        model = CloudChatModel(role)
        cfg = {**load_config("agent_config")["pipeline"], **(pipeline_overrides or {})}
        return cls(HealthRecGraph(tools, model.llm, cfg, model.fallback_llms), tools, model.model_name,
                   StateManager(memory_dir), model.fallback_names)

    @staticmethod
    def _session(user_id: str) -> str:
        return "user_" + re.sub(r"[^A-Za-z0-9_\-]", "_", str(user_id))

    async def chat(self, user_id: str, text: str) -> dict:
        t0 = time.perf_counter()
        usage = UsageTracker()
        ok, reason = await check_input(text)
        if not ok:
            return {"reply": f"Sorry, I can't help with that request ({reason}). Ask me about recipes or diet!",
                    "route": "blocked", "trace": [], "usage": usage.as_dict(), "latency_s": 0.0}
        text = redact_pii(text)
        session = self._session(user_id)
        state = {"user_id": str(user_id), "user_input": text, "history": self.memory.load_history(session), "trace": []}
        out = await self.graph.graph.ainvoke(state, config={"callbacks": [usage], "run_id": uuid.uuid4(),
                                                            "recursion_limit": 40})
        reply = redact_pii(out.get("reply", ""))
        self.memory.add_conversation(session, text, reply, kind="clarification" if out.get("route") == "clarify" else "answer")
        return {"reply": reply, "route": out.get("route"), "request": out.get("request"),
                "profile": out.get("profile"), "candidates": out.get("candidates", []),
                "review": out.get("review"), "revisions": out.get("revision_count", 0),
                "trace": out.get("trace", []), "usage": usage.as_dict(), "model": self.model_name,
                "latency_s": round(time.perf_counter() - t0, 2)}

    def reset_memory(self, user_id: str):
        self.memory.clear_memory(self._session(user_id))


class SyncHealthRecAgent:
    """Thread-safe synchronous facade (one background event loop) for Streamlit and scripts."""

    def __init__(self, **kwargs):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self.agent: HealthRecAgent = self._run(HealthRecAgent.create(**kwargs))

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def chat(self, user_id: str, text: str) -> dict:
        return self._run(self.agent.chat(user_id, text))

    def call_tool(self, name: str, **args) -> str:
        from agent.mcp_client import tool_text
        return tool_text(self._run(self.agent.tools[name].ainvoke(args)))
