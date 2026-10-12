"""
Scripted multi-turn conversations through the full 3-agent pipeline (uses real LLM calls).

    python scripts/agent_smoke_test.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.profile_store import ProfileStore  # noqa: E402
from agent.runtime import HealthRecAgent  # noqa: E402

SCRIPT = [
    ("1634", "Can you suggest a dinner I can make in under 45 minutes?"),
    ("1634", "The first one sounds too plain and I don't like asparagus. Something with chicken instead?"),
    ("demo_bob", "I'd like some breakfast ideas."),
    ("demo_bob", "I have type 2 diabetes and I'm allergic to peanuts."),
    ("1634", "How much salt can I have per day with my blood pressure?"),
]


async def main():
    ProfileStore().reset_demo_user("demo_bob")
    agent = await HealthRecAgent.create()
    for uid in {u for u, _ in SCRIPT}:
        agent.reset_memory(uid)
    for uid, text in SCRIPT:
        out = await agent.chat(uid, text)
        print("=" * 100)
        print(f"[{uid}] USER: {text}")
        print(f"route={out['route']} revisions={out['revisions']} latency={out['latency_s']}s "
              f"llm_calls={out['usage']['llm_calls']} tokens={out['usage']['total_tokens']}")
        if out.get("request"):
            print("request:", json.dumps(out["request"], ensure_ascii=False))
        for tr in out["trace"]:
            for t in tr.get("tools", []):
                print(f"   tool[{tr['node']}] {t['tool']}({t['args']}) {t['ms']}ms ok={t['ok']}")
            if tr.get("node") == "safety_reviewer":
                print("   review:", tr["review"])
        print("ASSISTANT:\n" + out["reply"])


if __name__ == "__main__":
    asyncio.run(main())
