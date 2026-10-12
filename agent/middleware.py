"""
Agent middleware (LangChain 1.x) shared by the profile and recommender agents.

Per-request state lives in `AgentContext`, passed as the runtime context of each agent call
(`agent.ainvoke(..., context=AgentContext(...))`), so concurrent requests and different users never
share state.

1. inject_user_id  (@wrap_tool_call)  forces `user_id` in every tool call to the logged-in user —
   the LLM cannot read or modify another user's profile — and records each call in ctx.trace.
2. context_prompt  (@dynamic_prompt)  uses the per-request system prompt built by the graph node
   (profile summary, parsed request, reviewer feedback).
"""
import time
from dataclasses import dataclass, field

from langchain.agents.middleware import dynamic_prompt, wrap_tool_call

from src.utils.logger import get_logger

logger = get_logger("healthrec.middleware")


@dataclass
class AgentContext:
    user_id: str
    system_prompt: str = ""
    trace: list = field(default_factory=list)      # [{tool, args, ms, ok, preview}] for this request only


@wrap_tool_call
async def inject_user_id(request, handler):
    ctx: AgentContext = request.runtime.context
    call = dict(request.tool_call)
    args = dict(call.get("args") or {})
    tool_params = getattr(request.tool, "args", {}) or {}
    if "user_id" in tool_params or "user_id" in args:
        if args.get("user_id") not in (None, "", ctx.user_id):
            logger.info(f"[middleware] overriding LLM-supplied user_id={args.get('user_id')!r}")
        args["user_id"] = ctx.user_id
    request = request.override(tool_call={**call, "args": args})
    t0 = time.perf_counter()
    try:
        result = await handler(request)
        ok = True
        return result
    except Exception as e:
        ok, result = False, e
        raise
    finally:
        shown = {k: v for k, v in args.items() if k != "user_id" and v not in (None, [], "")}
        content = getattr(result, "content", result)
        ctx.trace.append({"tool": call.get("name"), "args": shown, "ms": round((time.perf_counter() - t0) * 1000),
                          "ok": ok, "preview": str(content)[:160]})
        logger.info(f"[tool] {call.get('name')}({shown}) {'ok' if ok else 'FAILED'}")


@dynamic_prompt
async def context_prompt(request) -> str:
    return request.runtime.context.system_prompt


AGENT_MIDDLEWARE = [inject_user_id, context_prompt]
