"""
Three-agent LangGraph pipeline (fixed order, no LLM router):

    user message
       │
    load_profile ─► PROFILE AGENT ── need clarification? ──yes──► clarify ─► END (ask one question)
                    (parse request, update profile, record feedback)
                       │ no
                       ▼
                    RECOMMENDER AGENT (recommend_recipes tool → grounded draft; guideline RAG)
                       │
                       ▼
                    SAFETY REVIEWER (rule check + LLM check)
                       ├─ pass ───────────────────────────► finalize ─► END
                       ├─ fail, revisions < max ──────────► RECOMMENDER AGENT (with reviewer issues)
                       └─ fail after max revisions ───────► fallback (conservative reply) ─► END

Switches (configs/agent_config.yaml, `pipeline`): clarification, safety_reviewer, reviewer_llm_check,
recommender_tool (False = pure-LLM baseline), history_turns.
"""
import json
import operator
import re
import time
from typing import Annotated, Literal, Optional, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent.mcp_client import tool_text
from agent.middleware import AGENT_MIDDLEWARE, AgentContext
from src.config import load_config
from src.utils.logger import get_logger

logger = get_logger("healthrec.graph")
_CITE = re.compile(r"#(\d{2,7})\b")


# ---------------------------------------------------------------- structured outputs
class Constraints(BaseModel):
    meal_type: Optional[Literal["breakfast", "main", "side", "dessert", "snack", "drink"]] = Field(
        None, description="breakfast; main = lunch/dinner main dish; side; dessert; snack; drink")
    max_minutes: Optional[int] = Field(None, description="maximum total cooking time in minutes")
    max_calories: Optional[float] = Field(None, description="maximum kcal per serving")
    categories: list[str] = Field(default_factory=list,
                                  description="only for a specific dish family, e.g. Chicken, Pork, Soups, Curries")
    include_ingredients: list[str] = Field(default_factory=list)
    exclude_ingredients: list[str] = Field(default_factory=list)
    exclude_recipe_ids: list[int] = Field(default_factory=list)
    k: int = Field(5, description="number of recipes, 1-10")


class ProfileDecision(BaseModel):
    """Structured result of the profile agent."""
    intent: Literal["recommend", "question", "recipe_detail", "profile_update", "chitchat"]
    constraints: Constraints = Field(default_factory=Constraints)
    need_clarification: bool = False
    clarification_question: Optional[str] = None
    summary: str = Field("", description="one sentence: what the user wants")


class ReviewResult(BaseModel):
    """Safety reviewer verdict."""
    passed: bool
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- graph state
class GraphState(TypedDict, total=False):
    user_id: str
    user_input: str
    history: list                 # previous HumanMessage/AIMessage (sliding window)
    profile: dict
    request: dict
    candidates: list              # recipes returned by recommend_recipes this turn
    draft: str
    review: dict
    revision_count: int
    reply: str
    route: str                    # how the turn ended: clarify | reviewed | unreviewed | fallback
    trace: Annotated[list, operator.add]


def _fill(template: str, **kw) -> str:
    for k, v in kw.items():
        template = template.replace(f"<<{k}>>", v)
    return template


def _profile_text(p: dict) -> str:
    keep = {k: p.get(k) for k in ("conditions", "allergies", "goals", "taste_prefs", "confirmed", "source")}
    keep["history"] = "has rating history" if p.get("source") == "foodcom_warm" else "no rating history (cold start)"
    return json.dumps(keep, ensure_ascii=False)


class HealthRecGraph:
    def __init__(self, tools: dict, llm, cfg: dict | None = None, fallback_llms: list | None = None):
        self.cfg = cfg or load_config("agent_config")["pipeline"]
        self.max_revisions = load_config("agent_config")["review"]["max_revisions"]
        self.prompts = load_config("prompts")
        self.tools, self.llm = tools, llm
        # fallback models (demo): tried in order when a call to the main model fails
        fallback_llms = fallback_llms or []
        middleware = AGENT_MIDDLEWARE + ([ModelFallbackMiddleware(*fallback_llms)] if fallback_llms else [])
        self.profile_agent = create_agent(
            llm, [tools["get_user_profile"], tools["update_user_profile"], tools["record_feedback"]],
            middleware=middleware, context_schema=AgentContext,
            response_format=ToolStrategy(ProfileDecision), name="profile_agent")
        rec_tools = [tools["get_recipe_details"], tools["search_diet_guidelines"]]
        if self.cfg.get("recommender_tool", True):
            rec_tools.insert(0, tools["recommend_recipes"])
        self.rec_agent = create_agent(llm, rec_tools, middleware=middleware, context_schema=AgentContext,
                                      name="recommender_agent")
        self.reviewer = llm.with_structured_output(ReviewResult, method="function_calling")
        if fallback_llms:
            self.reviewer = self.reviewer.with_fallbacks(
                [f.with_structured_output(ReviewResult, method="function_calling") for f in fallback_llms])
        self.graph = self._build()

    # ------------------------------------------------------------ helpers
    def _window(self, state: GraphState) -> list:
        """Recent history (older assistant replies truncated to save tokens) + the new user message."""
        n = self.cfg.get("history_turns", 4)
        hist = list(state.get("history", []))[-n:]
        out = [AIMessage(m.content[:800] + (" …" if len(m.content) > 800 else ""), additional_kwargs=m.additional_kwargs)
               if isinstance(m, AIMessage) else m for m in hist]
        return out + [HumanMessage(state["user_input"])]

    async def _tool(self, name: str, **args) -> str:
        return tool_text(await self.tools[name].ainvoke(args))

    # ------------------------------------------------------------ nodes
    async def load_profile(self, state: GraphState) -> dict:
        p = json.loads(await self._tool("get_user_profile", user_id=state["user_id"]))
        return {"profile": p, "revision_count": 0, "trace": [{"node": "load_profile"}]}

    async def profile_node(self, state: GraphState) -> dict:
        t0 = time.perf_counter()
        ctx = AgentContext(user_id=state["user_id"], system_prompt=_fill(
            self.prompts["profile_agent"], profile=_profile_text(state["profile"])))
        out = await self.profile_agent.ainvoke({"messages": self._window(state)}, context=ctx)
        decision: ProfileDecision = out.get("structured_response") or ProfileDecision(intent="recommend")
        profile = json.loads(await self._tool("get_user_profile", user_id=state["user_id"]))   # may have changed
        return {"request": decision.model_dump(), "profile": profile,
                "trace": [{"node": "profile_agent", "ms": round((time.perf_counter() - t0) * 1000),
                           "decision": decision.model_dump(), "tools": ctx.trace}]}

    def route_after_profile(self, state: GraphState) -> str:
        req = state["request"]
        last_ai = next((m for m in reversed(state.get("history", [])) if isinstance(m, AIMessage)), None)
        asked_before = bool(last_ai and last_ai.additional_kwargs.get("kind") == "clarification")
        if self.cfg.get("clarification", True) and req.get("need_clarification") and not asked_before:
            return "clarify"
        return "recommender"

    async def clarify_node(self, state: GraphState) -> dict:
        q = state["request"].get("clarification_question") or (
            "Before I suggest recipes: do you have diabetes, high blood pressure or gout/high uric acid, "
            "and any food allergies?")
        return {"reply": q, "route": "clarify", "trace": [{"node": "clarify"}]}

    async def recommender_node(self, state: GraphState) -> dict:
        t0 = time.perf_counter()
        review_txt = ""
        if state.get("review") and not state["review"].get("passed", True):
            review_txt = ("\nYour previous draft was REJECTED by the safety reviewer. Fix these issues and rewrite it:\n- "
                          + "\n- ".join(state["review"]["issues"]) + f"\nPrevious draft:\n{state.get('draft', '')}\n")
        key = "recommender_agent" if self.cfg.get("recommender_tool", True) else "recommender_agent_no_tool"
        if state.get("request", {}).get("intent") == "question":
            # ground diet questions deterministically in the guideline RAG
            t_rag = time.perf_counter()
            passages = await self._tool("search_diet_guidelines", query=state["user_input"])
            review_txt += f"\nGuideline passages retrieved for this question:\n{passages}\n"
            rag_trace = [{"tool": "search_diet_guidelines", "args": {"query": state["user_input"]},
                          "ms": round((time.perf_counter() - t_rag) * 1000), "ok": True, "preview": passages[:160]}]
        else:
            rag_trace = []
        ctx = AgentContext(user_id=state["user_id"], system_prompt=_fill(
            self.prompts[key], profile=_profile_text(state["profile"]),
            request=json.dumps(state.get("request", {}), ensure_ascii=False), review=review_txt))
        out = await self.rec_agent.ainvoke({"messages": self._window(state)}, context=ctx)
        msgs = out["messages"]
        draft = next((m.content for m in reversed(msgs) if isinstance(m, AIMessage) and m.content), "")
        cands, seen = [], set()
        for m in msgs:
            if isinstance(m, ToolMessage) and m.name == "recommend_recipes":
                try:
                    for r in json.loads(tool_text(m.content)).get("recipes", []):
                        if r["recipe_id"] not in seen:
                            seen.add(r["recipe_id"])
                            cands.append(r)
                except (ValueError, KeyError):
                    pass
        return {"draft": draft if isinstance(draft, str) else str(draft), "candidates": cands,
                "trace": [{"node": "recommender_agent", "ms": round((time.perf_counter() - t0) * 1000),
                           "revision": state.get("revision_count", 0), "tools": rag_trace + ctx.trace}]}

    async def reviewer_node(self, state: GraphState) -> dict:
        t0 = time.perf_counter()
        profile, draft, cands = state["profile"], state.get("draft", ""), state.get("candidates", [])
        issues, by_id = [], {r["recipe_id"]: r for r in cands}
        tool_mode = self.cfg.get("recommender_tool", True)
        # --- rule check (deterministic): recommendations must be grounded; cited recipes must be real and allergen-free
        intent = state.get("request", {}).get("intent")
        cited = list(dict.fromkeys(int(x) for x in _CITE.findall(draft)))
        if intent == "recommend" and tool_mode and not cands and not cited:
            issues.append("Recommendations must come from the recommend_recipes tool; call it and cite its recipes.")
        facts = []
        for rid in cited:
            if rid in by_id:
                facts.append(by_id[rid])
                continue
            info = json.loads(await self._tool("get_recipe_details", recipe_id=rid, user_id=state["user_id"]))
            if "error" in info:
                issues.append(f"#{rid} is not a real recipe in the catalog; remove it.")
                continue
            if info.get("contains_user_allergens"):
                issues.append(f"#{rid} ({info['name']}) contains the user's allergen(s) "
                              f"{info['contains_user_allergens']}; remove it.")
            facts.append(info)   # e.g. a recipe recommended in an earlier turn that the user picked
        rule_issues = list(issues)
        # --- LLM check
        llm_verdict = None
        if self.cfg.get("reviewer_llm_check", True):
            if facts:
                facts_txt = json.dumps([{k: r.get(k) for k in ("recipe_id", "name", "total_minutes", "nutrition_per_serving",
                                                              "complies_with", "violations", "ingredients")} for r in facts],
                                       ensure_ascii=False)   # complies_with: true = passes that condition's guideline rules
            elif tool_mode:
                facts_txt = "(the draft cites no recipe ids)"
            else:
                facts_txt = ("(no recommender tool in this configuration: recipes and nutrition values are the assistant's own "
                             "estimates. Do not fail the draft for that; fail only if estimates are presented as exact facts, "
                             "or for safety / medical-advice problems.)")
            prompt = _fill(self.prompts["safety_reviewer"], conditions=", ".join(profile["conditions"]) or "none",
                           allergies=", ".join(profile["allergies"]) or "none", facts=facts_txt, draft=draft)
            try:
                llm_verdict = await self.reviewer.ainvoke(prompt)
                if not llm_verdict.passed:
                    issues += llm_verdict.issues or ["Reviewer flagged the draft as unsafe or unfaithful."]
            except Exception as e:   # reviewer failure must not block the reply; rule check still applies
                logger.warning(f"[reviewer] LLM check failed: {e}")
        review = {"passed": not issues, "issues": issues, "rule_issues": rule_issues,
                  "llm_passed": None if llm_verdict is None else llm_verdict.passed}
        return {"review": review, "trace": [{"node": "safety_reviewer", "ms": round((time.perf_counter() - t0) * 1000),
                                             "review": review, "revision": state.get("revision_count", 0)}]}

    def route_after_review(self, state: GraphState) -> str:
        if state["review"]["passed"]:
            return "finalize"
        return "revise" if state.get("revision_count", 0) < self.max_revisions else "fallback"

    async def revise_node(self, state: GraphState) -> dict:
        return {"revision_count": state.get("revision_count", 0) + 1}

    async def fallback_node(self, state: GraphState) -> dict:
        safe = [r for r in state.get("candidates", []) if not r.get("violations")][:3]
        lines = "\n".join(f"- **{r['name']}** (#{r['recipe_id']}): " + ", ".join(
            f"{k} {v}" for k, v in r["nutrition_per_serving"].items() if k in ("kcal", "carbs_g", "sugar_g", "sodium_mg"))
            for r in safe)
        text = _fill(self.prompts["fallback_reply"],
                     recipes=("These recipes passed all automatic safety checks:\n" + lines) if safe else "")
        return {"reply": text, "route": "fallback", "trace": [{"node": "fallback"}]}

    async def finalize_node(self, state: GraphState) -> dict:
        route = "reviewed" if self.cfg.get("safety_reviewer", True) else "unreviewed"
        return {"reply": state.get("draft", ""), "route": route, "trace": [{"node": "finalize"}]}

    # ------------------------------------------------------------ wiring
    def _build(self):
        g = StateGraph(GraphState)
        for name, fn in [("load_profile", self.load_profile), ("profile_agent", self.profile_node),
                         ("clarify", self.clarify_node), ("recommender_agent", self.recommender_node),
                         ("safety_reviewer", self.reviewer_node), ("revise", self.revise_node),
                         ("fallback", self.fallback_node), ("finalize", self.finalize_node)]:
            g.add_node(name, fn)
        g.add_edge(START, "load_profile")
        g.add_edge("load_profile", "profile_agent")
        g.add_conditional_edges("profile_agent", self.route_after_profile,
                                {"clarify": "clarify", "recommender": "recommender_agent"})
        g.add_edge("clarify", END)
        if self.cfg.get("safety_reviewer", True):
            g.add_edge("recommender_agent", "safety_reviewer")
            g.add_conditional_edges("safety_reviewer", self.route_after_review,
                                    {"finalize": "finalize", "revise": "revise", "fallback": "fallback"})
            g.add_edge("revise", "recommender_agent")
        else:
            g.add_edge("recommender_agent", "finalize")
        g.add_edge("fallback", END)
        g.add_edge("finalize", END)
        return g.compile()
