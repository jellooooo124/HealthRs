"""
HealthRec MCP server: exposes the recommender, profile store and guideline RAG as MCP tools.

Runs as a long-lived HTTP service (streamable-http) so the recommender model is loaded once:
    python -m agent.mcp_server            # http://127.0.0.1:8765/mcp  (host/port in configs/agent_config.yaml)
agent/mcp_client.py starts it automatically when it is not running.

`user_id` arguments are overwritten by the agent middleware with the logged-in user, so the LLM
can never read or modify another user's profile.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from agent.recommender_service import RecommenderService  # noqa: E402
from src.config import get_agent_config  # noqa: E402
from src.utils.logger import get_logger, log_tool_call  # noqa: E402

logger = get_logger("healthrec.mcp_server")
_cfg = get_agent_config()["mcp"]
_port = int(os.environ.get("HEALTHREC_MCP_PORT", _cfg["port"]))
if os.environ.get("HEALTHREC_EVAL_MODE") == "1":   # evaluation: no extra LLM calls inside RAG (planned token saving)
    get_agent_config()["rag"].update(query_rewrite=False, self_reflection=False)
mcp = FastMCP("HealthRec", host=_cfg["host"], port=_port, stateless_http=True, json_response=True,
              log_level="WARNING")


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


@mcp.tool()
@log_tool_call
def recommend_recipes(user_id: str, meal_type: str | None = None, max_minutes: int | None = None,
                      max_calories: float | None = None,
                      categories: list[str] | None = None, include_ingredients: list[str] | None = None,
                      exclude_ingredients: list[str] | None = None, exclude_recipe_ids: list[int] | None = None,
                      k: int = 5) -> str:
    """
    Personalised, health-aware recipe recommendations from the trained two-stage recommender
    (retrieval: taste channel + healthy-popular channel -> ranking: BPR-MF with nutrition/condition
    features -> safety rerank). The user's conditions and allergies are read from the database and
    always applied; allergens are never returned. Returns JSON with up to k recipes and per-serving nutrition.

    Args:
        user_id: filled in by the system.
        meal_type: one of "breakfast", "main" (lunch/dinner main dish), "side", "dessert", "snack", "drink".
        max_minutes: maximum total cooking time in minutes.
        max_calories: maximum kcal per serving.
        categories: optional Food.com categories to keep, e.g. ["Chicken", "Pork", "Soups"] (prefer meal_type).
        include_ingredients: ingredients that must appear, e.g. ["chicken"].
        exclude_ingredients: ingredients to avoid (dislikes), e.g. ["pork", "mushrooms"].
        exclude_recipe_ids: recipes to leave out (already shown or rejected).
        k: number of recipes (1-10, default 5).
    """
    constraints = {"meal_type": meal_type, "max_minutes": max_minutes, "max_calories": max_calories, "categories": categories,
                   "include_ingredients": include_ingredients, "exclude_ingredients": exclude_ingredients}
    return _dump(RecommenderService.get().recommend(user_id, constraints, k, exclude_recipe_ids))


@mcp.tool()
@log_tool_call
def get_recipe_details(recipe_id: int, user_id: str = "") -> str:
    """
    Ingredients, cooking steps, servings and per-serving nutrition of one recipe, plus whether it
    complies with the user's conditions and contains any of their allergens.

    Args:
        recipe_id: Food.com recipe id (the number cited as #id).
        user_id: filled in by the system.
    """
    return _dump(RecommenderService.get().details(recipe_id, user_id or None))


@mcp.tool()
@log_tool_call
def get_user_profile(user_id: str) -> str:
    """
    The stored health profile: conditions, food allergies, goals, taste preferences, whether the
    profile has been confirmed, and recent recipe feedback.

    Args:
        user_id: filled in by the system.
    """
    svc = RecommenderService.get()
    p = svc.store.get(user_id) or svc.store.create(user_id)
    p["recent_feedback"] = svc.store.feedback(user_id)[-10:]
    return _dump(p)


@mcp.tool()
@log_tool_call
def update_user_profile(user_id: str, conditions: list[str] | None = None, allergies: list[str] | None = None,
                        taste_prefs: list[str] | None = None, confirmed: bool | None = None) -> str:
    """
    Replace profile fields (omit a field to keep it). Give COMPLETE lists, not just the change.

    Args:
        user_id: filled in by the system.
        conditions: subset of ["diabetes", "hypertension", "hyperuricemia"]; [] = none.
        allergies: subset of ["shellfish", "milk", "peanut", "tree_nut", "fish", "egg", "wheat", "soy", "sesame"]; [] = none.
        taste_prefs: short preference phrases, e.g. ["likes Asian food", "no desserts"].
        confirmed: true once the user has told us their conditions and allergies.
    """
    p = RecommenderService.get().store.update(user_id, conditions, allergies, taste_prefs, confirmed)
    return _dump({"status": "updated", "profile": p})


@mcp.tool()
@log_tool_call
def record_feedback(user_id: str, recipe_id: int, signal: str, reason: str = "") -> str:
    """
    Store the user's reaction to a recommended recipe. "dislike" removes the recipe from future
    recommendations.

    Args:
        user_id: filled in by the system.
        recipe_id: the recipe's id (#id).
        signal: one of "like", "dislike", "cooked".
        reason: the user's reason in a few words (e.g., "too bland").
    """
    if signal not in ("like", "dislike", "cooked"):
        return _dump({"status": "error", "message": "signal must be like, dislike or cooked"})
    RecommenderService.get().store.add_feedback(user_id, recipe_id, signal, reason)
    return _dump({"status": "recorded", "recipe_id": recipe_id, "signal": signal})


@mcp.tool()
@log_tool_call
def search_diet_guidelines(query: str) -> str:
    """
    Search summaries of public diet guidelines (WHO healthy diet, ADA diabetes nutrition, ACC/AHA
    hypertension/DASH, ACR 2020 gout, food allergens and labels, and this system's compliance rules).
    Returns relevant passages with their source file.

    Args:
        query: the diet or nutrition question.
    """
    from rag.agentic_rag import search_guidelines
    return search_guidelines(query)


@mcp.tool()
@log_tool_call
def ingest_guideline_document(path: str) -> str:
    """
    Admin tool used by the demo UI (not given to the agents): index an uploaded guideline document
    (.md/.txt/.pdf under data/guidelines/) into the knowledge base. Runs inside this server process so the
    vector store it searches stays consistent.

    Args:
        path: file path under data/guidelines/.
    """
    from rag.vector_stores import GUIDELINE_DIR, ingest_document
    full = os.path.realpath(path)
    if not full.startswith(os.path.realpath(GUIDELINE_DIR)):
        return _dump({"status": "error", "message": "only files under data/guidelines/ can be indexed"})
    return _dump({"status": "ok", "chunks_added": ingest_document(full)})


if __name__ == "__main__":
    RecommenderService.get()   # load models before accepting requests
    logger.info(f"HealthRec MCP server on http://{_cfg['host']}:{_port}/mcp")
    mcp.run(transport="streamable-http")
