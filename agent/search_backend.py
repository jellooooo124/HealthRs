"""Optional Elasticsearch backend for the existing LangGraph/MCP recipe tool.

Set HEALTHREC_RETRIEVAL_BACKEND=elasticsearch before starting the MCP server.
The default remains the original trained recommender. Search errors never
silently fall back to unsafe or unfiltered suggestions.
"""
from __future__ import annotations

import os
from functools import lru_cache


def backend_mode() -> str:
    mode = os.environ.get("HEALTHREC_RETRIEVAL_BACKEND", "legacy").strip().lower()
    if mode not in {"legacy", "elasticsearch"}:
        raise ValueError("HEALTHREC_RETRIEVAL_BACKEND must be legacy or elasticsearch")
    return mode


@lru_cache(maxsize=1)
def es_store():
    from prototype_elasticsearch.profiles import make_store
    return make_store()


@lru_cache(maxsize=1)
def es_engine():
    from prototype_elasticsearch.recommender import HybridRecommender
    from prototype_elasticsearch.settings import es_client
    return HybridRecommender(
        es_client(), es_store(),
        use_bpr=os.environ.get("HEALTHREC_USE_BPR", "1") == "1",
    )


def _compact(doc: dict, profile: dict, details: bool = False) -> dict:
    nutr = doc.get("nutrition") or {}
    nutrition = {("kcal" if key == "calories" else key): value
                 for key, value in nutr.items() if value is not None}
    conditions = profile.get("conditions") or []
    result = {
        "recipe_id": int(doc["recipe_id"]),
        "name": doc.get("name", ""),
        "category": doc.get("category"),
        "total_minutes": doc.get("total_minutes"),
        "nutrition_per_serving": nutrition,
        "complies_with": {c: c in (doc.get("compliant_conditions") or []) for c in conditions},
        "violations": {c: "Recipe does not meet configured rules"
                       for c in conditions if c not in (doc.get("compliant_conditions") or [])},
        "ingredients": list(doc.get("ingredients") or [])[:30],
        # The safety reviewer must see the explicit per-user allergen-label
        # check, rather than guessing risk from unrelated ingredient names.
        "contains_user_allergens": sorted(
            set(profile.get("allergies") or []) & set(doc.get("allergens") or [])),
    }
    if details:
        result["steps"] = list(doc.get("steps") or [])[:12]
        result["servings"] = doc.get("servings")
        result["image_url"] = doc.get("image_url")
        result["contains_user_allergens"] = sorted(
            set(profile.get("allergies") or []) & set(doc.get("allergens") or []))
    else:
        result["retrieved_by"] = "elasticsearch_bm25_hybrid"
        result["score"] = doc.get("score")
        result["score_components"] = doc.get("score_components", {})
        result["reasons"] = doc.get("reasons", [])
    return result


def es_recommend(user_id: str, constraints: dict | None = None, k: int = 5,
                 exclude_recipe_ids: list[int] | None = None, query: str = "") -> dict:
    store = es_store()
    profile = store.get(user_id) or store.create(user_id)
    c = {key: value for key, value in (constraints or {}).items() if value not in (None, [], "")}
    text = query.strip() or " ".join(c.get("include_ingredients") or [])
    raw = es_engine().recommend(user_id, text, constraints=c, k=max(1, min(int(k), 10)),
                                exclude_recipe_ids=exclude_recipe_ids)
    return {
        "user": {"user_id": user_id, "conditions": profile.get("conditions", []),
                 "allergies": profile.get("allergies", []),
                 "history": "rating history (warm user)" if profile.get("source") == "foodcom_warm"
                            else "no history (cold start)"},
        "applied_constraints": c,
        "n_candidates": raw.get("n_candidates", 0),
        "backend": "elasticsearch",
        "bpr_used": raw.get("bpr_used", False),
        "recipes": [_compact(doc, profile) for doc in raw.get("recipes", [])],
        "note": (raw.get("message") if raw.get("status") != "ok" else
                 "Research prototype. Health/allergen labels are heuristic, not clinical clearance."),
    }


def es_details(recipe_id: int, user_id: str | None = None) -> dict:
    engine = es_engine()
    if not engine.es.exists(index=engine.index, id=str(recipe_id)):
        return {"error": f"recipe {recipe_id} not in the Elasticsearch index"}
    doc = engine.es.get(index=engine.index, id=str(recipe_id))["_source"]
    profile = es_store().get(user_id) if user_id else None
    return _compact(doc, profile or {}, details=True)
