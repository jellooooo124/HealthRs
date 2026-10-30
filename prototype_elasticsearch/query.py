"""Elasticsearch candidate retrieval. Safety filters are mandatory and independent of BM25."""
from __future__ import annotations

from typing import Any


def build_query(
    query: str,
    conditions: list[str],
    allergies: list[str],
    constraints: dict[str, Any] | None = None,
    excluded_ids: set[int] | None = None,
) -> dict[str, Any]:
    c = constraints or {}
    filters: list[dict] = [{"term": {"nutrition_complete": True}}]
    must_not: list[dict] = []
    for condition in sorted(set(conditions)):
        filters.append({"term": {"compliant_conditions": condition}})
    if allergies:
        must_not.append({"terms": {"allergens": sorted(set(allergies))}})
    if excluded_ids:
        must_not.append({"terms": {"recipe_id": sorted(excluded_ids)}})
    for field in ("max_minutes", "max_calories"):
        if c.get(field) is not None:
            value = float(c[field])
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
            target = "total_minutes" if field == "max_minutes" else "nutrition.calories"
            filters.append({"range": {target: {"lte": value}}})
    if c.get("category"):
        filters.append({"term": {"category": str(c["category"])}})
    included = [p.strip() for p in c.get("include_ingredients", []) if p and p.strip()]
    if included:
        filters.append({"bool": {"should": [{"match_phrase": {"ingredient_text": p}} for p in included],
                                    "minimum_should_match": 1}})
    excluded = [p.strip() for p in c.get("exclude_ingredients", []) if p and p.strip()]
    for phrase in excluded:
        must_not.append({"match_phrase": {"ingredient_text": phrase}})
    searchable = {"multi_match": {"query": query.strip(),
                                  "fields": ["name^4", "ingredients^2", "description^1.5", "keywords^2"],
                                  "type": "best_fields"}} if query.strip() else {"match_all": {}}
    return {"bool": {"must": [searchable], "filter": filters, "must_not": must_not}}


def retrieve(client: Any, index: str, query: dict, candidate_size: int = 100) -> list[dict]:
    if not 1 <= candidate_size <= 500:
        raise ValueError("candidate_size must be between 1 and 500")
    # In the no-text case sort by popularity rather than arbitrary ES document order.
    is_match_all = "match_all" in query["bool"]["must"][0]
    opts = {"sort": [{"popularity": "desc"}, {"recipe_id": "asc"}]} if is_match_all else {}
    result = client.search(index=index, query=query, size=candidate_size, **opts)
    hits = result["hits"]["hits"]
    return [{**h["_source"], "_bm25_score": float(h.get("_score") or 0)} for h in hits]
