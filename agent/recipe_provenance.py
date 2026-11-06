"""Deterministic recipe provenance gate for LangGraph outputs.

For recommendation turns, existence in Elasticsearch is NOT sufficient:
every cited recipe must come from the current recommender-tool result.
For detail turns, only recipes already shown in previous assistant replies
may be revisited. This avoids hallucinated but valid-looking catalog IDs.
"""
from __future__ import annotations

import re

CITE = re.compile(r"#(\d{2,7})\b")


def cited_recipe_ids(text: str) -> list[int]:
    return list(dict.fromkeys(int(x) for x in CITE.findall(text or "")))


def recommendation_provenance_issues(draft: str, candidates: list[dict]) -> list[str]:
    cited = cited_recipe_ids(draft)
    allowed = {int(r["recipe_id"]) for r in candidates}
    issues = []
    if not candidates:
        issues.append("No verified recipes were returned by recommend_recipes; do not invent recommendations.")
    if not cited:
        issues.append("Recommendation must cite at least one recipe ID returned by recommend_recipes.")
    for rid in cited:
        if rid not in allowed:
            issues.append(f"#{rid} was not returned by recommend_recipes in this turn; remove it.")
    return issues


def detail_provenance_issues(user_text: str, draft: str, history: list) -> list[str]:
    """Allow details only for recipe IDs already exposed by the assistant."""
    seen: set[int] = set()
    for msg in history or []:
        if getattr(msg, "type", "") == "ai":
            seen.update(cited_recipe_ids(str(getattr(msg, "content", ""))))
    requested = set(cited_recipe_ids(user_text))
    cited = set(cited_recipe_ids(draft))
    return [f"#{rid} was not previously recommended to this user; ask for a valid recipe ID."
            for rid in sorted(requested | cited) if rid not in seen]
