"""Deterministic, inspectable personalized ranking; health/allergy checks fail closed."""
from __future__ import annotations

import math
from typing import Any


def _scale(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    return [(v - lo) / (hi - lo) for v in values] if hi > lo else [0.0] * len(values)


def safe_for_profile(doc: dict, profile: dict) -> bool:
    """Second safety check AFTER Elasticsearch; unknown labels are not assumed safe."""
    if not doc.get("nutrition_complete", False) or not doc.get("ingredients"):
        return False
    if not set(profile.get("conditions") or []).issubset(set(doc.get("compliant_conditions") or [])):
        return False
    if set(profile.get("allergies") or []) & set(doc.get("allergens") or []):
        return False
    return True


def _taste_match(doc: dict, prefs: list[str]) -> tuple[float, list[str]]:
    haystack = " ".join(str(doc.get(k, "")) for k in ("name", "category", "description"))
    haystack += " " + " ".join(doc.get("ingredients") or []) + " " + " ".join(doc.get("keywords") or [])
    haystack = haystack.casefold()
    found = [p for p in prefs if p and p.casefold() in haystack]
    return min(1.0, len(found) / 2), found


def rank_candidates(
    docs: list[dict], profile: dict, latest_feedback: dict[int, dict],
    k: int = 5, bpr_scores: dict[int, float] | None = None, max_per_category: int = 2,
    liked_docs: list[dict] | None = None,
) -> list[dict]:
    if not 1 <= k <= 50:
        raise ValueError("k must be between 1 and 50")
    feedback = {int(key): value for key, value in latest_feedback.items()}
    blocked = {rid for rid, f in feedback.items() if f.get("signal") == "dislike"}
    candidates = [d for d in docs if int(d["recipe_id"]) not in blocked and safe_for_profile(d, profile)]
    if not candidates:
        return []
    positives = {rid for rid, f in feedback.items() if f.get("signal") in {"like", "cooked"}}
    likes = [d for d in (liked_docs if liked_docs is not None else docs) if int(d["recipe_id"]) in positives]
    liked_categories = {d.get("category") for d in likes}
    prefs = list(profile.get("taste_prefs") or [])
    bm25 = _scale([float(d.get("_bm25_score") or 0) for d in candidates])
    popularity = _scale([math.log1p(max(0.0, float(d.get("popularity") or 0))) for d in candidates])
    taste = []
    matches = []
    for d in candidates:
        value, names = _taste_match(d, prefs)
        same_category = bool(liked_categories and d.get("category") in liked_categories)
        taste.append(min(1.0, value * 0.8 + (0.2 if same_category else 0.0)))
        matches.append(names)
    has_bpr = bool(bpr_scores) and any(int(d["recipe_id"]) in bpr_scores for d in candidates)
    bpr = _scale([float((bpr_scores or {}).get(int(d["recipe_id"]), 0.0)) for d in candidates]) if has_bpr else [0.0] * len(candidates)
    weights = {"bm25": 0.25, "bpr": 0.40, "taste": 0.25, "popularity": 0.10} if has_bpr else \
              {"bm25": 0.40, "bpr": 0.0, "taste": 0.40, "popularity": 0.20}
    ranked = []
    for i, d in enumerate(candidates):
        score_parts = {"bm25": round(bm25[i], 4), "bpr": round(bpr[i], 4),
                       "taste": round(taste[i], 4), "popularity": round(popularity[i], 4)}
        score = (weights["bm25"] * bm25[i] + weights["bpr"] * bpr[i]
                 + weights["taste"] * taste[i] + weights["popularity"] * popularity[i])
        reasons = []
        if matches[i]:
            reasons.append("符合偏好：" + "、".join(matches[i][:3]))
        if profile.get("conditions"):
            reasons.append("通過既有規則的條件標籤：" + "、".join(profile["conditions"]))
        if profile.get("allergies"):
            reasons.append("已依資料標籤排除指定過敏原")
        if has_bpr:
            reasons.append("包含原 HealthRec BPR-MF 排序訊號")
        if not reasons:
            reasons.append("依搜尋相關性及食譜受歡迎程度排序")
        ranked.append({**d, "score": round(score, 5), "score_components": score_parts,
                       "score_weights": weights, "reasons": reasons})
    ranked.sort(key=lambda d: (-d["score"], int(d["recipe_id"])))
    selected, rest, counts = [], [], {}
    for doc in ranked:
        cat = doc.get("category") or "Other"
        if counts.get(cat, 0) < max_per_category and len(selected) < k:
            selected.append(doc)
            counts[cat] = counts.get(cat, 0) + 1
        else:
            rest.append(doc)
    return (selected + rest)[:k]
