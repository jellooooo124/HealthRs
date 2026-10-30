"""Search -> profile-enforced filtering -> optional original BPR -> ranking -> audit."""
from __future__ import annotations

import os
from typing import Any
from .bridge import optional_bpr_scores
from .query import build_query, retrieve
from .ranking import rank_candidates

INDEX = os.getenv("HEALTHREC_ES_INDEX", "healthrec-recipes-v1")


class HybridRecommender:
    def __init__(self, es: Any, store: Any, index: str = INDEX, use_bpr: bool = True):
        self.es, self.store, self.index, self.use_bpr = es, store, index, use_bpr

    def recommend(self, user_id: str, text: str = "", constraints: dict | None = None,
                  k: int = 5, candidate_size: int = 100) -> dict:
        # Critical: conditions/allergies are READ FROM THE TRUSTED PROFILE, never request payload.
        profile = self.store.get(str(user_id)) or self.store.create(str(user_id))
        if not profile.get("confirmed"):
            return {"status": "needs_profile_confirmation", "recipes": [], "user_id": user_id,
                    "message": "Confirm health conditions and allergies before receiving recommendations."}
        feedback = self.store.latest_feedback(user_id)
        excluded = {int(rid) for rid, f in feedback.items() if f.get("signal") == "dislike"}
        dsl = build_query(text, profile["conditions"], profile["allergies"], constraints, excluded)
        docs = retrieve(self.es, self.index, dsl, candidate_size)
        model_scores = optional_bpr_scores(profile, docs, enabled=self.use_bpr)
        liked_ids = {int(rid) for rid, f in feedback.items() if f.get("signal") in {"like", "cooked"}}
        liked_docs = []
        if liked_ids:
            raw = self.es.mget(index=self.index, ids=[str(rid) for rid in sorted(liked_ids)[:100]])
            liked_docs = [doc["_source"] for doc in raw.get("docs", []) if doc.get("found")]
        recipes = rank_candidates(docs, profile, feedback, k=k, bpr_scores=model_scores,
                                  liked_docs=liked_docs)
        return {"status": "ok", "user_id": user_id, "query": text, "n_candidates": len(docs),
                "conditions": profile["conditions"], "allergies": profile["allergies"],
                "bpr_used": bool(model_scores), "recipes": recipes,
                "note": "Demo research prototype. Rules are heuristic, not clinical clearance. Verify ingredients independently."}
