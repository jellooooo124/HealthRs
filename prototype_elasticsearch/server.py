"""Minimal API using the existing HealthRec profile store and a live ES index."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .profiles import make_store
from .recommender import HybridRecommender
from .settings import es_client

app = FastAPI(title="HealthRec Elasticsearch + Personalized Recommendation", version="0.1.0")


class Constraints(BaseModel):
    max_minutes: float | None = Field(None, ge=0)
    max_calories: float | None = Field(None, ge=0)
    category: str | None = None
    include_ingredients: list[str] = Field(default_factory=list, max_length=10)
    exclude_ingredients: list[str] = Field(default_factory=list, max_length=10)


class RecommendRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100, pattern=r"^[\w.-]+$")
    query: str = Field(default="", max_length=500)
    constraints: Constraints = Field(default_factory=Constraints)
    k: int = Field(default=5, ge=1, le=20)
    candidate_size: int = Field(default=100, ge=1, le=500)


class ProfileUpdate(BaseModel):
    conditions: list[Literal["diabetes", "hypertension", "hyperuricemia"]]
    allergies: list[Literal["shellfish", "milk", "peanut", "tree_nut", "fish", "egg", "wheat", "soy", "sesame"]]
    taste_prefs: list[str] = Field(default_factory=list, max_length=10)
    confirmed: bool = True


class FeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100, pattern=r"^[\w.-]+$")
    recipe_id: int = Field(gt=0)
    signal: Literal["like", "dislike", "cooked", "clear"]
    reason: str = Field(default="", max_length=300)


@lru_cache

def store():
    return make_store()


@lru_cache

def engine():
    return HybridRecommender(es_client(), store(), use_bpr=os.getenv("HEALTHREC_USE_BPR", "1") == "1")


@app.get("/health")
def health():
    try:
        es = engine().es
        return {"ok": es.ping(), "elasticsearch": es.info().get("version", {}).get("number"),
                "index": engine().index, "count": es.count(index=engine().index)["count"]}
    except Exception as e:
        raise HTTPException(503, "Elasticsearch/index unavailable. Start Docker and index recipes.") from e


@app.get("/profiles/{user_id}")
def get_profile(user_id: str):
    if not user_id or len(user_id) > 100:
        raise HTTPException(400, "Invalid user id")
    return store().get(user_id) or store().create(user_id)


@app.put("/profiles/{user_id}")
def update_profile(user_id: str, req: ProfileUpdate):
    if not user_id or len(user_id) > 100:
        raise HTTPException(400, "Invalid user id")
    return store().update(user_id, req.conditions, req.allergies, req.taste_prefs, req.confirmed)


@app.post("/recommendations")
def recommendations(req: RecommendRequest):
    try:
        return engine().recommend(req.user_id, req.query, req.constraints.model_dump(exclude_none=True),
                                  req.k, req.candidate_size)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(503, "Search is unavailable; inspect server logs.") from e


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    # Require existing confirmed profile to avoid introducing arbitrary users into the profile store.
    profile = store().get(req.user_id)
    if not profile or not profile["confirmed"]:
        raise HTTPException(409, "Confirm user profile first")
    # A recipe must exist in the index; rejects fabricated recipe ids.
    if not engine().es.exists(index=engine().index, id=str(req.recipe_id)):
        raise HTTPException(404, "Unknown recipe")
    store().add_feedback(req.user_id, req.recipe_id, req.signal, req.reason)
    return {"ok": True, "recipe_id": req.recipe_id, "signal": req.signal}
