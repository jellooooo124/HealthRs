"""Unit contract tests for the LangGraph -> MCP -> Elasticsearch adapter."""
import os

from agent import search_backend


def test_default_mode_keeps_original_recommender(monkeypatch):
    monkeypatch.delenv("HEALTHREC_RETRIEVAL_BACKEND", raising=False)
    assert search_backend.backend_mode() == "legacy"


def test_es_mode_requires_explicit_switch(monkeypatch):
    monkeypatch.setenv("HEALTHREC_RETRIEVAL_BACKEND", "elasticsearch")
    assert search_backend.backend_mode() == "elasticsearch"


def test_compact_es_results_match_existing_agent_contract():
    doc = {
        "recipe_id": 12, "name": "Chicken soup", "category": "Soups",
        "total_minutes": 30, "nutrition": {"calories": 220, "sodium_mg": 300},
        "ingredients": ["chicken", "carrot"],
        "compliant_conditions": ["hypertension"], "allergens": [],
        "score": 0.91, "score_components": {"bm25": 1.0},
    }
    profile = {"conditions": ["hypertension"], "allergies": ["peanut"]}
    result = search_backend._compact(doc, profile)
    assert result["recipe_id"] == 12
    assert result["nutrition_per_serving"]["kcal"] == 220
    assert result["complies_with"] == {"hypertension": True}
    assert result["violations"] == {}
    assert result["retrieved_by"] == "elasticsearch_bm25_hybrid"
    assert search_backend._compact(doc, profile, details=True)["contains_user_allergens"] == []


def test_es_adapter_preserves_profile_confirmation_and_per_turn_exclusions(monkeypatch):
    user = {"user_id": "u", "confirmed": True, "conditions": [], "allergies": [], "source": "demo_new"}
    class Store:
        def get(self, uid): return user
    class Engine:
        def recommend(self, user_id, text, **kwargs):
            assert user_id == "u"
            assert text == "chicken soup"
            assert kwargs["exclude_recipe_ids"] == [42]
            assert kwargs["constraints"]["categories"] == ["Soups"]
            return {"status": "ok", "recipes": [], "n_candidates": 0, "bpr_used": False}
    monkeypatch.setattr(search_backend, "es_store", lambda: Store())
    monkeypatch.setattr(search_backend, "es_engine", lambda: Engine())
    result = search_backend.es_recommend("u", {"categories": ["Soups"]}, 5, [42], "chicken soup")
    assert result["backend"] == "elasticsearch"
    assert result["recipes"] == []
    assert result["bpr_used"] is False
