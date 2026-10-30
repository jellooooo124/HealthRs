from prototype_elasticsearch.ranking import rank_candidates, safe_for_profile
from prototype_elasticsearch.recommender import HybridRecommender

PROFILE = {"conditions": ["diabetes"], "allergies": ["peanut"], "taste_prefs": ["chicken"], "confirmed": True}


def doc(i, name, category="Main", allergens=(), compliant=("diabetes",), complete=True, bm25=1, pop=5):
    return {"recipe_id": i, "name": name, "description": name, "category": category,
            "ingredients": [name], "keywords": [], "nutrition_complete": complete,
            "compliant_conditions": list(compliant), "allergens": list(allergens),
            "_bm25_score": bm25, "popularity": pop}


def test_safety_fail_closed_even_with_high_rank_score():
    bad_allergy = doc(1, "chicken peanut", allergens=("peanut",), bm25=1000)
    bad_condition = doc(2, "chicken sweet", compliant=(), bm25=1000)
    unknown = doc(3, "chicken", complete=False, bm25=1000)
    good = doc(4, "chicken salad", bm25=1)
    ranked = rank_candidates([bad_allergy, bad_condition, unknown, good], PROFILE, {}, k=10)
    assert [r["recipe_id"] for r in ranked] == [4]
    assert all(safe_for_profile(r, PROFILE) for r in ranked)


def test_personalization_shifts_rank_to_matching_preference():
    docs = [doc(1, "plain barley", pop=10, bm25=2), doc(2, "chicken bowl", pop=10, bm25=2)]
    result = rank_candidates(docs, PROFILE, {}, k=2)
    assert result[0]["recipe_id"] == 2
    assert "chicken" in result[0]["reasons"][0]


def test_feedback_dislike_excluded_and_like_boosts_category():
    p = {"conditions": [], "allergies": [], "taste_prefs": [], "confirmed": True}
    docs = [doc(1, "liked", "Soup"), doc(2, "same category", "Soup"), doc(3, "another category", "Pasta")]
    fb = {1: {"signal": "like"}, 3: {"signal": "dislike"}}
    ranked = rank_candidates(docs, p, fb, k=3, liked_docs=[docs[0]])
    assert 3 not in [x["recipe_id"] for x in ranked]
    assert ranked[1]["recipe_id"] == 2 or ranked[0]["recipe_id"] == 2
    assert next(x for x in ranked if x["recipe_id"] == 2)["score_components"]["taste"] > 0


def test_unconfirmed_profile_blocks_es_calls():
    class Store:
        def get(self, x): return {"confirmed": False}
    class ES:
        def search(self, **kwargs): raise AssertionError("ES should not be called")
    result = HybridRecommender(ES(), Store(), use_bpr=False).recommend("cold")
    assert result["status"] == "needs_profile_confirmation"
    assert not result["recipes"]


def test_recommender_uses_trusted_profile_and_feedback():
    class Store:
        def get(self, x): return PROFILE
        def latest_feedback(self, x): return {1: {"signal": "dislike"}}
    class ES:
        def search(self, **kwargs):
            self.query = kwargs["query"]
            return {"hits": {"hits": [{"_source": doc(2, "chicken salad"), "_score": 2}]}}
    es = ES()
    result = HybridRecommender(es, Store(), use_bpr=False).recommend("u", "chicken", k=3)
    assert result["recipes"][0]["recipe_id"] == 2
    assert {"terms": {"recipe_id": [1]}} in es.query["bool"]["must_not"]
    assert {"terms": {"allergens": ["peanut"]}} in es.query["bool"]["must_not"]
