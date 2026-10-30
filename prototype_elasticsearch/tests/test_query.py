from prototype_elasticsearch.query import build_query, retrieve


def test_search_enforces_every_condition_and_allergen_exclusion():
    q = build_query("quick dinner", ["hypertension", "diabetes"], ["peanut", "egg"],
                    {"max_minutes": 30, "max_calories": 500, "exclude_ingredients": ["bacon"]}, {42})
    inner = q["bool"]
    filters = inner["filter"]
    assert {"term": {"compliant_conditions": "diabetes"}} in filters
    assert {"term": {"compliant_conditions": "hypertension"}} in filters
    assert {"term": {"nutrition_complete": True}} in filters
    assert {"terms": {"allergens": ["egg", "peanut"]}} in inner["must_not"]
    assert {"terms": {"recipe_id": [42]}} in inner["must_not"]
    assert {"range": {"total_minutes": {"lte": 30}}} in filters
    assert {"range": {"nutrition.calories": {"lte": 500}}} in filters
    assert {"match_phrase": {"ingredient_text": "bacon"}} in inner["must_not"]


def test_empty_query_sorts_by_popularity():
    class ES:
        def search(self, **kw):
            self.kw = kw
            return {"hits": {"hits": [{"_score": None, "_source": {"recipe_id": 1}}]}}
    es = ES()
    docs = retrieve(es, "demo", build_query("", [], []), 10)
    assert es.kw["sort"][0] == {"popularity": "desc"}
    assert docs[0]["_bm25_score"] == 0
