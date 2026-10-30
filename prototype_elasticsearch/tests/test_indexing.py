def test_to_document_reuses_existing_health_labels():
    from prototype_elasticsearch.schema import to_document
    row = {"RecipeId": 10, "Name": "Baked potato", "RecipeCategory": "Side",
           "RecipeIngredientParts": ["potato", "butter"], "RecipeInstructions": ["Bake"],
           "Calories": 200, "SodiumContent": 90, "CarbohydrateContent": 30,
           "SugarContent": 1, "SaturatedFatContent": 1,
           "ok_diabetes": True, "ok_hypertension": True, "contains_milk": True}
    x = to_document(row, ["diabetes", "hypertension"], ["milk", "peanut"])
    assert x["nutrition_complete"] is True
    assert set(x["compliant_conditions"]) == {"diabetes", "hypertension"}
    assert x["allergens"] == ["milk"]
