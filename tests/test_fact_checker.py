"""Regression tests for factual claim verification and uncertain LLM critiques."""
from agent.fact_checker import factual_claim_issues, grounded_llm_issues


FACTS = [
    {"recipe_id": 20755, "name": "Chicken Noodle Soup", "total_minutes": 40.0,
     "nutrition_per_serving": {"sodium_mg": 1085.2, "sugar_g": 3.6}},
]
NO_RESTRICTIONS = {"conditions": [], "allergies": []}


def test_minutes_40_equals_40_point_zero():
    assert factual_claim_issues(
        "Chicken Noodle Soup (#20755) takes 40 minutes.", FACTS, NO_RESTRICTIONS
    ) == []


def test_unfounded_low_sodium_claim_is_blocking():
    assert any("low-sodium" in i for i in factual_claim_issues(
        "Chicken Noodle Soup (#20755) is low in sodium.", FACTS, NO_RESTRICTIONS
    ))


def test_nonzero_sugar_cannot_be_sugar_free():
    assert any("sugar-free" in i for i in factual_claim_issues(
        "Chicken Noodle Soup (#20755) is sugar-free.", FACTS, NO_RESTRICTIONS
    ))


def test_unknown_nutrients_cannot_support_zero_claims():
    assert factual_claim_issues(
        "This recipe is sodium-free.", [{"nutrition_per_serving": {}}], NO_RESTRICTIONS,
    )


def test_rejects_blanket_allergen_free_even_without_reported_allergies():
    assert any("allergen-free" in i for i in factual_claim_issues(
        "Both dishes are allergen-free.", FACTS, NO_RESTRICTIONS
    ))


def test_unverified_llm_criticism_does_not_block_unrestricted_profile():
    critique = ["40 minutes conflicts with the tool's 40.0 minutes."]
    blocking, unverified = grounded_llm_issues(
        critique, "40 minutes", '"total_minutes": 40.0',
        "Chicken Noodle Soup takes 40 minutes.",
        '[{"total_minutes": 40.0}]',
        NO_RESTRICTIONS,
    )
    assert blocking == [] and unverified == critique


def test_llm_warning_fails_closed_with_known_allergy():
    critique = ["This recipe may contain the user's peanut allergen."]
    blocking, unverified = grounded_llm_issues(
        critique, "", "", "Chicken soup", "[]",
        {"conditions": [], "allergies": ["peanut"]},
    )
    assert blocking == critique and unverified == []


def test_medical_treatment_advice_blocks_without_conditions():
    critique = ["Medication change advice is unsafe."]
    blocking, unverified = grounded_llm_issues(
        critique, "", "", "Stop taking insulin today.", "[]",
        NO_RESTRICTIONS,
    )
    assert blocking == critique and unverified == []


def test_reviewer_hallucinated_extra_soy_sauce_is_recorded_not_trusted():
    """Regression from real Qwen2.5:3b: a draft invented by the reviewer itself."""
    actual_draft = "Try **Bourbon Chicken** (#45809)."
    critique = ["The draft tells the user to add extra soy sauce."]
    fabricated = "Try **Bourbon Chicken** (#45809). Add extra soy sauce to enhance the flavor."
    blocking, unverified = grounded_llm_issues(
        critique, fabricated, '"ingredients": ["soy sauce"]',
        actual_draft, '{"ingredients": ["soy sauce"]}',
        NO_RESTRICTIONS,
    )
    assert blocking == []
    assert unverified == critique


def test_hallucinated_model_warning_still_blocks_for_known_allergy():
    """Conservative safety rule must not silently release restricted profiles."""
    critique = ["This recipe contains a possible peanut allergen."]
    blocking, unverified = grounded_llm_issues(
        critique, "Made up sentence", "",
        "Try **Bourbon Chicken** (#45809).",
        '{"contains_user_allergens": []}',
        {"conditions": [], "allergies": ["peanut"]},
    )
    assert blocking == critique
    assert unverified == []
