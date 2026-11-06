"""Strict regression tests: LLM-suggested health changes have no authority.

These are deliberately deterministic; a passing result means the Python
authorization/provenance gates work, NOT that a real LLM behaves correctly.
"""
from types import SimpleNamespace

from agent.health_write_policy import authorized_health_changes
from agent.recipe_provenance import (
    recommendation_provenance_issues, detail_provenance_issues,
)


PROFILE = {"conditions": [], "allergies": [], "confirmed": True}
CANDIDATES = [{"recipe_id": 42, "name": "Chicken soup"}]


def test_more_recipes_cannot_invent_diabetes_or_wheat_allergy():
    assert authorized_health_changes(
        "Can you suggest two more chicken meals, different from the first ones?", PROFILE,
    ) == {}


def test_recipe_terms_cannot_be_mistaken_for_medical_state():
    assert authorized_health_changes("Suggest a low-sodium diabetes-friendly dinner.", PROFILE) == {}
    assert authorized_health_changes("What if I have a peanut allergy?", PROFILE) == {}
    assert authorized_health_changes("My friend has diabetes.", PROFILE) == {}


def test_explicit_first_person_peanut_allergy_can_be_added():
    assert authorized_health_changes("I have a peanut allergy.", PROFILE) == {
        "allergies": ["peanut"],
    }
    assert authorized_health_changes("I'm allergic to peanuts.", PROFILE) == {
        "allergies": ["peanut"],
    }


def test_explicit_diabetes_and_denial_are_handled_separately():
    assert authorized_health_changes("I have diabetes.", PROFILE) == {
        "conditions": ["diabetes"],
    }
    current = {"conditions": ["diabetes"], "allergies": ["peanut"], "confirmed": True}
    assert authorized_health_changes("I don't have diabetes.", current) == {
        "conditions": [],
    }


def test_profile_explicitly_confirms_no_conditions_and_no_allergies():
    current = {"conditions": ["diabetes"], "allergies": ["peanut"], "confirmed": False}
    assert authorized_health_changes(
        "I have no medical conditions and I have no food allergies.", current,
    ) == {"conditions": [], "allergies": [], "confirmed": True}


def test_unrecognized_speech_never_silently_replaces_health_data():
    current = {"conditions": ["hypertension"], "allergies": ["peanut"], "confirmed": True}
    assert authorized_health_changes("Please give me some more chicken dishes.", current) == {}


def test_recommendations_only_accept_current_tool_ids():
    assert recommendation_provenance_issues("Try chicken soup (#42).", CANDIDATES) == []
    issues = recommendation_provenance_issues("Try #130001 or #42.", CANDIDATES)
    assert any("#130001" in x for x in issues)


def test_recommendation_rejects_existing_but_not_retrieved_recipe():
    other_catalog_id = 20755
    issues = recommendation_provenance_issues(f"Try #{other_catalog_id}.", CANDIDATES)
    assert any(str(other_catalog_id) in x for x in issues)


def test_recommendation_requires_result_and_citation():
    assert recommendation_provenance_issues("Try my fictional chicken pasta.", CANDIDATES)
    assert recommendation_provenance_issues("Try #130001.", [])


def test_recipe_detail_only_for_previously_shown_assistant_ids():
    history = [SimpleNamespace(type="human", content="Are there recipes?"),
               SimpleNamespace(type="ai", content="Try soup (#42).")]
    assert detail_provenance_issues("What is in #42?", "Ingredients for #42", history) == []
    assert detail_provenance_issues("What is in #99?", "Ingredients for #99", history)
