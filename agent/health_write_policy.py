"""Conservative, non-LLM authorization for conversational health-profile changes.

Only the CURRENT user's own explicit, first-person health statements may modify
conditions or allergies. If wording is ambiguous, this extractor does not change
health state; a separately confirmed UI form is the safer way to handle it.
The LLM profile agent has NO update_user_profile tool; the graph calls it only
with changes derived from the original unmodified user message.
"""
from __future__ import annotations

import re

CONDITIONS = {
    "diabetes": r"(?:type[\s-]*2\s+diabetes|diabetes)",
    "hypertension": r"(?:high\s+blood\s+pressure|hypertension)",
    "hyperuricemia": r"(?:hyperuricemia|gout|high\s+uric\s+acid)",
}
ALLERGIES = {
    "peanut": r"peanuts?",
    "milk": r"(?:milk|dairy)",
    "wheat": r"(?:wheat|gluten)",
    "egg": r"eggs?",
    "fish": r"fish",
    "shellfish": r"(?:shellfish|shrimp|prawns?|crab)",
    "soy": r"(?:soy|soya)",
    "sesame": r"sesame",
    "tree_nut": r"(?:tree[\s-]*nuts?|almonds?|walnuts?|cashews?)",
}

# Statements are deliberately narrow. Questions, hypotheticals, reported
# third-party speech, and lists of example conditions do not authorize writes.
THIRD_PARTY = re.compile(
    r"\b(?:my (?:friend|wife|husband|child|mother|father|patient)|"
    r"for (?:a|my|someone)|(?:what if|suppose|imagine|example|e\.g\.|hypothetically))\b",
    re.I,
)
POSITIVE_CONDITION = r"(?:i\s+(?:have|was diagnosed with)|i(?:'m| am)\s+(?:living with|diagnosed with))"
NEGATIVE_CONDITION = r"(?:i\s+(?:don't|do not|never|no longer)\s+have|i(?:'m| am)\s+not\s+diagnosed with)"
POSITIVE_ALLERGY = r"(?:i\s+have\s+(?:an?\s+)?(?:\w+\s+){0,2}allerg(?:y|ies)\s+to|i(?:'m| am)\s+allergic to)"
NEGATIVE_ALLERGY = r"(?:i\s+(?:don't|do not|no longer)\s+have\s+(?:an?\s+)?(?:food\s+)?allerg(?:y|ies)\s+to|i(?:'m| am)\s+not\s+allergic to)"


def _mentions_after(text: str, preamble: str, synonyms: dict[str, str]) -> tuple[set[str], set[str]]:
    add: set[str] = set()
    remove: set[str] = set()
    for key, phrase in synonyms.items():
        positive = re.search(rf"\b{preamble}\s+(?:a\s+|an\s+)?{phrase}\b", text, re.I)
        if positive:
            add.add(key)
    return add, remove


def authorized_health_changes(message: str, current: dict) -> dict:
    """Return ONLY changes explicitly authorized by the raw user message.

    Unknown syntax leaves all health fields untouched; the application can ask
    for clarification or use a verified profile-edit form instead. This parser
    never interprets a request like 'suggest two more recipes' as health data.
    """
    original = (message or "").strip()
    txt = re.sub(r"\s+", " ", original.casefold().replace("’", "'"))
    if not txt or THIRD_PARTY.search(txt) or txt.endswith("?"):
        return {}
    result: dict = {}
    conditions = set(current.get("conditions") or [])
    allergies = set(current.get("allergies") or [])
    cond_changed = False
    allergy_changed = False

    # Explicit no-known-condition and no-known-allergies statements.
    if re.search(r"\bi\s+(?:have|have got)\s+(?:no|zero)\s+(?:medical\s+|health\s+)?conditions\b", txt):
        conditions.clear()
        cond_changed = True
    if re.search(r"\bi\s+(?:have|have got)\s+(?:no|zero)\s+(?:known\s+|food\s+)?allerg(?:y|ies)\b", txt):
        allergies.clear()
        allergy_changed = True

    for key, phrase in CONDITIONS.items():
        yes = re.search(rf"\b{POSITIVE_CONDITION}\s+(?:a\s+|an\s+)?{phrase}\b", txt)
        no = re.search(rf"\b{NEGATIVE_CONDITION}\s+(?:a\s+|an\s+)?{phrase}\b", txt)
        if yes or no:
            cond_changed = True
            conditions.discard(key) if no else conditions.add(key)

    for key, phrase in ALLERGIES.items():
        yes = re.search(rf"\b{POSITIVE_ALLERGY}\s+{phrase}\b", txt)
        no = re.search(rf"\b{NEGATIVE_ALLERGY}\s+{phrase}\b", txt)
        if yes or no:
            allergy_changed = True
            allergies.discard(key) if no else allergies.add(key)

    # A user may assert none of both without an LLM deciding which fields to
    # write; never infer a negative declaration merely from a food request.
    if cond_changed:
        result["conditions"] = sorted(conditions)
    if allergy_changed:
        result["allergies"] = sorted(allergies)
    if cond_changed and allergy_changed:
        result["confirmed"] = True
    return result
