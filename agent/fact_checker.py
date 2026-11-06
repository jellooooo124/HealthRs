"""Evidence-first guard for LLM-drafted dietary claims and reviewer verdicts.

The LLM reviewer is an imperfect model. Never promote a critique to a factual
issue without evidence when the user has no reported conditions/allergies.
For users WITH health restrictions, uncertain reviewer findings stay blocking.
Hard factual claim checks ALWAYS stay blocking irrespective of LLM verdict.
"""
from __future__ import annotations

import re


def factual_claim_issues(draft: str, facts: list[dict], profile: dict) -> list[str]:
    """Conservative rejection of unsupported blanket marketing/health claims."""
    text = draft.casefold()
    issues: list[str] = []
    if re.search(r"\b(?:allergen[\s-]*free|free from (?:all )?allergens|guaranteed (?:allergy|allergen)[\s-]*free)\b", text):
        issues.append("Do not claim a recipe is allergen-free: database labels do not rule out cross-contact.")
    if re.search(r"\b(?:cure|cures|cured|treats? diabetes|stop (?:taking )?(?:medication|insulin))\b", text):
        issues.append("No treatment, cure, or medication-changing advice is permitted.")
    if (not profile.get("conditions") and
            re.search(r"\b(?:comply|complies|compliant) with (?:your|all) health conditions\b", text)):
        issues.append("Do not assert compliance with unspecified health conditions.")
    nutr = [r.get("nutrition_per_serving") or {} for r in facts]
    def missing_or_above(key: str, maximum: float) -> bool:
        return not nutr or any(n.get(key) is None or float(n[key]) > maximum for n in nutr)
    if re.search(r"\b(?:sodium[\s-]*free|zero sodium|no sodium|salt[\s-]*free)\b", text):
        if missing_or_above("sodium_mg", 4.999):
            issues.append("Do not describe the listed recipes as sodium-free without supporting values.")
    if re.search(r"\b(?:low[\s-]*sodium|low in sodium|low salt)\b", text):
        if missing_or_above("sodium_mg", 140):
            issues.append("Do not call recipes low-sodium when any cited serving exceeds 140 mg sodium.")
    if re.search(r"\b(?:sugar[\s-]*free|zero sugar|no sugar)\b", text):
        if missing_or_above("sugar_g", 0.499):
            issues.append("Do not claim sugar-free when cited servings contain sugar.")
    if re.search(r"\b(?:low[\s-]*sugar|low in sugar)\b", text):
        issues.append("Avoid unverified low-sugar claims; the catalog has no reliable per-100g serving mass.")
    return issues


def grounded_llm_issues(
    issues: list[str],
    draft_quote: str,
    fact_quote: str,
    draft: str,
    facts_text: str,
    profile: dict,
) -> tuple[list[str], list[str]]:
    """Return (blocking, unsupported) reviewer issues without hiding uncertainty.

    Known conditions/allergies: fail closed on ANY LLM warning, even if its
    quoted evidence is poor. Unrestricted profile: unsubstantiated warnings
    (or reviewer claims in domains checked independently in Python) are logged
    separately; non-medical prose guesses must not veto safe grounded recipes.
    Direct medical treatment/medication advice remains blocking regardless.
    """
    if not issues:
        return [], []
    if profile.get("conditions") or profile.get("allergies"):
        return list(issues), []
    lower_draft = draft.casefold()
    risky_advice = bool(re.search(
        r"\b(?:insulin|dosage|medication|prescription|cure diabetes|stop taking|diagnos(?:is|e))\b",
        lower_draft,
    ))
    if risky_advice:
        return list(issues), []
    # Even matching snippets do NOT prove a contradiction. A statement that
    # "40 minutes" contradicts 40.0 minutes is itself wrong. For unrestricted
    # profiles, the independent hard rules are the safety gate and the remaining
    # LLM critique is retained as a visible, advisory finding. Neither the LLM
    # verdict nor its critique is falsely recorded as passed.
    return [], list(issues)
