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
    if re.search(r"\b(?:sodium[\s-]*free|zero sodium|no sodium|salt[\s-]*free)\b", text):
        if not nutr or any(float(n.get("sodium_mg") or 0) >= 5 for n in nutr):
            issues.append("Do not describe the listed recipes as sodium-free without supporting values.")
    if re.search(r"\b(?:low[\s-]*sodium|low in sodium|low salt)\b", text):
        if not nutr or any(float(n.get("sodium_mg") or 0) > 140 for n in nutr):
            issues.append("Do not call recipes low-sodium when any cited serving exceeds 140 mg sodium.")
    if re.search(r"\b(?:sugar[\s-]*free|zero sugar|no sugar)\b", text):
        if not nutr or any(float(n.get("sugar_g") or 0) >= 0.5 for n in nutr):
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
    """Return (blocking, unsupported) LLM issues.

    The reviewer must quote BOTH the draft and the factual evidence verbatim.
    A failure of proof on an unrestricted profile is logged, not trusted as a
    safety fact. If user reports health conditions or allergies, fail closed.
    """
    if not issues:
        return [], []
    linked = (len(draft_quote.strip()) >= 4 and len(fact_quote.strip()) >= 4
              and draft_quote.strip().casefold() in draft.casefold()
              and fact_quote.strip().casefold() in facts_text.casefold())
    if linked or profile.get("conditions") or profile.get("allergies"):
        return list(issues), []
    return [], list(issues)
