"""Document schema and source-to-ES conversion; deliberately independent from an ES server."""
from __future__ import annotations
import html
import math
from typing import Any

NUTRI_MAP = {
    "Calories": "calories", "CarbohydrateContent": "carbs_g", "SugarContent": "sugar_g",
    "SodiumContent": "sodium_mg", "SaturatedFatContent": "sat_fat_g", "ProteinContent": "protein_g",
    "FatContent": "fat_g", "FiberContent": "fiber_g", "CholesterolContent": "cholesterol_mg",
}

MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"dynamic": "strict", "properties": {
        "recipe_id": {"type": "integer"},
        "item_idx": {"type": "integer"},
        "name": {"type": "text"},
        "description": {"type": "text"},
        "category": {"type": "keyword"},
        "keywords": {"type": "text"},
        "ingredients": {"type": "text"},
        "ingredient_text": {"type": "text"},
        "image_url": {"type": "keyword", "index": False},
        "total_minutes": {"type": "float"},
        "nutrition": {"properties": {name: {"type": "float"} for name in NUTRI_MAP.values()}},
        "nutrition_complete": {"type": "boolean"},
        "allergens": {"type": "keyword"},
        "compliant_conditions": {"type": "keyword"},
        "violated_conditions": {"type": "keyword"},
        "popularity": {"type": "float"},
    }}
}


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) and v >= 0 else None
    except (ValueError, TypeError):
        return None


def strings(value) -> list[str]:
    if not isinstance(value, (tuple, list)):
        return []
    return [html.unescape(str(x)).strip() for x in value if x is not None and str(x).strip()]


def to_document(row: dict, conditions: list[str], allergens: list[str]) -> dict:
    nutrition = {new: number(row.get(old)) for old, new in NUTRI_MAP.items()}
    ingredient_names = strings(row.get("RecipeIngredientParts"))
    # Safeguard: a missing nutritional field cannot be treated as compliant.
    critical = ("calories", "carbs_g", "sugar_g", "sodium_mg", "sat_fat_g")
    complete = bool(ingredient_names and all(nutrition[k] is not None for k in critical))
    conditions_ok = [c for c in conditions if complete and str(row.get("ok_" + c)).lower() == "true"]
    allergies_present = [a for a in allergens if str(row.get("contains_" + a)).lower() == "true"]
    rid = int(row["RecipeId"])
    item_idx = row.get("item_idx")
    try:
        item_idx = int(item_idx) if item_idx is not None and math.isfinite(float(item_idx)) else None
    except (TypeError, ValueError):
        item_idx = None
    return {
        "recipe_id": rid, "item_idx": item_idx,
        "name": html.unescape(str(row.get("Name") or "")),
        "description": html.unescape(str(row.get("Description") or "")),
        "category": str(row.get("RecipeCategory") or ""),
        "keywords": strings(row.get("Keywords")), "ingredients": ingredient_names,
        "ingredient_text": " | ".join([*ingredient_names, str(row.get("Name") or ""), *strings(row.get("RecipeInstructions"))]),
        "image_url": row.get("ImageUrl") if isinstance(row.get("ImageUrl"), str) else None,
        "total_minutes": number(row.get("TotalMinutes")),
        "nutrition": nutrition,
        "nutrition_complete": complete,
        "allergens": allergies_present,
        "compliant_conditions": conditions_ok,
        "violated_conditions": [c for c in conditions if c not in conditions_ok],
        "popularity": number(row.get("ReviewCount")) or 0.0,
    }

