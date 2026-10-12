"""
Guideline-based health labels for recipes (configs/health_rules.yaml).

For every recipe we compute
  * `contains_<group>` for ingredient groups (high_purine, alcohol, hfcs) and allergens
  * `ok_<condition>`   True iff the recipe passes every per-serving rule of that condition
  * `viol_<condition>` the failed rules, e.g. "SodiumContent>500;SaturatedFatContent>6"
These labels drive the health loss (adaptation B), the hard filter and the violation-rate metric.
"""
import re

import pandas as pd


class IngredientMatcher:
    """Whole-word phrase matcher with exclusion phrases (removed before matching)."""

    def __init__(self, include: list[str], exclude: list[str]):
        def alt(words):  # longest first so multi-word phrases win
            return "|".join(re.escape(w.lower()) for w in sorted(set(words), key=len, reverse=True))

        self._inc = re.compile(rf"\b(?:{alt(include)})\b")
        self._exc = re.compile(rf"\b(?:{alt(exclude)})\b") if exclude else None

    def matches(self, ingredient: str) -> bool:
        s = ingredient.lower()
        if self._exc is not None:
            s = self._exc.sub(" ", s)
        return bool(self._inc.search(s))

    def any(self, ingredients: list[str]) -> bool:
        return any(self.matches(i) for i in ingredients)


def build_matchers(groups: dict) -> dict[str, IngredientMatcher]:
    return {name: IngredientMatcher(g["include"], g.get("exclude", [])) for name, g in groups.items()}


def label_recipes(recipes: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Return a frame (indexed like `recipes`) with contains_*, ok_* and viol_* columns.

    Ingredient rules scan the ingredient names AND the recipe name and cooking steps: Food.com's
    RecipeIngredientParts silently drops some ingredients (~70% of recipes list fewer names than
    quantities), so names alone would miss allergens. Scanning the steps is conservative (safety first).
    """
    text = recipe_texts(recipes).map(lambda xs: " | ".join(xs).lower())
    out = pd.DataFrame(index=recipes.index)

    matchers = {**build_matchers(rules["ingredient_groups"]), **build_matchers(rules["allergens"])}
    for name, m in matchers.items():
        out[f"contains_{name}"] = text.map(m.matches)

    for cond, spec in rules["conditions"].items():
        failed = pd.DataFrame(index=recipes.index)
        for col, limit in spec.get("nutrient_max", {}).items():
            failed[f"{col}>{limit}"] = recipes[col] > limit
        for group in spec.get("ingredients_forbidden", []):
            failed[f"contains:{group}"] = out[f"contains_{group}"]
        out[f"ok_{cond}"] = ~failed.any(axis=1)
        reasons = pd.Series("", index=recipes.index)
        for rule in failed.columns:
            reasons = reasons + failed[rule].map({True: rule + ";", False: ""})
        out[f"viol_{cond}"] = reasons.str.rstrip(";")
    return out


def recipe_texts(recipes: pd.DataFrame) -> pd.Series:
    """Strings scanned by ingredient rules: ingredient names + recipe name + cooking steps."""
    steps = recipes["RecipeInstructions"] if "RecipeInstructions" in recipes else pd.Series([[]] * len(recipes), index=recipes.index)
    return pd.Series([list(p) + [n] + list(st) for p, n, st in zip(recipes["RecipeIngredientParts"], recipes["Name"], steps)],
                     index=recipes.index)


def allergen_names(rules: dict) -> list[str]:
    return list(rules["allergens"].keys())


def condition_names(rules: dict) -> list[str]:
    return list(rules["conditions"].keys())
