"""
Recommender service behind the `recommend_recipes` MCP tool: wraps the two-stage recommender
(src/models/retrieval.py) with the profile store.

Safety contract: conditions and allergies always come from the profile store (never from the LLM),
recipes the user disliked are excluded, and every returned recipe is allergen-free.
"""
import html
import os
import threading

import numpy as np

from agent.profile_store import ProfileStore
from src.config import get_project_root, load_config, get_outputs_dir
from src.data.foodcom import NUTRITION_COLS
from src.models.data import RecData
from src.models.retrieval import TwoStageRecommender
from src.models.trainer import load_checkpoint
from src.utils.logger import get_logger

logger = get_logger("healthrec.service")

# compact nutrition keys for LLM context (per serving)
_SHORT = {"Calories": "kcal", "CarbohydrateContent": "carbs_g", "SugarContent": "sugar_g", "SodiumContent": "sodium_mg",
          "SaturatedFatContent": "sat_fat_g", "FatContent": "fat_g", "ProteinContent": "protein_g",
          "FiberContent": "fiber_g", "CholesterolContent": "cholesterol_mg"}
CONSTRAINT_KEYS = ("meal_type", "max_minutes", "max_calories", "categories", "include_ingredients", "exclude_ingredients")


class RecommenderService:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "RecommenderService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        root = get_project_root()
        model_cfg = load_config("model")
        self.data = RecData(os.path.join(root, load_config("data")["paths"]["processed_dir"]), load_config("health_rules"))
        ck = lambda n: load_checkpoint(get_outputs_dir("checkpoints", f"{n}.pt"), self.data)  # noqa: E731
        serving = model_cfg["serving"]
        self.rec = TwoStageRecommender(self.data, ck(serving["taste_checkpoint"]), ck(serving["ranker_checkpoint"]),
                                       model_cfg["retrieval"])
        self.store = ProfileStore()
        self.recipes = self.data.recipes
        self.item_of = dict(zip(self.recipes["RecipeId"].astype(int), self.recipes["item_idx"].astype(int)))
        logger.info(f"[service] ready: {self.data.n_items} recipes, models {serving}")

    # ---------------- helpers
    def _user_idx(self, profile: dict) -> int | None:
        if profile.get("source") == "foodcom_warm" and profile.get("user_idx") is not None:
            return int(profile["user_idx"])
        return None   # cold start (demo-created or Food.com user outside the k-core)

    def compact(self, i: int, conditions: list[str], with_steps: bool = False) -> dict:
        r = self.recipes.iloc[i]
        out = {
            "recipe_id": int(r["RecipeId"]), "name": html.unescape(r["Name"]), "category": r["RecipeCategory"],
            "total_minutes": None if np.isnan(r["TotalMinutes"]) else int(r["TotalMinutes"]),
            "nutrition_per_serving": {_SHORT[c]: round(float(r[c]), 1) for c in NUTRITION_COLS},
            "complies_with": {c: bool(r[f"ok_{c}"]) for c in conditions},
            "violations": {c: r[f"viol_{c}"] for c in conditions if not r[f"ok_{c}"]},
            "ingredients": [html.unescape(x) for x in r["RecipeIngredientParts"]][:6 if not with_steps else 30],
        }
        if with_steps:
            out["steps"] = [html.unescape(x) for x in r["RecipeInstructions"]][:12]
            out["servings"] = None if np.isnan(r["RecipeServings"]) else float(r["RecipeServings"])
            out["image_url"] = r["ImageUrl"]
        return out

    # ---------------- tool implementations
    def recommend(self, user_id: str, constraints: dict | None = None, k: int = 5,
                  exclude_recipe_ids: list[int] | None = None) -> dict:
        profile = self.store.get(user_id) or self.store.create(user_id)
        c = {key: v for key, v in (constraints or {}).items() if key in CONSTRAINT_KEYS and v not in (None, [], "")}
        k = int(max(1, min(int(k or 5), 10)))
        ctx = self.rec._context(self._user_idx(profile), profile["conditions"], profile["allergies"])
        exclude = [self.item_of[r] for r in set(exclude_recipe_ids or []) | self.store.disliked(user_id) if r in self.item_of]
        allowed = self.rec.allowed_mask(ctx, c, extra_exclude=np.array(exclude, dtype=np.int64) if exclude else None)
        cands, source, scores, top = self.rec.run(ctx, allowed, k)
        recipes = [self.compact(i, profile["conditions"]) | {"retrieved_by": source[i]} for i in top]
        return {
            "user": {"user_id": user_id, "conditions": profile["conditions"], "allergies": profile["allergies"],
                     "history": "rating history (warm user)" if ctx["user_idx"] is not None else "no history (cold start)"},
            "applied_constraints": c, "n_candidates": int(len(cands)), "recipes": recipes,
            "note": ("All recipes exclude the user's allergens. Recipes with non-empty 'violations' exceed a "
                     "per-serving guideline limit for that condition and were only used because too few "
                     "compliant recipes matched the constraints." if recipes else
                     "No recipe matched these constraints; relax a constraint and try again."),
        }

    def details(self, recipe_id: int, user_id: str | None = None) -> dict:
        i = self.item_of.get(int(recipe_id))
        if i is None:
            return {"error": f"recipe {recipe_id} not in the catalog"}
        profile = self.store.get(user_id) if user_id else None
        out = self.compact(i, profile["conditions"] if profile else [], with_steps=True)
        if profile and profile["allergies"]:
            hits = [a for a in profile["allergies"] if self.data.item_allergen[i, self.data.allergens.index(a)]]
            out["contains_user_allergens"] = hits
        return out

    def recipe_ok_for(self, recipe_id: int, conditions: list[str], allergies: list[str]) -> dict:
        """Ground-truth safety check used by the safety reviewer."""
        i = self.item_of.get(int(recipe_id))
        if i is None:
            return {"known": False}
        r = self.recipes.iloc[i]
        return {"known": True, "name": html.unescape(r["Name"]),
                "violations": {c: r[f"viol_{c}"] for c in conditions if not r[f"ok_{c}"]},
                "allergens": [a for a in allergies if self.data.item_allergen[i, self.data.allergens.index(a)]]}
