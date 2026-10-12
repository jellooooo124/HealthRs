"""
Two-stage recommender used by the agent's `recommend_recipes` tool.

    all catalog recipes (~46k)
      │  structured constraints (allergens, excluded/required ingredients, time, category, calories, seen)
      ▼
    RETRIEVAL (~300 candidates)
      ├─ channel 1 "taste":   plain BPR-MF top-N            (warm users only)
      └─ channel 2 "healthy": most popular recipes that comply with all of the user's conditions
      ▼
    RANKING: BPR-MF + adaptation A, popularity-calibrated (z(score) + α·z(log pop)); cold-start users
             (no history) are ranked by popularity; minus health_penalty for violating recipes (compliant first)
      ▼
    RERANK: safety hard check (allergens / excluded ingredients, optional strict health),
            diversity (≤ max_per_category per RecipeCategory)  →  top-K with nutrition facts

Recipes are not embedded or vector-searched: the LLM turns the request into structured constraints.
"""
import re

import numpy as np
import torch

from src.data.foodcom import NUTRITION_COLS
from src.data.health import recipe_texts

# meal types -> Food.com RecipeCategory / Keywords (the raw category field mixes dish type, ingredient and time)
_DESSERT = {"Dessert", "Bar Cookie", "Drop Cookies", "Pie", "Candy", "Frozen Desserts", "Cheesecake", "Gelatin", "Tarts"}
_DRINK = {"Beverages", "Smoothies", "Punch Beverage", "Shakes", "Lemon"}
_BAKERY = {"Quick Breads", "Breads", "Yeast Breads", "Sourdough Breads"}
_CONDIMENT = {"Sauces", "Salad Dressings", "Spreads", "Chutneys", "Jellies", "Low Protein"}
_SIDE = {"Vegetable", "Potato", "Rice", "Long Grain Rice", "White Rice", "Brown Rice", "Beans", "Corn", "Greens",
         "Spinach", "Cauliflower", "Yam/Sweet Potato", "Onions", "Grains", "Black Beans"}
MEAL_TYPES = ("breakfast", "main", "side", "dessert", "snack", "drink")

NUTRITION_UNITS = {"Calories": "kcal", "CholesterolContent": "mg", "SodiumContent": "mg"}


class TwoStageRecommender:
    def __init__(self, data, taste_model, health_model, cfg: dict):
        self.data = data
        self.cfg = cfg
        self.recipes = data.recipes
        self.conditions, self.allergens = data.conditions, data.allergens
        with torch.no_grad():
            self.taste_items = taste_model.item_vec().cpu().numpy()
            self.taste_bias = taste_model.bias().cpu().numpy()
            self.health_items = health_model.item_vec().cpu().numpy()
            self.health_bias = health_model.bias().cpu().numpy()
        self.taste_model, self.health_model = taste_model, health_model
        self.pop_order = np.argsort(-data.item_pop, kind="stable")
        self.log_pop = np.log1p(data.item_pop).astype(np.float32)
        self.log_pop_z = (self.log_pop - self.log_pop.mean()) / self.log_pop.std()
        self.ingredients = self.recipes["RecipeIngredientParts"].map(lambda xs: " | ".join(xs).lower()).to_numpy()
        # names + recipe name + steps: used for exclusions (ingredient lists in Food.com are incomplete)
        self.fulltext = recipe_texts(self.recipes).map(lambda xs: " | ".join(xs).lower()).to_numpy()
        self.category = self.recipes["RecipeCategory"].fillna("").str.lower().to_numpy()
        self.minutes = self.recipes["TotalMinutes"].to_numpy(float)
        self.calories = self.recipes["Calories"].to_numpy(float)
        self.meal_masks = self._meal_masks()

    def _meal_masks(self) -> dict:
        cat = self.recipes["RecipeCategory"].fillna("")
        kw = self.recipes["Keywords"].map(set)
        has = lambda word: kw.map(lambda s: word in s).to_numpy()  # noqa: E731
        dessert = cat.isin(_DESSERT).to_numpy() | has("Dessert")
        drink = cat.isin(_DRINK).to_numpy() | has("Beverages")
        breakfast = (cat == "Breakfast").to_numpy() | has("Breakfast") | has("Brunch")
        snack = (cat == "Lunch/Snacks").to_numpy() | has("Lunch/Snacks")
        other = dessert | drink | cat.isin(_BAKERY | _CONDIMENT).to_numpy()
        side = cat.isin(_SIDE).to_numpy() & ~other
        main = ~(other | (cat == "Breakfast").to_numpy() | snack | side)
        return {"breakfast": breakfast & ~drink, "main": main, "side": side, "dessert": dessert,
                "snack": snack & ~dessert & ~drink, "drink": drink}

    # ---------------- user context
    def _context(self, user_idx, conditions, allergies) -> dict:
        warm = user_idx is not None and 0 <= user_idx < self.data.n_users
        if conditions is None:
            conditions = [c for c, on in zip(self.conditions, self.data.user_cond[user_idx]) if on] if warm else []
        if allergies is None:
            allergies = [a for a, on in zip(self.allergens, self.data.user_alg[user_idx]) if on] if warm else []
        cond = np.array([c in conditions for c in self.conditions])
        alg = np.array([a in allergies for a in self.allergens])
        return {"user_idx": user_idx if warm else None, "conditions": list(conditions), "allergies": list(allergies),
                "cond": cond, "alg": alg}

    def _phrase_mask(self, phrases: list[str], texts: np.ndarray | None = None) -> np.ndarray:
        pat = re.compile(r"\b(?:" + "|".join(re.escape(p.lower().strip()) for p in phrases) + r")s?\b")
        texts = self.ingredients if texts is None else texts
        return np.array([bool(pat.search(s)) for s in texts])

    def allowed_mask(self, ctx: dict, constraints: dict, exclude_seen: bool = True,
                     extra_exclude: np.ndarray | None = None) -> np.ndarray:
        ok = np.ones(self.data.n_items, bool)
        if ctx["alg"].any():
            ok &= ~self.data.item_allergen[:, ctx["alg"]].any(1)
        c = constraints or {}
        if c.get("exclude_ingredients"):
            ok &= ~self._phrase_mask(c["exclude_ingredients"], self.fulltext)   # conservative: also scan steps
        if c.get("include_ingredients"):
            ok &= self._phrase_mask(c["include_ingredients"])
        if c.get("max_minutes"):
            ok &= self.minutes <= float(c["max_minutes"])        # unknown time -> excluded when a limit is given
        if c.get("max_calories"):
            ok &= self.calories <= float(c["max_calories"])
        if c.get("meal_type") in self.meal_masks:
            ok &= self.meal_masks[c["meal_type"]]
        if c.get("categories"):
            wanted = [x.lower() for x in c["categories"]]
            ok &= np.array([any(w in cat for w in wanted) for cat in self.category])
        if exclude_seen and ctx["user_idx"] is not None:
            ok[self.data.train_csr[ctx["user_idx"]].indices] = False
        if extra_exclude is not None:
            ok[extra_exclude] = False
        return ok

    def compliant_mask(self, ctx: dict) -> np.ndarray:
        if not ctx["cond"].any():
            return np.ones(self.data.n_items, bool)
        return self.data.item_ok[:, ctx["cond"]].all(1)

    # ---------------- stage 1: retrieval
    def retrieve(self, ctx: dict, allowed: np.ndarray) -> tuple[np.ndarray, dict]:
        cands, source = [], {}
        if ctx["user_idx"] is not None:
            with torch.no_grad():
                uv = self.taste_model.user_vec(torch.tensor([ctx["user_idx"]])).cpu().numpy()[0]
            s = np.where(allowed, self.taste_items @ uv + self.taste_bias, -np.inf)
            n = min(self.cfg["taste_n"], int(allowed.sum()))
            top = np.argpartition(-s, n - 1)[:n] if n else np.array([], int)
            for i in top[np.argsort(-s[top])]:
                cands.append(i)
                source[int(i)] = "taste"
        healthy = self.pop_order[(allowed & self.compliant_mask(ctx))[self.pop_order]][: self.cfg["healthy_n"]]
        for i in healthy:
            if int(i) not in source:
                cands.append(i)
                source[int(i)] = "healthy"
            else:
                source[int(i)] = "taste+healthy"
        return np.array(cands, dtype=np.int64), source

    # ---------------- stage 2: ranking
    def rank(self, ctx: dict, cands: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if ctx["user_idx"] is not None:
                uv = self.health_model.user_vec(torch.tensor([ctx["user_idx"]])).cpu().numpy()[0]
            elif hasattr(self.health_model, "cold_user_vec"):
                uv = self.health_model.cold_user_vec(torch.tensor(ctx["cond"][None], dtype=torch.float32)).cpu().numpy()[0]
            else:
                uv = np.zeros(self.health_items.shape[1], np.float32)
        alpha = self.cfg.get("pop_alpha", 0.0)
        if ctx["user_idx"] is None:
            # cold-start: no taste signal (condition embeddings carry none, step-4 finding) ->
            # popularity among the safe candidates (allergens already removed, compliant ones first below)
            scores = self.log_pop[cands].copy()
        elif alpha:   # popularity calibration: z-scored model scores (over all items) + α · z-scored log popularity
            full = self.health_items @ uv + self.health_bias
            sd = full.std()
            scores = (full[cands] - full.mean()) / (sd if sd > 0 else 1.0) + alpha * self.log_pop_z[cands]
        elif not np.any(uv) and not np.any(self.health_bias):
            # no user signal at all (cold user without conditions, or model without A and bias): popularity
            scores = self.log_pop[cands].copy()
        else:
            scores = self.health_items[cands] @ uv + self.health_bias[cands] + 1e-3 * self.log_pop[cands]
        return scores - self.cfg.get("health_penalty", 0.0) * ~self.compliant_mask(ctx)[cands]

    # ---------------- stage 3: rerank
    def rerank(self, ctx: dict, cands: np.ndarray, scores: np.ndarray, k: int, strict_health: bool) -> list[int]:
        compliant = self.compliant_mask(ctx)
        safe = [int(i) for i in cands[np.argsort(-scores, kind="stable")]
                if not (ctx["alg"].any() and self.data.item_allergen[i, ctx["alg"]].any())   # safety re-check
                and (compliant[i] or not strict_health)]
        per_cat, out = {}, []
        for i in safe:
            cat = self.category[i]
            if per_cat.get(cat, 0) < self.cfg["max_per_category"]:
                per_cat[cat] = per_cat.get(cat, 0) + 1
                out.append(i)
            if len(out) == k:
                return out
        # diversity cap left slots empty (e.g. the user asked for one category): fill in score order
        return out + [i for i in safe if i not in out][: k - len(out)]

    # ---------------- public API
    def recommend(self, user_idx: int | None = None, conditions: list[str] | None = None,
                  allergies: list[str] | None = None, constraints: dict | None = None, k: int = 10,
                  strict_health: bool = False) -> dict:
        ctx = self._context(user_idx, conditions, allergies)
        cands, source, scores, top = self.run(ctx, self.allowed_mask(ctx, constraints), k, strict_health)
        score_of = dict(zip(cands.tolist(), scores.tolist()))
        return {"user": self._public_ctx(ctx), "n_candidates": int(len(cands)),
                "recipes": [self._describe(i, ctx, score_of[i], source[i]) for i in top]}

    def run(self, ctx: dict, allowed: np.ndarray, k: int = 10, strict_health: bool = False):
        """Retrieval -> ranking -> rerank on item indices. Returns (candidates, source, scores, top-k)."""
        cands, source = self.retrieve(ctx, allowed)
        if len(cands) == 0:
            return cands, source, np.array([]), []
        scores = self.rank(ctx, cands)
        return cands, source, scores, self.rerank(ctx, cands, scores, k, strict_health)

    def _public_ctx(self, ctx):
        return {k: ctx[k] for k in ("user_idx", "conditions", "allergies")}

    def _describe(self, i: int, ctx: dict, score: float, source: str) -> dict:
        r = self.recipes.iloc[i]
        violations = {c: r[f"viol_{c}"] for c in ctx["conditions"] if not r[f"ok_{c}"]}
        return {
            "recipe_id": int(r["RecipeId"]), "name": r["Name"], "category": r["RecipeCategory"],
            "total_minutes": None if np.isnan(r["TotalMinutes"]) else float(r["TotalMinutes"]),
            "nutrition_per_serving": {c.replace("Content", ""): f"{r[c]:.1f} {NUTRITION_UNITS.get(c, 'g')}" for c in NUTRITION_COLS},
            "complies_with": {c: bool(r[f"ok_{c}"]) for c in ctx["conditions"]},
            "violations": violations,
            "ingredients": list(r["RecipeIngredientParts"]),
            "image_url": r["ImageUrl"], "score": round(float(score), 4), "retrieved_by": source,
        }
