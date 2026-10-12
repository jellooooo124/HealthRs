"""
Synthetic health profiles for Food.com users (fixed seed, fully reproducible).

* Conditions: independent Bernoulli draws with US-adult prevalence (configs/health_rules.yaml).
  They are NOT derived from rating history, so the model cannot "discover" a signal we planted.
* Allergies: drawn with US-adult prevalence, but only for users none of whose positive
  recipes contain that allergen (a peanut-allergic user cannot have loved peanut recipes).
  Effective prevalence is therefore lower than the population rate; this is reported.
* taste_prefs: the user's 3 most frequent recipe categories in training positives.
"""
import json

import numpy as np
import pandas as pd


def synthesize_profiles(
    users: np.ndarray,
    user_type: dict,                 # user_id -> "warm" | "cold"
    positives: pd.DataFrame,         # UserId, RecipeId (all known positives, used for allergy consistency)
    train_pos: pd.DataFrame,         # UserId, RecipeId (training positives, used for taste prefs)
    recipe_info: pd.DataFrame,       # indexed by RecipeId: contains_<allergen>, RecipeCategory
    rules: dict,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = np.sort(users)
    n = len(users)
    prev = rules["prevalence"]

    conditions = {c: rng.random(n) < p for c, p in prev["conditions"].items()}

    # allergen exposure in each user's positives -> eligibility for that allergy
    allergens = list(prev["allergies"].keys())
    exposed = (
        positives[["UserId", "RecipeId"]]
        .join(recipe_info[[f"contains_{a}" for a in allergens]], on="RecipeId")
        .groupby("UserId")[[f"contains_{a}" for a in allergens]].any()
        .reindex(users, fill_value=False)
    )
    allergies = {a: (rng.random(n) < p) & ~exposed[f"contains_{a}"].to_numpy() for a, p in prev["allergies"].items()}

    cats = (
        train_pos.join(recipe_info["RecipeCategory"], on="RecipeId")
        .groupby("UserId")["RecipeCategory"]
        .agg(lambda s: s.value_counts().index[:3].tolist())
    )

    rows = []
    for i, u in enumerate(users):
        conds = [c for c in conditions if conditions[c][i]]
        rows.append({
            "user_id": int(u),
            "source": f"foodcom_{user_type[u]}",
            "conditions": json.dumps(conds),
            "allergies": json.dumps([a for a in allergies if allergies[a][i]]),
            "goals": json.dumps([rules["goals"][c] for c in conds]),
            "taste_prefs": json.dumps(cats.get(u, [])),
        })
    return pd.DataFrame(rows)
