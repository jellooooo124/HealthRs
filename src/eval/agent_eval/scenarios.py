"""
Simulated-user scenarios for the agent evaluation (deterministic, seed from configs/data.yaml).

Each scenario = a persona (ground truth known to the simulated user and the judge) + a task with a
scripted opening message. Warm scenarios use Food.com users with ≥1 condition (their synthetic profile
is already stored); cold scenarios use cold-start users whose health status the agent must ask for.
Personas also carry a disliked ingredient and a time limit that are NOT in the stored profile, so the
agent must react to feedback.
"""
import html
import json
import os

import numpy as np
import pandas as pd

from src.config import get_project_root, load_config

TASKS = [
    {"task": "dinner", "opener": "Can you suggest a dinner I could cook tonight?", "meal": "a dinner main dish"},
    {"task": "breakfast", "opener": "I need some breakfast ideas for this week.", "meal": "breakfast"},
    {"task": "sweet", "opener": "I'm craving something sweet. Any ideas?", "meal": "a dessert or sweet snack"},
    {"task": "lunch_chicken", "opener": "What could I make for lunch with chicken?", "meal": "a lunch dish with chicken"},
]
DISLIKES = ["mushrooms", "olives", "coconut", "cilantro", "pork", "eggplant", "tofu", "blue cheese", "raisins", "spinach"]
TIME_LIMITS = [30, 45, 60]


def build_scenarios(n_warm: int = 11, n_cold: int = 4, n_warm_allergy: int = 4) -> list[dict]:
    root = get_project_root()
    seed = load_config("data")["seed"]
    rng = np.random.default_rng(seed + 7)
    proc = os.path.join(root, load_config("data")["paths"]["processed_dir"])
    prof = pd.read_csv(os.path.join(proc, "profiles.csv"), dtype={"user_idx": "Int64"})
    prof["cond"] = prof["conditions"].map(json.loads)
    prof["alg"] = prof["allergies"].map(json.loads)
    inter = pd.read_parquet(os.path.join(proc, "interactions.parquet"))
    train = inter[inter["split"] == "train"]
    names = pd.read_parquet(os.path.join(proc, "recipes.parquet"), columns=["RecipeId", "Name"]).set_index("RecipeId")["Name"]
    n_train = train.groupby("UserId").size()

    warm = prof[(prof["source"] == "foodcom_warm") & (prof["cond"].map(len) > 0)].copy()
    warm["n"] = warm["user_id"].map(n_train)
    warm = warm[(warm["n"] >= 10) & (warm["n"] <= 300)]
    with_alg = warm[warm["alg"].map(len) > 0]
    no_alg = warm[warm["alg"].map(len) == 0]
    pick = pd.concat([with_alg.sample(n_warm_allergy, random_state=seed), no_alg.sample(n_warm - n_warm_allergy, random_state=seed)])
    cold = prof[(prof["source"] == "foodcom_cold") & (prof["cond"].map(len) > 0)].sample(n_cold, random_state=seed)

    scenarios = []
    for kind, df in (("warm", pick), ("cold", cold)):
        for _, r in df.iterrows():
            i = len(scenarios)
            task = TASKS[i % len(TASKS)]
            liked = []
            if kind == "warm":
                hist = train[train["UserId"] == r["user_id"]].sort_values("DateSubmitted")["RecipeId"].tolist()
                liked = [html.unescape(names[x]) for x in hist[-3:]]
            scenarios.append({
                "id": f"s{i:02d}_{kind}", "kind": kind, "source_user": str(r["user_id"]),
                "conditions": r["cond"], "allergies": r["alg"],
                "taste": json.loads(r["taste_prefs"]), "liked_recipes": liked,
                "dislike": DISLIKES[int(rng.integers(len(DISLIKES)))],
                "time_limit": int(TIME_LIMITS[int(rng.integers(len(TIME_LIMITS)))]),
                **task,
            })
    return scenarios
