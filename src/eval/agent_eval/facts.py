"""Ground-truth recipe facts for the agent evaluation (catalog lookups, no LLM)."""
import html
import os
import re

import numpy as np
import pandas as pd

from src.config import get_project_root, load_config
from src.data.health import build_matchers, recipe_texts

CITE = re.compile(r"#(\d{2,7})\b")


class RecipeFacts:
    def __init__(self):
        root = get_project_root()
        rules = load_config("health_rules")
        self.conditions, self.allergens = list(rules["conditions"]), list(rules["allergens"])
        df = pd.read_parquet(os.path.join(root, load_config("data")["paths"]["processed_dir"], "recipes.parquet"))
        self.df = df.set_index("RecipeId")
        self.text = pd.Series(recipe_texts(df).map(lambda xs: " | ".join(xs).lower()).to_numpy(), index=df["RecipeId"])

    def known(self, rid: int) -> bool:
        return rid in self.df.index

    def check(self, rid: int, conditions: list, allergies: list, dislike: str | None = None,
              time_limit: float | None = None) -> dict:
        if not self.known(rid):
            return {"recipe_id": rid, "known": False}
        r = self.df.loc[rid]
        minutes = None if np.isnan(r["TotalMinutes"]) else float(r["TotalMinutes"])
        dis = bool(dislike) and bool(build_matchers({"d": {"include": [dislike, dislike.rstrip("s")]}})["d"].matches(self.text[rid]))
        return {"recipe_id": int(rid), "known": True, "name": html.unescape(r["Name"]), "category": r["RecipeCategory"],
                "minutes": minutes,
                "nutrition_per_serving": {"kcal": round(r["Calories"], 1), "carbs_g": round(r["CarbohydrateContent"], 1),
                                          "sugar_g": round(r["SugarContent"], 1), "sodium_mg": round(r["SodiumContent"], 1),
                                          "sat_fat_g": round(r["SaturatedFatContent"], 1), "protein_g": round(r["ProteinContent"], 1)},
                "violations": {c: r[f"viol_{c}"] for c in conditions if not r[f"ok_{c}"]},
                "allergens": [a for a in allergies if r[f"contains_{a}"]],
                "contains_dislike": dis, "over_time": bool(time_limit and minutes and minutes > time_limit)}

    def misattributed(self, text: str) -> list[tuple]:
        """Cited "**Name** (#id)" pairs whose name is not (a shortening of) the catalog name of that id."""
        bad = []
        for name, rid in re.findall(r"\*\*([^*]{3,120})\*\*\s*\(#(\d{2,7})\)", text or ""):
            rid = int(rid)
            if not self.known(rid):
                continue
            cited, true = re.sub(r"\W+", " ", name.lower()).strip(), re.sub(r"\W+", " ", html.unescape(self.df.loc[rid, "Name"]).lower()).strip()
            if cited not in true and true not in cited:
                bad.append((name, rid, html.unescape(self.df.loc[rid, "Name"])))
        return bad

    def cited(self, text: str) -> list[int]:
        return list(dict.fromkeys(int(x) for x in CITE.findall(text or "")))
