"""
Load and clean the raw Food.com (Kaggle irkaal/foodcom-recipes-and-reviews) CSV files.

Raw quirks handled here:
  * list columns are R literals, e.g. c("blueberries", "granulated sugar") -> Python list
  * times are ISO-8601 durations, e.g. PT2H35M -> minutes
  * nutrition is per serving (kcal; sodium/cholesterol in mg; others in g) with some
    data-entry outliers that are removed by `clean_recipes`
"""
import os
import re

import numpy as np
import pandas as pd

NUTRITION_COLS = [
    "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent", "SodiumContent",
    "CarbohydrateContent", "FiberContent", "SugarContent", "ProteinContent",
]
LIST_COLS = ["Images", "Keywords", "RecipeIngredientQuantities", "RecipeIngredientParts"]
RECIPE_COLS = [
    "RecipeId", "Name", "DatePublished", "CookTime", "PrepTime", "TotalTime", "Images",
    "RecipeCategory", "Keywords", "RecipeIngredientParts", "AggregatedRating", "ReviewCount",
    *NUTRITION_COLS, "RecipeServings", "Description", "RecipeIngredientQuantities", "RecipeInstructions",
]

_R_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
_DURATION = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def parse_r_list(value) -> list:
    """c("a", "b") -> ["a", "b"]; a bare "a" -> ["a"]; NA / character(0) -> []."""
    if not isinstance(value, str):
        return []
    return [s.strip() for s in _R_STRING.findall(value) if s.strip() and s.strip() != "NA"]


_R_ITEM = re.compile(r'"((?:[^"\\]|\\.)*)"|\bNA\b')


def parse_r_list_aligned(value) -> list:
    """Like parse_r_list but keeps positions: bare NA -> "" (ingredient quantities align with parts)."""
    if not isinstance(value, str):
        return []
    return [(m.group(1) or "").strip() for m in _R_ITEM.finditer(value)]


def parse_duration_minutes(value) -> float:
    """ISO-8601 duration (PT1H30M) -> minutes; NaN if missing/unparseable."""
    if not isinstance(value, str):
        return np.nan
    m = _DURATION.fullmatch(value.strip())
    if not m or not any(m.groups()):
        return np.nan
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 60 + mi + s / 60


def load_recipes(raw_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(raw_dir, "recipes.csv"), usecols=RECIPE_COLS)
    for col in ("Images", "Keywords", "RecipeIngredientParts", "RecipeInstructions"):
        df[col] = df[col].map(parse_r_list)
    df["RecipeIngredientQuantities"] = df["RecipeIngredientQuantities"].map(parse_r_list_aligned)
    df["ImageUrl"] = df.pop("Images").map(lambda xs: xs[0] if xs else None)
    for col in ("CookTime", "PrepTime", "TotalTime"):
        df[col.replace("Time", "Minutes")] = df.pop(col).map(parse_duration_minutes)
    df["DatePublished"] = pd.to_datetime(df["DatePublished"], utc=True)
    return df


def load_reviews(raw_dir: str) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(raw_dir, "reviews.csv"),
        usecols=["ReviewId", "RecipeId", "AuthorId", "Rating", "DateSubmitted"],
    )
    df["DateSubmitted"] = pd.to_datetime(df["DateSubmitted"], utc=True)
    return df.rename(columns={"AuthorId": "UserId"})


def clean_recipes(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Drop recipes with implausible per-serving nutrition. Returns (clean_df, removal counts)."""
    sodium_cap = df["SodiumContent"].quantile(cfg["sodium_quantile"])
    rules = {
        "zero_calories": df["Calories"] <= cfg["min_calories"],
        "calories_gt_max": df["Calories"] > cfg["max_calories"],
        "sodium_outlier": df["SodiumContent"] > sodium_cap,
        "no_ingredients": df["RecipeIngredientParts"].map(len) == 0,
    }
    bad = np.zeros(len(df), dtype=bool)
    removed = {}
    for name, mask in rules.items():
        removed[name] = int((mask & ~bad).sum())   # counted in order, no double counting
        bad |= mask.to_numpy()
    removed["sodium_cap_mg"] = float(round(sodium_cap, 1))
    return df[~bad].reset_index(drop=True), removed
