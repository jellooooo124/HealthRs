"""
Interaction preprocessing: ratings -> positives -> k-core -> temporal leave-one-out split,
plus a cold-start user set drawn from users that the k-core removed.
"""
import numpy as np
import pandas as pd


def build_positives(reviews: pd.DataFrame, recipe_ids: pd.Series, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split reviews on known recipes into positives (rating >= threshold) and observed non-positives (1..threshold-1)."""
    r = reviews[reviews["RecipeId"].isin(recipe_ids)]
    if cfg["drop_zero"]:
        r = r[r["Rating"] > 0]
    r = r.sort_values(["DateSubmitted", "ReviewId"]).drop_duplicates(["UserId", "RecipeId"], keep="last")
    pos = r[r["Rating"] >= cfg["positive_threshold"]].reset_index(drop=True)
    nonpos = r[r["Rating"] < cfg["positive_threshold"]].reset_index(drop=True)
    return pos, nonpos


def kcore(df: pd.DataFrame, k: int, user_col: str = "UserId", item_col: str = "RecipeId") -> pd.DataFrame:
    """Iteratively drop users and items with fewer than k interactions."""
    while True:
        u_cnt = df[user_col].map(df[user_col].value_counts())
        i_cnt = df[item_col].map(df[item_col].value_counts())
        keep = (u_cnt >= k) & (i_cnt >= k)
        if keep.all():
            return df.reset_index(drop=True)
        df = df[keep]


def temporal_leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    """Per user, by (DateSubmitted, ReviewId): last -> test, second last -> val, rest -> train."""
    df = df.sort_values(["UserId", "DateSubmitted", "ReviewId"]).reset_index(drop=True)
    rank_from_end = df.groupby("UserId").cumcount(ascending=False)
    df["split"] = np.select([rank_from_end == 0, rank_from_end == 1], ["test", "val"], "train")
    return df


def cold_start_users(pos_all: pd.DataFrame, warm_users: set, catalog: set, n: int, seed: int) -> pd.DataFrame:
    """
    Users outside the k-core (never seen in training) with >= 1 positive on catalog recipes.
    All their catalog positives are kept as test targets; `n` users are sampled with a fixed seed.
    """
    cold = pos_all[~pos_all["UserId"].isin(warm_users) & pos_all["RecipeId"].isin(catalog)]
    users = np.sort(cold["UserId"].unique())
    rng = np.random.default_rng(seed)
    picked = rng.choice(users, size=min(n, len(users)), replace=False)
    out = cold[cold["UserId"].isin(picked)].copy()
    out["split"] = "cold_test"
    return out.reset_index(drop=True), len(users)


def index_ids(inter: pd.DataFrame) -> tuple[dict, dict]:
    """Contiguous 0-based indices (sorted by raw id) for warm users and catalog items."""
    users = np.sort(inter["UserId"].unique())
    items = np.sort(inter["RecipeId"].unique())
    return {u: i for i, u in enumerate(users)}, {r: i for i, r in enumerate(items)}
