"""
Build the small sample shipped with the repository (data/sample/) from the full Food.com files.

    python scripts/make_sample.py            # needs data/raw/foodcom/{recipes,reviews}.csv

Selection (deterministic, seed from configs/data.yaml): the 1,000 most-reviewed recipes form a dense block;
500 users with >= 10 reviews inside the block are sampled, plus 150 users with 1-4 reviews (cold-start pool).
Files keep the raw Kaggle format so the normal pipeline runs unchanged (configs/data_sample.yaml);
the free-text review column is dropped.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_project_root, load_config  # noqa: E402

N_RECIPES, N_USERS, N_COLD = 1000, 500, 150


def main():
    root = get_project_root()
    cfg = load_config("data")
    raw = os.path.join(root, cfg["paths"]["raw_dir"])
    out = os.path.join(root, "data", "sample")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    reviews = pd.read_csv(os.path.join(raw, "reviews.csv"),
                          usecols=["ReviewId", "RecipeId", "AuthorId", "Rating", "DateSubmitted"])
    block = reviews["RecipeId"].value_counts().index[:N_RECIPES]
    inb = reviews[reviews["RecipeId"].isin(block)]
    per_user = inb.groupby("AuthorId").size()
    active = np.sort(per_user[per_user >= 10].index.to_numpy())
    light = np.sort(per_user[per_user <= 4].index.to_numpy())
    users = np.concatenate([rng.choice(active, N_USERS, replace=False), rng.choice(light, N_COLD, replace=False)])
    sample_reviews = inb[inb["AuthorId"].isin(users)].sort_values("ReviewId")

    recipes = pd.read_csv(os.path.join(raw, "recipes.csv"), dtype=str, keep_default_na=False)
    sample_recipes = recipes[recipes["RecipeId"].astype(int).isin(sample_reviews["RecipeId"].unique())]

    sample_recipes.to_csv(os.path.join(out, "recipes.csv"), index=False)
    sample_reviews.to_csv(os.path.join(out, "reviews.csv"), index=False)
    print(f"sample: {len(sample_recipes)} recipes, {len(sample_reviews)} reviews, "
          f"{sample_reviews['AuthorId'].nunique()} users -> {out}")


if __name__ == "__main__":
    main()
