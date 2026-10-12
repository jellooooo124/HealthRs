"""
Build all processed data from the raw Food.com files.

    python scripts/prepare_data.py            # reads configs/data.yaml + configs/health_rules.yaml

Outputs (data/processed/):
    recipes.parquet       catalog recipes (k-core items): metadata, nutrition, health labels, item_idx
    recipes_all.parquet   all cleaned recipes: nutrition + health labels only (for EDA)
    interactions.parquet  warm-user positives with split = train / val / test, user_idx, item_idx
    nonpos.parquet        warm-user ratings 1..3 on catalog recipes (excluded from negative sampling)
    cold_start.parquet    sampled cold-start users' positives on catalog recipes (split = cold_test)
    profiles.csv          synthetic health profiles for warm + sampled cold-start users
    summary.json          counts at every pipeline stage (used by scripts/data_report.py)
"""
import json
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_project_root, load_config  # noqa: E402
from src.data.foodcom import NUTRITION_COLS, clean_recipes, load_recipes, load_reviews  # noqa: E402
from src.data.health import label_recipes  # noqa: E402
from src.data.preprocess import build_positives, cold_start_users, index_ids, kcore, temporal_leave_one_out  # noqa: E402
from src.data.profiles import synthesize_profiles  # noqa: E402


def stats(df, user_col="UserId", item_col="RecipeId") -> dict:
    u, i = df[user_col].nunique(), df[item_col].nunique()
    return {"users": u, "items": i, "interactions": len(df), "density": len(df) / (u * i) if u and i else 0.0}


def main():
    t0 = time.time()
    root = get_project_root()
    cfg, rules = load_config("data"), load_config("health_rules")
    raw_dir = os.path.join(root, cfg["paths"]["raw_dir"])
    out_dir = os.path.join(root, cfg["paths"]["processed_dir"])
    os.makedirs(out_dir, exist_ok=True)
    summary = {"seed": cfg["seed"]}

    # 1. recipes: load, clean, label
    recipes = load_recipes(raw_dir)
    summary["raw_recipes"] = len(recipes)
    recipes, removed = clean_recipes(recipes, cfg["cleaning"])
    summary["cleaning_removed"] = removed
    summary["clean_recipes"] = len(recipes)
    labels = label_recipes(recipes, rules)
    recipes = recipes.join(labels)
    print(f"[recipes] raw={summary['raw_recipes']} clean={len(recipes)} ({time.time() - t0:.0f}s)")

    # 2. reviews -> positives / non-positives on clean recipes
    reviews = load_reviews(raw_dir)
    summary["raw_reviews"] = stats(reviews)
    summary["rating_counts"] = {int(k): int(v) for k, v in reviews["Rating"].value_counts().sort_index().items()}
    pos_all, nonpos_all = build_positives(reviews, recipes["RecipeId"], cfg["ratings"])
    summary["positives"] = stats(pos_all)

    # 3. k-core + temporal leave-one-out
    inter = temporal_leave_one_out(kcore(pos_all, cfg["kcore"]))
    summary["kcore"] = stats(inter)
    summary["split_sizes"] = {k: int(v) for k, v in inter["split"].value_counts().items()}
    user_idx, item_idx = index_ids(inter)
    inter["user_idx"] = inter["UserId"].map(user_idx)
    inter["item_idx"] = inter["RecipeId"].map(item_idx)
    print(f"[k-core] {summary['kcore']}")

    warm, catalog = set(user_idx), set(item_idx)
    nonpos = nonpos_all[nonpos_all["UserId"].isin(warm) & nonpos_all["RecipeId"].isin(catalog)].copy()
    nonpos["user_idx"] = nonpos["UserId"].map(user_idx)
    nonpos["item_idx"] = nonpos["RecipeId"].map(item_idx)

    # 4. cold-start users (outside the k-core, positives on catalog recipes)
    cold, n_cold_pool = cold_start_users(pos_all, warm, catalog, cfg["cold_start"]["n_users"], cfg["seed"])
    cold["item_idx"] = cold["RecipeId"].map(item_idx)
    summary["cold_start"] = {"pool_users": n_cold_pool, **stats(cold)}

    # 5. synthetic health profiles (warm + sampled cold users)
    recipe_info = recipes.set_index("RecipeId")
    all_users = list(warm) + list(cold["UserId"].unique())
    user_type = {u: "warm" for u in warm} | {u: "cold" for u in cold["UserId"].unique()}
    profiles = synthesize_profiles(
        users=all_users, user_type=user_type,
        positives=pos_all[pos_all["UserId"].isin(set(user_type))],
        train_pos=inter[inter["split"] == "train"],
        recipe_info=recipe_info, rules=rules, seed=cfg["seed"],
    )
    profiles.insert(1, "user_idx", profiles["user_id"].map(user_idx).astype("Int64"))

    # 6. save
    catalog_df = recipes[recipes["RecipeId"].isin(catalog)].copy()
    catalog_df.insert(0, "item_idx", catalog_df["RecipeId"].map(item_idx))
    catalog_df = catalog_df.sort_values("item_idx").reset_index(drop=True)
    catalog_df.to_parquet(os.path.join(out_dir, "recipes.parquet"), index=False)
    eda_cols = ["RecipeId", "RecipeCategory", "TotalMinutes", *NUTRITION_COLS, *labels.columns]
    recipes[eda_cols].to_parquet(os.path.join(out_dir, "recipes_all.parquet"), index=False)
    keep = ["UserId", "RecipeId", "user_idx", "item_idx", "Rating", "DateSubmitted", "ReviewId", "split"]
    inter[keep].to_parquet(os.path.join(out_dir, "interactions.parquet"), index=False)
    nonpos[keep[:-1]].to_parquet(os.path.join(out_dir, "nonpos.parquet"), index=False)
    cold[["UserId", "RecipeId", "item_idx", "Rating", "DateSubmitted", "ReviewId", "split"]].to_parquet(
        os.path.join(out_dir, "cold_start.parquet"), index=False)
    profiles.to_csv(os.path.join(out_dir, "profiles.csv"), index=False)
    summary["catalog_items"] = len(catalog_df)
    summary["nonpos_pairs"] = len(nonpos)
    summary["profiles"] = len(profiles)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
