"""
Inspect the two-stage recommender on a few users (warm, cold-start, with constraints).

    python scripts/demo_recommend.py --taste bpr_d64_lr2e-3_reg1e-5 --health hAB_l1
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_project_root, load_config, get_outputs_dir  # noqa: E402
from src.models.data import RecData  # noqa: E402
from src.models.retrieval import TwoStageRecommender  # noqa: E402
from src.models.trainer import load_checkpoint  # noqa: E402


def show(title: str, res: dict, ms: float):
    print(f"\n=== {title}  ({res['n_candidates']} candidates, {ms:.0f} ms)")
    print(f"    user: {res['user']}")
    for r in res["recipes"]:
        n = r["nutrition_per_serving"]
        viol = "; ".join(f"{c}: {v}" for c, v in r["violations"].items()) or "complies"
        mins = f"{r['total_minutes']:.0f} min" if r["total_minutes"] is not None else "? min"
        print(f"  - {r['name'][:42]:42s} {r['category'][:16]:16s} {mins:>8s}  sodium {n['Sodium']:>10s} "
              f"carbs {n['Carbohydrate']:>8s} sugar {n['Sugar']:>8s}  [{r['retrieved_by']}] {viol}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taste", required=True)
    ap.add_argument("--health", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    root = get_project_root()
    data = RecData(os.path.join(root, load_config("data")["paths"]["processed_dir"]), load_config("health_rules"))
    ck = lambda n: get_outputs_dir("checkpoints", f"{n}.pt")  # noqa: E731
    rec = TwoStageRecommender(data, load_checkpoint(ck(args.taste), data), load_checkpoint(ck(args.health), data),
                              load_config("model")["retrieval"])

    def run(title, **kw):
        t0 = time.time()
        res = rec.recommend(**kw)
        show(title, res, (time.time() - t0) * 1000)
        return res

    # a warm user with hypertension + another condition, and one with an allergy
    both = np.flatnonzero((data.user_cond.sum(1) >= 2) & (data.train_csr.getnnz(1) >= 20))[0]
    allergic = np.flatnonzero(data.user_alg[:, data.allergens.index("peanut")] > 0)[0]
    run("warm user, own profile", user_idx=int(both), k=8)
    run("warm user, own profile, strict health", user_idx=int(both), k=8, strict_health=True)
    res = run("warm user with peanut allergy", user_idx=int(allergic), k=8)
    assert not any(data.item_allergen[data.recipes.index[data.recipes['RecipeId'] == r['recipe_id']][0],
                                      data.allergens.index("peanut")] for r in res["recipes"])
    run("cold-start: diabetes + shellfish allergy, chicken under 45 min", conditions=["diabetes"],
        allergies=["shellfish"], constraints={"include_ingredients": ["chicken"], "max_minutes": 45}, k=8)
    run("cold-start: no conditions, dessert", constraints={"categories": ["dessert"]}, k=5)


if __name__ == "__main__":
    main()
