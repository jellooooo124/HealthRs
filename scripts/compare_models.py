"""
Compare trained models and simple baselines on one split (full ranking).

    python scripts/compare_models.py --split val --runs bpr_d64_lr2e-3_reg1e-5 hA hAB_l1 \
        --post-base bpr_d64_lr2e-3_reg1e-5 --penalties 0.5 1 2 4

Rows: MostPop, HealthyPop, every --runs checkpoint, and for --post-base: + allergen filter,
+ post-filter (drop violating/allergen recipes), + soft penalty rerank (score − β·violates) for each β.
Output: printed table + outputs/tables/compare_<split>.csv
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_project_root, load_config, get_outputs_dir  # noqa: E402
from src.eval.metrics import evaluate_ranking, healthy_popularity_scores, popularity_scores  # noqa: E402
from src.models.data import RecData  # noqa: E402
from src.models.trainer import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--post-base", default=None, help="run whose scores get the post-filter / penalty variants")
    ap.add_argument("--penalties", nargs="*", type=float, default=[])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    root = get_project_root()
    data = RecData(os.path.join(root, load_config("data")["paths"]["processed_dir"]), load_config("health_rules"))
    ckpt = lambda name: get_outputs_dir("checkpoints", f"{name}.pt")  # noqa: E731
    ev = lambda fn, **kw: evaluate_ranking(fn, data, args.split, **kw)  # noqa: E731

    rows = [{"model": "MostPop", **ev(popularity_scores(data))},
            {"model": "HealthyPop", **ev(healthy_popularity_scores(data))}]
    for name in args.runs:
        model = load_checkpoint(ckpt(name), data)
        rows.append({"model": name, **ev(model.full_scores)})
        print(f"evaluated {name}", flush=True)
    if args.post_base:
        model = load_checkpoint(ckpt(args.post_base), data)
        rows.append({"model": f"{args.post_base} + allergen filter", **ev(model.full_scores, allergen_filter=True)})
        rows.append({"model": f"{args.post_base} + post-filter", **ev(model.full_scores, postfilter=True)})
        for b in args.penalties:
            rows.append({"model": f"{args.post_base} + penalty β={b:g}", **ev(model.full_scores, penalty=b, allergen_filter=True)})

    df = pd.DataFrame(rows)
    cols = ["model", "ndcg@10", "hr@10", "violation@10", "user_violation@10", "allergen@10", "coverage@10"]
    out = args.out or get_outputs_dir("tables", f"compare_{args.split}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df[cols].to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 50):
        print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
