"""
Train one recommender and save its checkpoint + metrics.

    python scripts/train.py --name bpr_d64
    python scripts/train.py --name health_A_B1 --set health.use_features=true health.lambda_h=1.0

Outputs: outputs/checkpoints/<name>.pt, outputs/runs/<name>.json
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_project_root, load_config, get_outputs_dir  # noqa: E402
from src.eval.metrics import evaluate_ranking, summarize  # noqa: E402
from src.models.data import RecData  # noqa: E402
from src.models.trainer import train  # noqa: E402


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """--set a.b=value (value parsed as YAML)."""
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.dim=128")
    args = ap.parse_args()

    root = get_project_root()
    cfg = apply_overrides(load_config("model"), args.set)
    torch.set_num_threads(cfg["threads"])
    data_cfg = load_config("data")
    t0 = time.time()
    data = RecData(os.path.join(root, data_cfg["paths"]["processed_dir"]), load_config("health_rules"))
    print(f"[{args.name}] data loaded in {time.time() - t0:.0f}s; config: {cfg['train']} {cfg['health']}", flush=True)

    log = lambda msg: print(f"[{args.name}] {msg}", flush=True)  # noqa: E731
    model, info = train(cfg, data, log)

    k, dev = cfg["eval"]["k"], cfg["device"]
    val = evaluate_ranking(model.full_scores, data, "val", k, cfg["eval"]["chunk_users"], dev)
    test = evaluate_ranking(model.full_scores, data, "test", k, cfg["eval"]["chunk_users"], dev)
    log(f"best val:  {summarize(val)}")
    log(f"test:      {summarize(test)}")

    ckpt_dir, run_dir = get_outputs_dir("checkpoints"), get_outputs_dir("runs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"config": cfg, "state_dict": model.state_dict(), "model_class": type(model).__name__},
               os.path.join(ckpt_dir, f"{args.name}.pt"))
    with open(os.path.join(run_dir, f"{args.name}.json"), "w") as f:
        json.dump({"name": args.name, "config": cfg, "val": val, "test": test,
                   "train_seconds": round(time.time() - t0, 1), **info}, f, indent=2)


if __name__ == "__main__":
    main()
