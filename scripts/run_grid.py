"""
Run several training configs in parallel processes (each process uses `threads` torch threads).

    python scripts/run_grid.py --grid configs/grids/step3_base.yaml --jobs 3

Grid file format (`common` overrides are prepended to every run's `set`):
    common: [threads=2]
    runs:
      - name: bpr_d64
        set: [train.dim=64]
Logs go to outputs/logs/<name>.log; runs whose outputs/runs/<name>.json exists are skipped (use --force to rerun).
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import get_outputs_dir  # noqa: E402


def run_one(run: dict, common: list, force: bool) -> tuple[str, int, float]:
    name = run["name"]
    if not force and os.path.exists(get_outputs_dir("runs", f"{name}.json")):
        return name, 0, 0.0
    os.makedirs(get_outputs_dir("logs"), exist_ok=True)
    t0 = time.time()
    with open(get_outputs_dir("logs", f"{name}.log"), "w") as log:
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--name", name, "--set", *common, *run.get("set", [])]
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return name, code, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    grid = yaml.safe_load(open(args.grid))
    runs, common = grid["runs"], grid.get("common", [])
    print(f"{len(runs)} runs, {args.jobs} in parallel", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        for name, code, sec in pool.map(lambda r: run_one(r, common, args.force), runs):
            status = "ok" if code == 0 else f"FAILED (exit {code}, see outputs/logs/{name}.log)"
            print(f"{name}: {status} {sec / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
