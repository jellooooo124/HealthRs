"""Reproducible *sample* search/re-ranking comparison with real Elasticsearch.

Ground truth: each warm user's held-out final positive interaction from the
existing temporal split (the val target is also masked at test time).
Queries and taste preferences use ONLY the user's TRAINING recipes. Never use
the held-out recipe's title/ingredients to construct the query.

This is a small-sample, train-derived-query benchmark, not an unbiased
measurement of live-user search quality or the original full-data results.

Prerequisites:
    HEALTHREC_DATA_CONFIG=data_sample python scripts/prepare_data.py
    HEALTHREC_DATA_CONFIG=data_sample HEALTHREC_OUTPUTS=outputs_search_bench \
        python scripts/train.py --name sample_bpr --set train.max_epochs=20 ...
    Docker Elasticsearch running.
Run:
    python -m scripts.benchmark_search --users 60 --out outputs/search_benchmark
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prototype_elasticsearch.index_recipes import index_catalog
from prototype_elasticsearch.query import build_query, retrieve
from prototype_elasticsearch.ranking import rank_candidates
from prototype_elasticsearch.settings import es_client
from src.config import get_project_root, load_config
from src.models.data import RecData
from src.models.trainer import load_checkpoint

STOP = {
    "the", "a", "an", "and", "for", "of", "with", "to", "in", "from",
    "easy", "best", "recipe", "simple", "quick", "homemade",
}
METHODS = ("BM25", "BM25 + taste/feedback", "BM25 + BPR + taste/feedback", "BPR-MF")


def training_query(titles: list[str]) -> tuple[str, list[str]]:
    """Only previously liked training titles; no validation/test text."""
    tokens: list[str] = []
    for title in titles[:3]:
        tokens.extend(word.lower() for word in re.findall(r"[A-Za-z]{4,}", str(title))
                      if word.lower() not in STOP)
    if not tokens:
        return "", []
    # Deterministic, maximum three distinct terms to reduce title-copy bias.
    unique = list(dict.fromkeys(tokens))
    return " ".join(unique[:3]), unique[:4]


def ranking_metrics(ranked: list[int], target: int, k: int = 10) -> tuple[float, float]:
    selected = ranked[:k]
    if target not in selected:
        return 0.0, 0.0
    rank = selected.index(target) + 1
    return 1.0 / math.log2(rank + 1), 1.0


def percentile95(latencies: list[float]) -> float:
    return float(np.percentile(latencies, 95)) if latencies else float("nan")


def benchmark(users: int, out: Path, index: str, candidate_size: int) -> dict:
    root = Path(get_project_root())
    processed = root / "data/processed_sample"
    ckpt = root / "outputs_search_bench/checkpoints/sample_bpr.pt"
    if not (processed / "interactions.parquet").exists():
        raise FileNotFoundError("Run prepare_data.py with HEALTHREC_DATA_CONFIG=data_sample")
    if not ckpt.exists():
        raise FileNotFoundError("Train sample_bpr checkpoint under outputs_search_bench/")
    data = RecData(str(processed), load_config("health_rules"))
    model = load_checkpoint(str(ckpt), data)
    es = es_client()
    assert es.ping(), "This benchmark requires a REAL running Elasticsearch instance"
    summary = index_catalog(source="processed-sample", index=index, recreate=True)
    assert summary["count_on_es"] == data.n_items, "ES/BPR catalogs must match exactly"

    rec = data.recipes
    recipe_ids = rec["RecipeId"].astype(int).to_numpy()
    id_to_item = {int(rid): int(i) for i, rid in enumerate(recipe_ids)}
    train = pd.read_parquet(processed / "interactions.parquet")
    train = train[train["split"] == "train"]
    by_user = defaultdict(list)
    for row in train[["user_idx", "item_idx"]].itertuples(index=False):
        by_user[int(row.user_idx)].append(int(row.item_idx))
    known = data.eval_mask("test")
    rng = np.random.default_rng(2026)
    sampled_users = rng.permutation(data.n_users).tolist()
    results = {name: {"ndcg": [], "recall": [], "latency_ms": []} for name in METHODS}
    cases = []
    # Cache catalog eligibility OUTSIDE per-query timing; never charge BPR latency
    # for hundreds of repeated ES document fetches.
    eligibility_docs = es.mget(index=index, ids=[str(int(rid)) for rid in recipe_ids])["docs"]
    allowed = np.array([bool(d.get("found") and d["_source"].get("nutrition_complete"))
                        for d in eligibility_docs], dtype=bool)
    skipped = {"incomplete_target": 0, "no_training_query": 0, "empty_pool": 0}
    k = 10

    for uid in sampled_users:
        if len(cases) >= users:
            break
        train_items = by_user.get(uid, [])
        if not train_items:
            skipped["no_training_query"] += 1
            continue
        query, prefs = training_query([rec.iloc[i]["Name"] for i in train_items[:3]])
        if not query:
            skipped["no_training_query"] += 1
            continue
        target_item = int(data.test_target[uid])
        target_id = int(recipe_ids[target_item])
        # The ES lexical query enforces this eligibility criterion; apply it
        # identically to the BPR-only baseline.
        raw = es.get(index=index, id=str(target_id))["_source"]
        if not raw["nutrition_complete"]:
            skipped["incomplete_target"] += 1
            continue
        seen_ids = {int(recipe_ids[i]) for i in known[uid].indices}
        qdsl = build_query(query, [], [], excluded_ids=seen_ids)

        def es_candidates():
            return retrieve(es, index, qdsl, candidate_size)

        # BM25 retrieval, no additional ranking.
        start = time.perf_counter()
        first = es_candidates()
        results["BM25"]["latency_ms"].append((time.perf_counter() - start) * 1000)
        raw_top = [int(d["recipe_id"]) for d in first[:k]]

        # Taste/feedback signals are simulated from TRAIN interactions only.
        likes = [int(recipe_ids[i]) for i in train_items[:3]]
        feedback = {rid: {"signal": "like"} for rid in likes}
        # ES already excludes train IDs; retrieve known likes directly
        # for category features, never from test targets.
        liked_docs = [x["_source"] for x in es.mget(index=index, ids=[str(x) for x in likes])["docs"]
                      if x.get("found")]
        profile = {"conditions": [], "allergies": [], "taste_prefs": prefs, "confirmed": True}

        start = time.perf_counter()
        taste_docs = es_candidates()
        taste_top = [int(d["recipe_id"]) for d in rank_candidates(
            taste_docs, profile, feedback, k=k, liked_docs=liked_docs, max_per_category=k)]
        results["BM25 + taste/feedback"]["latency_ms"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        hybrid_docs = es_candidates()
        with torch.no_grad():
            scores = model.full_scores(torch.tensor([uid]))[0].cpu().numpy()
        bpr = {int(d["recipe_id"]): float(scores[id_to_item[int(d["recipe_id"])]])
               for d in hybrid_docs}
        hybrid_top = [int(d["recipe_id"]) for d in rank_candidates(
            hybrid_docs, profile, feedback, k=k, bpr_scores=bpr,
            liked_docs=liked_docs, max_per_category=k)]
        results["BM25 + BPR + taste/feedback"]["latency_ms"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        with torch.no_grad():
            all_scores = model.full_scores(torch.tensor([uid]))[0].cpu().numpy().copy()
        all_scores[known[uid].indices] = -np.inf
        # Match the mandatory completeness filter, precomputed above.
        all_scores[~allowed] = -np.inf
        full_order = np.argsort(-all_scores, kind="stable")
        baseline_top = [int(recipe_ids[i]) for i in full_order[:k] if np.isfinite(all_scores[i])]
        results["BPR-MF"]["latency_ms"].append((time.perf_counter() - start) * 1000)

        for name, selected in zip(METHODS, (raw_top, taste_top, hybrid_top, baseline_top)):
            ndcg, recall = ranking_metrics(selected, target_id)
            results[name]["ndcg"].append(ndcg)
            results[name]["recall"].append(recall)
        cases.append({"user_idx": uid, "query_from_train": query, "test_target": target_id,
                      "retrieved_candidates": len(first)})

    if not cases:
        raise AssertionError("No eligible warm-user test queries; verify sample dataset and filters")
    aggregate = {
        name: {"ndcg@10": round(float(np.mean(values["ndcg"])), 6),
               "recall@10": round(float(np.mean(values["recall"])), 6),
               "p95_latency_ms": round(percentile95(values["latency_ms"]), 3),
               "mean_latency_ms": round(float(np.mean(values["latency_ms"])), 3)}
        for name, values in results.items()
    }
    result = {
        "dataset": "bundled Food.com sample with temporal test split",
        "query_generation": "up to three terms from each user's TRAIN recipe titles only",
        "relevance": "one held-out positive test recipe per warm user",
        "catalog_items": int(data.n_items),
        "n_queries": len(cases),
        "candidate_size": candidate_size,
        "seed": 2026,
        "model": "BPR-MF trained from scratch on sample training interactions",
        "health_conditions": "disabled for ranking relevance comparison; safety tested separately",
        "notes": [
            "Positive labels are historical ratings, not search-query judgements.",
            "One held-out positive per query: Recall@10 equals HitRate@10.",
            "BM25/hybrids retrieve at most candidate_size lexical matches; BPR scores full eligible catalog.",
            "The model is re-trained on the sample, not the original full-data serving checkpoint.",
            "Synthetic train-derived queries are not representative live search traffic.",
            "P95 includes live Elasticsearch calls for BM25/hybrids; BPR uses local CPU with precomputed eligibility.",
        ],
        "skipped": skipped, "results": aggregate, "cases": cases,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# HealthRec sample search benchmark", "",
        f"- Users / queries: {len(cases)}; catalog: {data.n_items}; seed: 2026; ES candidates: {candidate_size}",
        "- Ground truth: one withheld test positive per warm user; queries from **train titles only**",
        "- All runs use the same eligible catalog; BPR is trained on the sample train split",
        "", "| Method | NDCG@10 | Recall@10 | P95 latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    for name, vals in aggregate.items():
        lines.append(f"| {name} | {vals['ndcg@10']:.4f} | {vals['recall@10']:.4f} | {vals['p95_latency_ms']:.1f} |")
    lines.extend(["", "## Limitations", *["- " + note for note in result["notes"]], ""])
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=60)
    p.add_argument("--index", default="healthrec-search-benchmark")
    p.add_argument("--candidate-size", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("outputs/search_benchmark"))
    args = p.parse_args()
    assert 1 <= args.candidate_size <= 500
    benchmark(args.users, args.out, args.index, args.candidate_size)


if __name__ == "__main__":
    main()
