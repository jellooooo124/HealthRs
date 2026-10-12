"""
Evaluation of the two-stage recommender (src/models/retrieval.py) exactly as deployed.

* channel_ranks: per warm user, the 0-based rank of the held-out target inside each retrieval
  channel (taste = plain BPR-MF scores; healthy = popularity among recipes compliant with the
  user's conditions), both over allowed items (not seen, no allergen). Gives Recall@N curves.
* evaluate_pipeline / evaluate_pipeline_cold: run retrieval -> ranking -> rerank per user and
  score the final top-K (plus candidate recall and latency).
"""
import math
import time

import numpy as np
import torch


@torch.no_grad()
def channel_ranks(data, taste_model, split: str = "test", chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    target = torch.tensor(getattr(data, f"{split}_target"))
    mask = data.eval_mask(split)
    not_ok = torch.as_tensor(~data.item_ok, dtype=torch.float32)
    item_alg = torch.as_tensor(data.item_allergen, dtype=torch.float32)
    user_cond, user_alg = torch.as_tensor(data.user_cond), torch.as_tensor(data.user_alg)
    pop_order = np.argsort(-data.item_pop, kind="stable")
    posn = torch.empty(data.n_items, dtype=torch.long)
    posn[torch.as_tensor(pop_order)] = torch.arange(data.n_items)

    taste, healthy = [], []
    for start in range(0, data.n_users, chunk):
        users = torch.arange(start, min(start + chunk, data.n_users))
        seen = torch.zeros(len(users), data.n_items, dtype=torch.bool)
        sub = mask[start:start + len(users)].tocoo()
        seen[torch.as_tensor(sub.row), torch.as_tensor(sub.col)] = True
        allowed = ~seen & ~((user_alg[users] @ item_alg.T) > 0)
        t = target[users]

        s = taste_model.full_scores(users).float().masked_fill(~allowed, -math.inf)
        ts = s.gather(1, t[:, None])
        r = (s > ts).sum(1).float()
        taste.append(torch.where(torch.isinf(ts[:, 0]), torch.full_like(r, math.inf), r))

        h_ok = allowed & ~((user_cond[users] @ not_ok.T) > 0)
        hr = (h_ok & (posn[None, :] < posn[t][:, None])).sum(1).float()
        healthy.append(torch.where(h_ok.gather(1, t[:, None])[:, 0], hr, torch.full_like(hr, math.inf)))
    return torch.cat(taste).numpy(), torch.cat(healthy).numpy()


def _topk_health(data, top, cond_row, alg_row):
    if not top:
        return np.nan, np.nan
    viol = (~data.item_ok[top][:, cond_row.astype(bool)]).any(1).mean() if cond_row.any() else np.nan
    alg = data.item_allergen[top][:, alg_row.astype(bool)].any(1).mean() if alg_row.any() else np.nan
    return viol, alg


def evaluate_pipeline(rec, data, split: str = "test", k: int = 10) -> dict:
    target = getattr(data, f"{split}_target")
    discount = 1.0 / np.log2(np.arange(k) + 2)
    ndcg, hit, cand_hit, n_cands, viol, alg, ms, seen_items = [], [], [], [], [], [], [], set()
    for u in range(data.n_users):
        ctx = rec._context(u, None, None)
        extra = np.array([data.val_target[u]]) if split == "test" else None
        t0 = time.perf_counter()
        allowed = rec.allowed_mask(ctx, None, extra_exclude=extra)
        cands, _, _, top = rec.run(ctx, allowed, k)
        ms.append((time.perf_counter() - t0) * 1000)
        tgt = int(target[u])
        pos = top.index(tgt) if tgt in top else -1
        hit.append(pos >= 0)
        ndcg.append(discount[pos] if pos >= 0 else 0.0)
        cand_hit.append(tgt in set(cands.tolist()))
        n_cands.append(len(cands))
        v, a = _topk_health(data, top, data.user_cond[u], data.user_alg[u])
        viol.append(v)
        alg.append(a)
        seen_items.update(top)
    return {
        f"ndcg@{k}": float(np.mean(ndcg)), f"hr@{k}": float(np.mean(hit)), f"recall@{k}": float(np.mean(hit)),
        f"violation@{k}": float(np.nanmean(viol)), f"allergen@{k}": float(np.nanmean(alg)) if not np.all(np.isnan(alg)) else float("nan"),
        f"coverage@{k}": len(seen_items) / data.n_items,
        "candidate_recall": float(np.mean(cand_hit)), "avg_candidates": float(np.mean(n_cands)),
        "latency_ms_mean": float(np.mean(ms)), "latency_ms_p95": float(np.percentile(ms, 95)),
    }


def evaluate_pipeline_cold(rec, data, k: int = 10) -> dict:
    discount = 1.0 / np.log2(np.arange(k) + 2)
    ndcg, hit, rec_k, cand_hit, viol, alg, ms = [], [], [], [], [], [], []
    for row in range(len(data.cold_users)):
        conds = [c for c, on in zip(data.conditions, data.cold_cond[row]) if on]
        algs = [a for a, on in zip(data.allergens, data.cold_alg[row]) if on]
        ctx = rec._context(None, conds, algs)
        t0 = time.perf_counter()
        cands, _, _, top = rec.run(ctx, rec.allowed_mask(ctx, None), k)
        ms.append((time.perf_counter() - t0) * 1000)
        targets = set(data.cold_targets[row].indices.tolist())
        rel = np.array([i in targets for i in top] + [False] * (k - len(top)), dtype=float)
        hit.append(rel.any())
        rec_k.append(rel.sum() / min(k, len(targets)))
        ndcg.append((rel * discount).sum() / discount[: min(k, len(targets))].sum())
        cand_hit.append(bool(targets & set(cands.tolist())))
        v, a = _topk_health(data, top, data.cold_cond[row], data.cold_alg[row])
        viol.append(v)
        alg.append(a)
    return {
        f"ndcg@{k}": float(np.mean(ndcg)), f"hr@{k}": float(np.mean(hit)), f"recall@{k}": float(np.mean(rec_k)),
        f"violation@{k}": float(np.nanmean(viol)), f"allergen@{k}": float(np.nanmean(alg)) if not np.all(np.isnan(alg)) else float("nan"),
        "candidate_recall": float(np.mean(cand_hit)), "latency_ms_mean": float(np.mean(ms)),
    }
