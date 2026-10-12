"""
Full-ranking evaluation (every catalog item is a candidate; train positives are masked).

Accuracy (temporal leave-one-out, one target per user):
    HR@K / Recall@K (identical with a single target), NDCG@K = 1 / log2(rank + 2) if rank < K
Health (our metrics, thresholds from configs/health_rules.yaml):
    violation@K  share of top-K items violating ≥ 1 of the user's conditions (users with ≥ 1 condition)
    allergen@K   share of top-K items containing one of the user's allergens (users with ≥ 1 allergy)
Optional score post-processing (baselines):
    penalty  x_ui − β · violates(u, i)          ("BPR-MF + soft rerank")
    filter   x_ui = −inf if violates or allergen  ("BPR-MF + post-filter")
"""
import math

import numpy as np
import torch


@torch.no_grad()
def evaluate_ranking(score_fn, data, split: str = "val", k: int = 10, chunk: int = 1024, device="cpu",
                     penalty: float = 0.0, postfilter: bool = False, allergen_filter: bool = False) -> dict:
    """
    score_fn(user_idx LongTensor) -> (B, n_items) scores.  Returns aggregated metrics as floats.
    """
    target = torch.tensor(getattr(data, f"{split}_target"), device=device)
    mask = data.eval_mask(split)
    not_ok = torch.as_tensor(~data.item_ok, dtype=torch.float32, device=device)            # (n_items, n_cond)
    item_alg = torch.as_tensor(data.item_allergen, dtype=torch.float32, device=device)       # (n_items, n_alg)
    user_cond = torch.as_tensor(data.user_cond, device=device)
    user_alg = torch.as_tensor(data.user_alg, device=device)
    discount = 1.0 / torch.log2(torch.arange(k, device=device, dtype=torch.float32) + 2)

    ndcg, hit, viol, alg, viol_any, items_seen = [], [], [], [], [], set()
    for start in range(0, data.n_users, chunk):
        users = torch.arange(start, min(start + chunk, data.n_users), device=device)
        s = score_fn(users).float()
        sub = mask[start:start + len(users)].tocoo()
        s[torch.as_tensor(sub.row, device=device), torch.as_tensor(sub.col, device=device)] = -math.inf

        v = (user_cond[users] @ not_ok.T) > 0      # (B, n_items) violates any of the user's conditions
        a = (user_alg[users] @ item_alg.T) > 0     # (B, n_items) contains one of the user's allergens
        if penalty:
            s = s - penalty * v.float()
        if postfilter:
            s = s.masked_fill(v, -math.inf)
        if postfilter or allergen_filter:
            s = s.masked_fill(a, -math.inf)

        t = target[users]
        t_score = s.gather(1, t[:, None])
        rank = (s > t_score).sum(1)
        rank = torch.where(torch.isinf(t_score[:, 0]), torch.full_like(rank, data.n_items), rank)
        h = rank < k
        hit.append(h.float())
        ndcg.append(torch.where(h, discount[rank.clamp(max=k - 1)], 0.0))

        top = s.topk(k, dim=1).indices
        has_c = user_cond[users].sum(1) > 0
        has_a = user_alg[users].sum(1) > 0
        vk = v.gather(1, top).float()
        viol.append(vk.mean(1)[has_c])
        viol_any.append((vk.sum(1) > 0).float()[has_c])
        alg.append(a.gather(1, top).float().mean(1)[has_a])
        items_seen.update(top.flatten().tolist())

    cat = lambda xs: torch.cat(xs).mean().item() if sum(len(x) for x in xs) else float("nan")  # noqa: E731
    hr = cat(hit)
    return {
        f"ndcg@{k}": cat(ndcg), f"hr@{k}": hr, f"recall@{k}": hr,
        f"violation@{k}": cat(viol), f"user_violation@{k}": cat(viol_any), f"allergen@{k}": cat(alg),
        f"coverage@{k}": len(items_seen) / data.n_items,
    }


def popularity_scores(data, device="cpu"):
    """Score function for the most-popular baseline (train counts)."""
    pop = torch.as_tensor(data.item_pop, dtype=torch.float32, device=device)
    return lambda users: pop.expand(len(users), -1).clone()


def healthy_popularity_scores(data, device="cpu"):
    """Most-popular among recipes compliant with the user's conditions (violating items pushed to the bottom)."""
    pop = torch.as_tensor(data.item_pop, dtype=torch.float32, device=device)
    not_ok = torch.as_tensor(~data.item_ok, dtype=torch.float32, device=device)
    user_cond = torch.as_tensor(data.user_cond, device=device)
    big = float(pop.max()) + 1

    def fn(users):
        v = ((user_cond[users] @ not_ok.T) > 0).float()
        return pop.expand(len(users), -1) - big * v
    return fn


def summarize(metrics: dict, keys=("ndcg@10", "hr@10", "violation@10", "allergen@10")) -> str:
    return "  ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics and not np.isnan(metrics[k]))


@torch.no_grad()
def evaluate_cold(score_fn, data, k: int = 10, device="cpu", penalty: float = 0.0, postfilter: bool = False,
                  allergen_filter: bool = False) -> dict:
    """
    Cold-start users (no training history; several targets each).
    score_fn(cond FloatTensor (B, n_cond)) -> (B, n_items).
      HR@K      share of users with >= 1 target in the top-K
      Recall@K  hits / min(K, #targets)
      NDCG@K    DCG / ideal DCG over the user's targets
    """
    cond = torch.as_tensor(data.cold_cond, device=device)
    alg_u = torch.as_tensor(data.cold_alg, device=device)
    not_ok = torch.as_tensor(~data.item_ok, dtype=torch.float32, device=device)
    item_alg = torch.as_tensor(data.item_allergen, dtype=torch.float32, device=device)
    tgt = data.cold_targets
    discount = 1.0 / torch.log2(torch.arange(k, device=device, dtype=torch.float32) + 2)

    tops = []
    for start in range(0, len(cond), 1000):           # chunked: (users x items) score matrices are large
        c = cond[start:start + 1000]
        s = score_fn(c).float()
        v = (c @ not_ok.T) > 0
        if penalty:
            s = s - penalty * v.float()
        if postfilter:
            s = s.masked_fill(v, -math.inf)
        if postfilter or allergen_filter:
            s = s.masked_fill((alg_u[start:start + 1000] @ item_alg.T) > 0, -math.inf)
        tops.append(s.topk(k, dim=1).indices)
    top = torch.cat(tops).cpu().numpy()

    hr, rec, ndcg = [], [], []
    for row in range(len(top)):
        targets = set(tgt[row].indices.tolist())
        rel = np.array([i in targets for i in top[row]], dtype=np.float32)
        hr.append(rel.any())
        rec.append(rel.sum() / min(k, len(targets)))
        idcg = discount[: min(k, len(targets))].sum().item()
        ndcg.append(float((rel * discount.cpu().numpy()).sum()) / idcg)
    top_t = torch.as_tensor(top, device=device)
    has_c, has_a = cond.sum(1) > 0, alg_u.sum(1) > 0
    vk = (cond[:, None, :] * not_ok[top_t]).sum(-1).gt(0).float()      # (users, k) violates any condition
    ak = (alg_u[:, None, :] * item_alg[top_t]).sum(-1).gt(0).float()    # (users, k) contains an allergen
    return {
        f"ndcg@{k}": float(np.mean(ndcg)), f"hr@{k}": float(np.mean(hr)), f"recall@{k}": float(np.mean(rec)),
        f"violation@{k}": vk.mean(1)[has_c].mean().item() if has_c.any() else float("nan"),
        f"user_violation@{k}": (vk.sum(1) > 0).float()[has_c].mean().item() if has_c.any() else float("nan"),
        f"allergen@{k}": ak.mean(1)[has_a].mean().item() if has_a.any() else float("nan"),
        f"coverage@{k}": len(np.unique(top)) / data.n_items,
    }


def cold_score_fn(model, device="cpu"):
    """Cold-start scores of a trained model: Σ condition embeddings (adaptation A) · items + item bias."""
    items = model.item_vec().detach()
    bias = model.bias().detach()
    if hasattr(model, "cold_user_vec"):
        return lambda cond: model.cold_user_vec(cond) @ items.T + bias
    return lambda cond: bias.expand(len(cond), -1).clone()    # plain BPR-MF: no user vector -> item bias only


def cold_popularity_scores(data, healthy: bool = False, device="cpu"):
    pop = torch.as_tensor(data.item_pop, dtype=torch.float32, device=device)
    not_ok = torch.as_tensor(~data.item_ok, dtype=torch.float32, device=device)
    big = float(pop.max()) + 1

    def fn(cond):
        s = pop.expand(len(cond), -1).clone()
        return s - big * ((cond @ not_ok.T) > 0).float() if healthy else s
    return fn


def _zscore_log_pop(data, device="cpu") -> torch.Tensor:
    lp = torch.log1p(torch.as_tensor(data.item_pop, dtype=torch.float32, device=device))
    return (lp - lp.mean()) / lp.std()


def blend_scores(model, data, alpha: float, device="cpu"):
    """Popularity-calibrated scores: per-user z-scored model scores + α · z-scored log(1 + train popularity)."""
    lp = _zscore_log_pop(data, device)

    def fn(users):
        s = model.full_scores(users)
        return (s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True) + alpha * lp
    return fn


def cold_blend_fn(model, data, alpha: float, device="cpu"):
    base, lp = cold_score_fn(model, device), _zscore_log_pop(data, device)

    def fn(cond):
        s = base(cond)
        sd = s.std(1, keepdim=True)
        return (s - s.mean(1, keepdim=True)) / torch.where(sd > 0, sd, torch.ones_like(sd)) + alpha * lp
    return fn
