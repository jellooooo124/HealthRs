"""Training loop: fresh negatives (and health pairs) every epoch, early stopping on val NDCG@K."""
import copy
import time

import numpy as np
import torch

from src.eval.metrics import evaluate_ranking, summarize
from src.models.bpr_mf import BPRMF
from src.models.health_bpr import HealthBPR


def build_model(cfg: dict, data) -> BPRMF:
    t, h = cfg["train"], cfg["health"]
    bias = t.get("item_bias", False)
    if not h["use_features"] and h["lambda_h"] == 0:
        return BPRMF(data.n_users, data.n_items, t["dim"], item_bias=bias)
    return HealthBPR(data.n_users, data.n_items, t["dim"],
                     item_feat=torch.as_tensor(data.item_feat), user_cond=torch.as_tensor(data.user_cond),
                     use_features=h["use_features"], lambda_h=h["lambda_h"], item_bias=bias)


def load_checkpoint(path: str, data) -> BPRMF:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], data)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def train(cfg: dict, data, log=print) -> tuple[BPRMF, dict]:
    t, h = cfg["train"], cfg["health"]
    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = build_model(cfg, data).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    k, use_health = cfg["eval"]["k"], h["lambda_h"] > 0

    history, best, best_state, bad_evals = [], -1.0, None, 0
    for epoch in range(1, t["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        u, i, j = data.sample_epoch(rng)
        if use_health:
            hg, hb, hv = data.sample_health_pairs(u, i, rng, h["health_neg"])
        sums, n_batches = {}, 0
        for s in range(0, len(u), t["batch_size"]):
            sl = slice(s, s + t["batch_size"])
            batch = {"u": torch.as_tensor(u[sl], device=device), "i": torch.as_tensor(i[sl], device=device),
                     "j": torch.as_tensor(j[sl], device=device)}
            if use_health:
                batch |= {"h_good": torch.as_tensor(hg[sl], device=device), "h_bad": torch.as_tensor(hb[sl], device=device),
                          "h_valid": torch.as_tensor(hv[sl], device=device)}
            loss, logs = model.loss(batch, t["reg"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            for key, val in logs.items():
                sums[key] = sums.get(key, 0.0) + val
            n_batches += 1
        rec = {"epoch": epoch, "sec": round(time.time() - t0, 2), **{key: v / n_batches for key, v in sums.items()}}

        if epoch % t["eval_every"] == 0 or epoch == t["max_epochs"]:
            model.eval()
            m = evaluate_ranking(model.full_scores, data, "val", k, cfg["eval"]["chunk_users"], device)
            rec |= {f"val_{key}": v for key, v in m.items()}
            log(f"epoch {epoch:3d}  loss {rec.get('bpr', 0):.4f}  health {rec.get('health', 0):.4f}  {summarize(m)}  ({rec['sec']}s/epoch)")
            if m[f"ndcg@{k}"] > best:
                best, best_state, bad_evals = m[f"ndcg@{k}"], copy.deepcopy(model.state_dict()), 0
                rec["best"] = True
            else:
                bad_evals += 1
            history.append(rec)
            if bad_evals >= t["patience"]:
                log(f"early stop at epoch {epoch}; best val ndcg@{k}={best:.4f}")
                break
        else:
            history.append(rec)

    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_val_ndcg": best}
