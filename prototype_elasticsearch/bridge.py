"""Optional bridge to EXISTING HealthRec BPR-MF + Adaptation A ranker."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def optional_bpr_scores(profile: dict, candidates: list[dict], enabled: bool = True) -> dict[int, float]:
    """Return {} for small-sample/cold installs; never replace model scores with invented values."""
    if not enabled or profile.get("source") != "foodcom_warm" or profile.get("user_idx") is None:
        return {}
    if not (ROOT / "data/processed/recipes.parquet").exists():
        return {}
    try:
        import numpy as np
        from agent.recommender_service import RecommenderService

        svc = RecommenderService.get()
        ctx = svc.rec._context(svc._user_idx(profile), profile.get("conditions", []), profile.get("allergies", []))
        mapped = [(int(d["recipe_id"]), svc.item_of[int(d["recipe_id"])])
                  for d in candidates if int(d["recipe_id"]) in svc.item_of]
        if not mapped:
            return {}
        scores = svc.rec.rank(ctx, np.array([idx for _, idx in mapped], dtype=np.int64))
        return {rid: float(score) for (rid, _), score in zip(mapped, scores)}
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        logger.warning("BPR unavailable; use BM25/preferences/popularity: %s", exc)
        return {}
