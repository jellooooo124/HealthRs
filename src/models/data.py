"""
In-memory training/evaluation data built from data/processed/ (see scripts/prepare_data.py).

Index conventions: users and items use the contiguous `user_idx` / `item_idx` from preprocessing.
Conditions and allergens follow the order in configs/health_rules.yaml.
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.foodcom import NUTRITION_COLS


class RecData:
    def __init__(self, processed_dir: str, rules: dict):
        self.conditions = list(rules["conditions"])
        self.allergens = list(rules["allergens"])

        inter = pd.read_parquet(os.path.join(processed_dir, "interactions.parquet"))
        self.n_users = int(inter["user_idx"].max()) + 1
        self.n_items = int(inter["item_idx"].max()) + 1

        train = inter[inter["split"] == "train"]
        self.train_u = train["user_idx"].to_numpy(np.int64)
        self.train_i = train["item_idx"].to_numpy(np.int64)
        self.train_csr = self._csr(self.train_u, self.train_i)
        for split in ("val", "test"):
            s = inter[inter["split"] == split].sort_values("user_idx")
            assert len(s) == self.n_users
            setattr(self, f"{split}_target", s["item_idx"].to_numpy(np.int64))
        self.item_pop = np.bincount(self.train_i, minlength=self.n_items)   # train-only: no test leakage

        # pairs never used as negatives: train positives + observed low ratings (1-3)
        nonpos = pd.read_parquet(os.path.join(processed_dir, "nonpos.parquet"))
        known = np.concatenate([self.train_u * self.n_items + self.train_i,
                                nonpos["user_idx"].to_numpy(np.int64) * self.n_items + nonpos["item_idx"].to_numpy(np.int64)])
        self._known_codes = np.unique(known)

        # item side: nutrition features (log1p + z-score), compliance, allergens
        self.recipes = pd.read_parquet(os.path.join(processed_dir, "recipes.parquet")).sort_values("item_idx").reset_index(drop=True)
        assert (self.recipes["item_idx"].to_numpy() == np.arange(self.n_items)).all()
        x = np.log1p(self.recipes[NUTRITION_COLS].to_numpy(np.float64))
        self.item_feat = ((x - x.mean(0)) / x.std(0)).astype(np.float32)
        self.item_ok = self.recipes[[f"ok_{c}" for c in self.conditions]].to_numpy(bool)          # (n_items, n_cond)
        self.item_allergen = self.recipes[[f"contains_{a}" for a in self.allergens]].to_numpy(bool)  # (n_items, n_alg)

        # user side: synthetic profiles
        prof = pd.read_csv(os.path.join(processed_dir, "profiles.csv"), dtype={"user_idx": "Int64"})
        warm = prof[prof["source"] == "foodcom_warm"].sort_values("user_idx")
        assert len(warm) == self.n_users
        self.user_cond = self._multi_hot(warm["conditions"], self.conditions)
        self.user_alg = self._multi_hot(warm["allergies"], self.allergens)

        # cold-start users: no training history, targets = their positives on catalog recipes
        cold = pd.read_parquet(os.path.join(processed_dir, "cold_start.parquet"))
        cprof = prof[prof["source"] == "foodcom_cold"].set_index("user_id")
        self.cold_users = np.sort(cold["UserId"].unique())
        cprof = cprof.loc[self.cold_users]
        self.cold_cond = self._multi_hot(cprof["conditions"], self.conditions)
        self.cold_alg = self._multi_hot(cprof["allergies"], self.allergens)
        pos = {u: k for k, u in enumerate(self.cold_users)}
        self.cold_targets = self._csr(cold["UserId"].map(pos).to_numpy(np.int64), cold["item_idx"].to_numpy(np.int64),
                                      n_rows=len(self.cold_users))

        # health-pair pools per condition combination (bitmask over self.conditions)
        self.user_mask = (self.user_cond * (1 << np.arange(len(self.conditions)))).sum(1).astype(np.int64)
        self._compliant, self._violating = {}, {}
        for m in np.unique(self.user_mask[self.user_mask > 0]):
            bits = np.array([(m >> b) & 1 for b in range(len(self.conditions))], bool)
            ok = self.item_ok[:, bits].all(1)
            self._compliant[m] = np.flatnonzero(ok)
            self._violating[m] = np.flatnonzero(~ok)

    # ---------- helpers
    def _csr(self, rows, cols, n_rows=None):
        n_rows = self.n_users if n_rows is None else n_rows
        return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_rows, self.n_items))

    @staticmethod
    def _multi_hot(json_col: pd.Series, names: list[str]) -> np.ndarray:
        lists = json_col.map(json.loads)
        return np.stack([lists.map(lambda l, n=n: n in l).to_numpy() for n in names], 1).astype(np.float32)

    def eval_mask(self, split: str) -> sp.csr_matrix:
        """Items excluded from ranking when evaluating `split` (train positives; + val target when testing)."""
        if split == "val":
            return self.train_csr
        return self.train_csr + self._csr(np.arange(self.n_users), self.val_target)

    # ---------- sampling (fresh every epoch)
    def sample_negatives(self, users: np.ndarray, rng: np.random.Generator, max_tries: int = 10) -> np.ndarray:
        """Uniform negatives excluding the user's train positives and observed low ratings."""
        neg = rng.integers(0, self.n_items, len(users))
        for _ in range(max_tries):
            codes = users * self.n_items + neg
            pos = np.searchsorted(self._known_codes, codes)
            bad = (pos < len(self._known_codes)) & (self._known_codes[np.minimum(pos, len(self._known_codes) - 1)] == codes)
            if not bad.any():
                break
            neg[bad] = rng.integers(0, self.n_items, int(bad.sum()))
        return neg

    def sample_epoch(self, rng: np.random.Generator):
        perm = rng.permutation(len(self.train_u))
        u, i = self.train_u[perm], self.train_i[perm]
        return u, i, self.sample_negatives(u, rng)

    def sample_health_pairs(self, users: np.ndarray, pos_items: np.ndarray, rng: np.random.Generator, mode: str):
        """
        For users with >= 1 condition, return (compliant item, violating item) per row and a validity mask.
          uniform: both items drawn uniformly from the user's compliant / violating pools
          hard:    the violating item is the BPR positive itself (rows where it violates); compliant drawn uniformly
        """
        n = len(users)
        good = np.zeros(n, np.int64)
        bad = np.zeros(n, np.int64)
        valid = np.zeros(n, bool)
        masks = self.user_mask[users]
        for m, comp in self._compliant.items():
            rows = np.flatnonzero(masks == m)
            if len(rows) == 0:
                continue
            good[rows] = comp[rng.integers(0, len(comp), len(rows))]
            if mode == "uniform":
                viol = self._violating[m]
                bad[rows] = viol[rng.integers(0, len(viol), len(rows))]
                valid[rows] = True
            elif mode == "hard":
                bits = np.array([(m >> b) & 1 for b in range(len(self.conditions))], bool)
                violates = ~self.item_ok[pos_items[rows]][:, bits].all(1)
                bad[rows] = pos_items[rows]
                valid[rows] = violates
            else:
                raise ValueError(f"unknown health_neg mode: {mode}")
        return good, bad, valid
