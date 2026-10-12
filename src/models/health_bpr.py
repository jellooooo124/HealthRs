"""
Health-aware BPR-MF = BPR-MF + two adaptations for chronic-disease diet recommendation.

Adaptation A (input representation, `use_features`):
    item vector  q_i = e_i + W · f_i          f_i: standardised log1p nutrition per serving (9-d)
    user vector  p_u = e_u + Σ_{c ∈ C(u)} g_c  g_c: learned embedding of condition c
  Plain MF knows neither a recipe's nutrition nor a user's conditions. With A, a cold-start
  user (no ID embedding) still gets p_u = Σ g_c, and unseen/rare recipes are placed by nutrition.

Adaptation B (objective, `lambda_h`):
    L = L_BPR + λ · L_health,   L_health = −mean log σ(x_u,i+ − x_u,i−)
  over users with ≥ 1 condition, where i+ complies with all of the user's conditions and i−
  violates at least one (guideline thresholds, configs/health_rules.yaml). λ trades taste vs health.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bpr_mf import BPRMF


class HealthBPR(BPRMF):
    def __init__(self, n_users: int, n_items: int, dim: int, item_feat: torch.Tensor, user_cond: torch.Tensor,
                 use_features: bool = True, lambda_h: float = 0.0, item_bias: bool = False):
        super().__init__(n_users, n_items, dim, item_bias)
        self.use_features = use_features
        self.lambda_h = lambda_h
        self.register_buffer("item_feat", item_feat.float())   # (n_items, n_feat)
        self.register_buffer("user_cond", user_cond.float())   # (n_users, n_cond)
        if use_features:
            self.feat_proj = nn.Linear(item_feat.shape[1], dim, bias=False)
            self.cond_emb = nn.Parameter(torch.zeros(user_cond.shape[1], dim))
            nn.init.normal_(self.feat_proj.weight, std=0.01)
            nn.init.normal_(self.cond_emb, std=0.01)

    def user_vec(self, users: torch.Tensor) -> torch.Tensor:
        v = self.user_emb(users)
        if self.use_features:
            v = v + self.user_cond[users] @ self.cond_emb
        return v

    def item_vec(self, items: torch.Tensor | None = None) -> torch.Tensor:
        if items is None:
            v, f = self.item_emb.weight, self.item_feat
        else:
            v, f = self.item_emb(items), self.item_feat[items]
        if self.use_features:
            v = v + self.feat_proj(f)
        return v

    def cold_user_vec(self, cond: torch.Tensor) -> torch.Tensor:
        """Vector for users without an ID embedding: Σ condition embeddings (zeros without A)."""
        if not self.use_features:
            return torch.zeros(len(cond), self.item_emb.embedding_dim, device=cond.device)
        return cond.float() @ self.cond_emb

    def loss(self, batch: dict, reg: float) -> tuple[torch.Tensor, dict]:
        total, logs = self.bpr_loss(batch["u"], batch["i"], batch["j"], reg)
        if self.lambda_h > 0 and batch.get("h_valid") is not None and batch["h_valid"].any():
            m = batch["h_valid"]
            uv = self.user_vec(batch["u"][m])
            x = self.pair_scores(uv, batch["h_good"][m]) - self.pair_scores(uv, batch["h_bad"][m])
            l_h = -F.logsigmoid(x).mean()
            total = total + self.lambda_h * l_h
            logs["health"] = l_h.item()
        return total, logs
