"""
BPR-MF (Rendle et al., UAI 2009): matrix factorisation trained with the Bayesian Personalized
Ranking pairwise loss  -log σ(x_ui − x_uj)  on (user, positive, sampled negative) triples.

    x_ui = p_u · q_i + b_i      (item bias b_i optional, as in MyMediaLite's BPRMF)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, item_bias: bool = False):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.item_bias = nn.Embedding(n_items, 1) if item_bias else None
        if item_bias:
            nn.init.zeros_(self.item_bias.weight)

    # vectors (overridden by HealthBPR to add side information)
    def user_vec(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_emb(users)

    def item_vec(self, items: torch.Tensor | None = None) -> torch.Tensor:
        return self.item_emb.weight if items is None else self.item_emb(items)

    def bias(self, items: torch.Tensor | None = None) -> torch.Tensor:
        """Item bias b_i (zeros when disabled); all items if `items` is None."""
        if self.item_bias is None:
            n = self.item_emb.num_embeddings if items is None else len(items)
            return torch.zeros(n, device=self.item_emb.weight.device)
        return self.item_bias.weight[:, 0] if items is None else self.item_bias(items)[:, 0]

    def pair_scores(self, uv: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """x_ui for row-aligned user vectors and items."""
        return (uv * self.item_vec(items)).sum(-1) + self.bias(items)

    def full_scores(self, users: torch.Tensor) -> torch.Tensor:
        """Scores of `users` against every item: (len(users), n_items)."""
        return self.user_vec(users) @ self.item_vec().T + self.bias()

    def bpr_loss(self, u, i, j, reg: float) -> tuple[torch.Tensor, dict]:
        uv, iv, jv = self.user_vec(u), self.item_vec(i), self.item_vec(j)
        x_uij = (uv * iv).sum(-1) - (uv * jv).sum(-1) + self.bias(i) - self.bias(j)
        loss = -F.logsigmoid(x_uij).mean()
        l2 = uv.pow(2).sum() + iv.pow(2).sum() + jv.pow(2).sum()
        if self.item_bias is not None:
            l2 = l2 + self.bias(i).pow(2).sum() + self.bias(j).pow(2).sum()
        return loss + reg * l2 / len(u), {"bpr": loss.item()}

    def loss(self, batch: dict, reg: float) -> tuple[torch.Tensor, dict]:
        return self.bpr_loss(batch["u"], batch["i"], batch["j"], reg)
