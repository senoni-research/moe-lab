import torch
import torch.nn as nn
from .experts import ExpertMLP

class MoELayer(nn.Module):
    """
    Capacity-aware, top-K MoE layer (educational, not hyper-optimized).
    - Packs tokens per expert up to a capacity, with simple drop/backup policies.
    - Combines expert outputs weighted by gate scores.
    """
    def __init__(self, d_model: int, n_experts: int, k: int = 1,
                 d_hidden: int = None, capacity_factor: float = 1.0,
                 drop_policy: str = "drop", backup_m: int = 0):
        super().__init__()
        assert k >= 1 and k <= n_experts
        assert drop_policy in ("drop", "backup")
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.drop_policy = drop_policy
        self.backup_m = int(backup_m)
        self.experts = nn.ModuleList([ExpertMLP(d_model, d_hidden or 4 * d_model) for _ in range(n_experts)])
        self.last_stats: dict = {}

    @torch.no_grad()
    def _compute_capacity(self, tokens_per_batch: int) -> int:
        # Expected tokens per expert is tokens_per_batch * k / n_experts
        cap = int(self.capacity_factor * tokens_per_batch * self.k / self.n_experts)
        return max(cap, 1)

    def forward(self, h: torch.Tensor, router) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # h: [N, d]
        topk_idx, topk_vals, gates = router(h)  # [N,K], [N,K], [N,E]
        N, K = topk_idx.shape
        d = h.shape[-1]
        cap = self._compute_capacity(N)

        # Optionally extend candidate set for backup routing
        if self.drop_policy == "backup" and self.backup_m > 0:
            K_plus = min(self.n_experts, K + self.backup_m)
            topk_vals_full, topk_idx_full = torch.topk(gates, k=K_plus, dim=-1)
        else:
            K_plus = K
            topk_vals_full, topk_idx_full = topk_vals, topk_idx

        # Pack tokens for each expert
        expert_inputs = [[] for _ in range(self.n_experts)]
        token_slots = [[] for _ in range(self.n_experts)]  # (token_i, j_in_topk)
        assigned_per_expert = [0 for _ in range(self.n_experts)]
        assigned_total = 0
        dropped_total = 0

        for i in range(N):
            assigned = 0
            for j in range(K_plus):
                e = int(topk_idx_full[i, j].item())
                if len(expert_inputs[e]) < cap:
                    expert_inputs[e].append(h[i])
                    token_slots[e].append((i, j if j < K else (K - 1)))
                    assigned += 1
                    assigned_per_expert[e] += 1
                    assigned_total += 1
                else:
                    if self.drop_policy == "backup":
                        # try next expert in top-K (continue loop)
                        continue
                    # "drop": do nothing
                    dropped_total += 1
            # Note: with "drop", tokens that exceed capacity contribute less output (missing experts).

        # Compute per expert and scatter back (weighted by gates)
        outputs = torch.zeros_like(h)
        for e, xs in enumerate(expert_inputs):
            if not xs:
                continue
            x = torch.stack(xs, dim=0)          # [m, d]
            y = self.experts[e](x)              # [m, d]
            for (i, j), yij in zip(token_slots[e], y):
                outputs[i] += topk_vals[i, j] * yij

        # Stats
        total_routes = N * K
        drop_rate = float(dropped_total) / float(total_routes) if total_routes > 0 else 0.0
        self.last_stats = {
            "capacity": int(cap),
            "assigned_total": int(assigned_total),
            "dropped_total": int(dropped_total),
            "drop_rate": drop_rate,
            "assigned_per_expert": assigned_per_expert,
            "tokens": int(N),
            "topk": int(K),
            "topk_plus": int(K_plus),
        }

        return outputs, gates, topk_idx, topk_vals
