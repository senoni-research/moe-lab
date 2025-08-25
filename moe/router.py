import torch
import torch.nn as nn

class TopKRouter(nn.Module):
    """
    Simple top-K router with optional temperature scaling and additive Gaussian noise.
    Returns:
        topk_idx: [N, K] long
        topk_vals: [N, K] float
        gates: [N, E] float (softmax probabilities)
    """
    def __init__(self, d_model: int, n_experts: int, k: int = 1,
                 noisy_std: float = 0.0, temperature: float = 1.0):
        super().__init__()
        assert k >= 1 and k <= n_experts
        self.w = nn.Linear(d_model, n_experts, bias=False)
        self.k = k
        self.noisy_std = noisy_std
        self.temperature = temperature

    def forward(self, h: torch.Tensor):
        # h: [N, d]
        logits = self.w(h) / self.temperature
        if self.noisy_std > 0:
            logits = logits + torch.randn_like(logits) * self.noisy_std
        gates = torch.softmax(logits, dim=-1)                 # [N, E]
        topk_vals, topk_idx = torch.topk(gates, k=self.k, dim=-1)  # [N, K]
        return topk_idx, topk_vals, gates
