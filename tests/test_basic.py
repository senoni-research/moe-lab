import torch
from moe.router import TopKRouter
from moe.layer import MoELayer


def test_shapes_and_stats():
    d_model, n_experts, k = 64, 4, 2
    N = 16
    h = torch.randn(N, d_model)
    router = TopKRouter(d_model, n_experts, k=k)
    moe = MoELayer(d_model, n_experts, k=k, capacity_factor=1.0, drop_policy="drop")
    out, gates, topk_idx, topk_vals = moe(h, router)
    assert out.shape == (N, d_model)
    assert gates.shape == (N, n_experts)
    assert topk_idx.shape == (N, k)
    assert topk_vals.shape == (N, k)
    s = moe.last_stats
    assert "drop_rate" in s and 0.0 <= s["drop_rate"] <= 1.0


