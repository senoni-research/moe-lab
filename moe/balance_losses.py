import torch

def aux_load_balance_loss(gates: torch.Tensor, topk_idx: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    GShard-style auxiliary load balance loss.
      - gates: [N, E] softmax
      - topk_idx: [N, K] long
    Encourages both even 'importance' (sum of gates) and even 'load' (token count) across experts.
    """
    N, E = gates.shape
    K = topk_idx.shape[1]
    importance = gates.sum(0) / N                                        # [E]
    # Frequency an expert is selected among top-K (normalized)
    load_counts = torch.bincount(topk_idx.reshape(-1), minlength=E).float()  # [E]
    load = load_counts / (N * K)                                          # [E]
    # A simple symmetric objective around uniform usage
    loss = (importance * load).sum() * E
    return alpha * loss
