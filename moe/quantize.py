import torch

def calibrate_per_token_scale(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-token symmetric scale: max-abs -> 127 levels
    x: [N, d]
    Returns scale: [N, 1]
    """
    s = x.abs().amax(dim=-1, keepdim=True) / 127.0
    return s.clamp_min(eps)

def quantize_int8_per_token(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q = torch.round((x / scale).clamp(-127, 127)).to(torch.int8)
    return q

def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (q.float() * scale).contiguous()
