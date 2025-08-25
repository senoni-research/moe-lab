from dataclasses import dataclass

@dataclass
class CommConfig:
    ep: int                  # expert-parallel domain size (ranks)
    local_batch: int
    topk: int
    experts_per_rank: int
    hidden_dim: int
    dtype_bytes: int         # 2 for fp16/bf16, 1 for int8
    scale_overhead: int = 0  # per-token metadata bytes (e.g., scales)

def tokens_to_send(cfg: CommConfig) -> int:
    return cfg.local_batch * min(cfg.topk, cfg.experts_per_rank)

def bytes_per_token_dispatch(cfg: CommConfig) -> int:
    return cfg.hidden_dim * cfg.dtype_bytes + cfg.scale_overhead

def buffer_size_bytes(cfg: CommConfig) -> int:
    return cfg.ep * tokens_to_send(cfg) * bytes_per_token_dispatch(cfg)

def latency_model(cfg: CommConfig, link_bw_GBps: float = 100.0, startup_us: float = 10.0) -> float:
    """
    Very simple latency model: startup + (bytes / bandwidth).
    Bandwidth is in GB/s, returns microseconds.
    """
    bytes_total = buffer_size_bytes(cfg)
    GB = 1024**3
    xfer_us = (bytes_total / GB) / link_bw_GBps * 1e6
    return startup_us + xfer_us
