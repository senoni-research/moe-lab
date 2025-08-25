#!/usr/bin/env python3
import argparse
import csv
import itertools
import time
import torch
from moe.router import TopKRouter
from moe.layer import MoELayer
from moe.balance_losses import aux_load_balance_loss
from moe.quantize import calibrate_per_token_scale, quantize_int8_per_token, dequantize_int8
from moe.comm_models import CommConfig, buffer_size_bytes, latency_model


def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "t", "true", "y", "yes"}


def maybe_quant(x: torch.Tensor, do_int8: bool) -> torch.Tensor:
    if do_int8:
        s = calibrate_per_token_scale(x)
        q = quantize_int8_per_token(x, s)
        return dequantize_int8(q, s)
    return x


def run_once(cfg: dict) -> dict:
    N = cfg["batch"] * cfg["seq"]
    h = torch.randn(N, cfg["d_model"])

    router = TopKRouter(cfg["d_model"], cfg["n_experts"], k=cfg["k"],
                        noisy_std=cfg.get("noisy_std", 0.0),
                        temperature=cfg.get("temperature", 1.0))
    moe = MoELayer(cfg["d_model"], cfg["n_experts"], k=cfg["k"],
                   capacity_factor=cfg["capacity_factor"],
                   drop_policy=cfg["drop_policy"],
                   backup_m=cfg.get("backup_m", 0))

    h = maybe_quant(h, cfg["int8_comms"])
    t0 = time.perf_counter()
    out, gates, topk_idx, topk_vals = moe(h, router)
    aux = aux_load_balance_loss(gates, topk_idx, alpha=cfg["aux_alpha"]).item()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    comm = CommConfig(
        ep=cfg["ep"], local_batch=cfg["batch"], topk=cfg["k"],
        experts_per_rank=cfg["experts_per_rank"],
        hidden_dim=cfg["d_model"],
        dtype_bytes=1 if cfg["int8_comms"] else 2,
        scale_overhead=16,
    )
    buf_mb = buffer_size_bytes(comm) / 1e6
    lat_us = latency_model(comm)

    s = getattr(moe, "last_stats", {})
    # imbalance metrics
    apx = s.get("assigned_per_expert", [])
    total = float(sum(apx)) if apx else 0.0
    if total > 0:
        import math
        p = [a / total for a in apx]
        h = -sum(pi * math.log(pi + 1e-12) for pi in p)
        hn = h / math.log(len(apx)) if len(apx) > 1 else 0.0
        mean = total / len(apx)
        var = sum((a - mean) ** 2 for a in apx) / max(1, len(apx) - 1)
        cv = (var ** 0.5) / mean if mean > 0 else 0.0
    else:
        hn, cv = 0.0, 0.0
    row = dict(cfg)
    row.update({
        "aux_loss": aux,
        "fwd_ms": dt_ms,
        "dispatch_buf_mb": buf_mb,
        "dispatch_lat_us": lat_us,
        "drop_rate": s.get("drop_rate", 0.0),
        "assigned_total": s.get("assigned_total", 0),
        "load_entropy": hn,
        "load_cv": cv,
    })
    return row


def main():
    p = argparse.ArgumentParser(description="Sweep tiny MoE configs and log CSV results")
    p.add_argument("--out", type=str, default="sweep.csv")
    # grids (keep tiny defaults)
    p.add_argument("--d-model", type=int, nargs="+", default=[512])
    p.add_argument("--n-experts", type=int, nargs="+", default=[8])
    p.add_argument("--k", type=int, nargs="+", default=[1, 2])
    p.add_argument("--capacity-factor", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    p.add_argument("--drop-policy", type=str, nargs="+", default=["drop", "backup"])
    p.add_argument("--backup-m", type=int, nargs="+", default=[0, 2])
    p.add_argument("--aux-alpha", type=float, nargs="+", default=[0.0, 0.01])
    p.add_argument("--temperature", type=float, nargs="+", default=[1.0])
    p.add_argument("--noisy-std", type=float, nargs="+", default=[0.0])
    p.add_argument("--int8-comms", type=str, nargs="+", default=["false", "true"])
    p.add_argument("--batch", type=int, nargs="+", default=[32])
    p.add_argument("--seq", type=int, nargs="+", default=[64])
    # comm model params
    p.add_argument("--ep", type=int, nargs="+", default=[8, 32])
    p.add_argument("--experts-per-rank", type=int, nargs="+", default=[1])
    args = p.parse_args()

    keys = [
        "d_model", "n_experts", "k", "capacity_factor", "drop_policy", "backup_m",
        "aux_alpha", "temperature", "noisy_std", "int8_comms",
        "batch", "seq", "ep", "experts_per_rank",
    ]
    grids = [getattr(args, k.replace("-", "_")) for k in keys]
    combos = list(itertools.product(*grids))

    rows = []
    for vals in combos:
        cfg = dict(zip(keys, vals))
        cfg["int8_comms"] = str2bool(cfg["int8_comms"]) if isinstance(cfg["int8_comms"], str) else bool(cfg["int8_comms"]) 
        row = run_once(cfg)
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else keys
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.out} with {len(rows)} rows.")


if __name__ == "__main__":
    main()


