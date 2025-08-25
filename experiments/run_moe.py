import argparse
import time
import torch
from moe.router import TopKRouter
from moe.layer import MoELayer
from moe.balance_losses import aux_load_balance_loss
from moe.quantize import calibrate_per_token_scale, quantize_int8_per_token, dequantize_int8
from moe.comm_models import CommConfig, buffer_size_bytes, latency_model, tokens_to_send, bytes_per_token_dispatch
from moe.pipeline_sim import simulate_pipeline

def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "t", "true", "y", "yes"}

def main():
    p = argparse.ArgumentParser(description="Minimal MoE experiment (CPU-friendly).")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-experts", type=int, default=8)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--capacity-factor", type=float, default=1.0)
    p.add_argument("--drop-policy", type=str, default="drop", choices=["drop", "backup"])
    p.add_argument("--aux-alpha", type=float, default=0.01)
    p.add_argument("--int8-comms", type=str, default="false")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    N = args.batch * args.seq
    d = args.d-model if hasattr(args, "d-model") else args.d_model  # compatibility
    d = args.d_model

    # Random hidden states simulate token activations after attention.
    h = torch.randn(N, d)

    router = TopKRouter(d, args.n_experts, k=args.k)
    moe = MoELayer(d, args.n_experts, k=args.k,
                   capacity_factor=args.capacity_factor,
                   drop_policy=args.drop_policy)

    # Optional INT8 "comms" path (simulate quantize-dequantize around dispatch)
    def maybe_quant(x: torch.Tensor) -> torch.Tensor:
        if str2bool(args.int8_comms):
            s = calibrate_per_token_scale(x)
            q = quantize_int8_per_token(x, s)
            return dequantize_int8(q, s)
        return x

    # Forward
    t0 = time.time()
    h_q = maybe_quant(h)
    out, gates, topk_idx, topk_vals = moe(h_q, router)
    aux = aux_load_balance_loss(gates, topk_idx, alpha=args.aux_alpha)
    dt = (time.time() - t0) * 1000

    print("=== Forward Pass (toy) ===")
    print(f"N={N} tokens, d={d}, experts={args.n_experts}, topK={args.k}")
    print(f"Output shape: {tuple(out.shape)}, elapsed ~ {dt:.2f} ms")
    print(f"Aux load-balance loss: {aux.item():.6f}")

    # Communication model report (proxy)
    cfg = CommConfig(
        ep=32,
        local_batch=args.batch,
        topk=args.k,
        experts_per_rank=1,
        hidden_dim=d,
        dtype_bytes=1 if str2bool(args.int8_comms) else 2,
        scale_overhead=16 if str2bool(args.int8_comms) else 0,
    )
    buf_MB = buffer_size_bytes(cfg) / 1e6
    lat_us = latency_model(cfg, link_bw_GBps=100.0, startup_us=10.0)
    print("\n=== Communication Model (proxy) ===")
    print(f"Tokens to send (per rank): {tokens_to_send(cfg)}")
    print(f"Bytes per token (dispatch): {bytes_per_token_dispatch(cfg)}")
    print(f"Dispatch buffer ~ {buf_MB:.2f} MB")
    print(f"Modeled latency ~ {lat_us:.1f} µs @ 100 GB/s + 10 µs startup")

    # Pipeline simulator (3-stage: quant -> copy -> write)
    makespan = simulate_pipeline(num_buckets=8, quant_us=30, copy_us=50, write_us=60)
    naive_sum = 30 + 50 + 60
    print("\n=== Pipeline Simulator ===")
    print(f"Naive sum: {naive_sum} µs  |  Pipelined (8 buckets): {makespan} µs")

if __name__ == "__main__":
    main()
