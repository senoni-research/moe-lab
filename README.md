# moe-lab — From Textbook MoE to Production‑Ready (code‑first)

This is a **minimal, ready‑to‑run scaffold** you can use for a hands‑on blog post about Mixture‑of‑Experts (MoE).  
It includes a small MoE implementation, a communication/latency model, optional INT8 activation quantization for
“comms”, and a single experiment script you can run on CPU.

> Goal: let readers **toggle alternatives** (top‑K, capacity policy, load‑balance loss, INT8 comms) and **measure**
> throughput proxies, buffer sizes, and a simple pipeline timeline.

---

## Quick start

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps (CPU-only is fine)
pip install -r requirements.txt

# 3) Run the minimal experiment
python experiments/run_moe.py --d-model 512 --n-experts 8 --k 2 --batch 32 --seq 64   --capacity-factor 1.0 --drop-policy drop --aux-alpha 0.01 --int8-comms true

# Try a few toggles
python experiments/run_moe.py --k 1 --capacity-factor 1.2 --int8-comms false
python experiments/run_moe.py --k 2 --drop-policy backup --aux-alpha 0.0
```

What you should see:
- A forward pass through a toy MoE layer (random input), an auxiliary load‑balance loss value, and
- A **communication model** report (predicted dispatch buffer size and latency) that responds to flags,
- A simple **3‑stage pipeline simulator** showing overlapped makespan vs. naive sum of stages.

### Parameter sweep & plots

Generate a small grid and write `sweep.csv`:

```bash
python experiments/sweep.py --out sweep.csv
```

Plot a simple trade‑off chart (drop rate vs buffer size):

```bash
pip install matplotlib
python experiments/plots.py
```

You can filter the CSV by `k`, `capacity_factor`, or `int8_comms` to reproduce figures in the blog post.

> The code avoids real networking so it runs anywhere, but the knobs reflect real trade‑offs: routing (top‑K),
> capacity management, activation INT8 for comms, and overlapping stages in a pipelined schedule.

---

## Repo layout

```
moe-lab/
  ├─ moe/
  │   ├─ __init__.py
  │   ├─ router.py            # Top‑K router (with optional noise & temperature)
  │   ├─ experts.py           # Simple MLP expert
  │   ├─ layer.py             # Capacity-aware dispatch/compute/combine
  │   ├─ balance_losses.py    # GShard-style auxiliary loss
  │   ├─ quantize.py          # Per-token INT8 activation quant/dequant
  │   ├─ comm_models.py       # Byte/latency model (buffer sizing, link BW)
  │   └─ pipeline_sim.py      # Toy 3-stage pipeline makespan simulator
  ├─ experiments/
  │   └─ run_moe.py           # One experiment: toggles + prints metrics
  ├─ requirements.txt
  └─ README.md
```

---

## Arguments (key)

- `--d-model` (int): hidden size (also used for comm byte modeling)
- `--n-experts` (int): number of experts
- `--k` (int): top‑K routing (1 = Switch, 2 = GShard-like)
- `--capacity-factor` (float): capacity per expert relative to expected tokens
- `--drop-policy` (`drop`|`backup`): what to do when expert is full
- `--aux-alpha` (float): weight of GShard-style load-balance loss
- `--int8-comms` (`true`|`false`): quantize activations to INT8 for “comms” path (simulated)
- `--batch`, `--seq` (int): to build a token count N = batch * seq

---

## Notes

- This is designed to be **transparent and hackable**, not the fastest. Packing tokens per expert uses simple Python
  lists to keep the code readable. Keep `N` small for interactive runs.
- The **communication model** is a proxy (bytes & link bandwidth). It’s here so your post can discuss how different
  choices—`k`, INT8, expert‑parallel size—move the buffer/latency numbers.
- Everything runs on CPU by default; if you have CUDA, PyTorch will still work out of the box.

---

## License
MIT — see `LICENSE`.
