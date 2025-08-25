def simulate_pipeline(num_buckets: int = 8, quant_us: int = 30, copy_us: int = 50, write_us: int = 60):
    """
    Simple 3-stage pipeline makespan:
      - Fill: quant -> copy -> write
      - Each additional bucket adds max(stage_time)
    Returns total microseconds.
    """
    stage = [quant_us, copy_us, write_us]
    # fill
    t = sum(stage)
    # pipeline (steady state): each extra bucket adds max(stage)
    if num_buckets > 1:
        t += (num_buckets - 1) * max(stage)
    return t
