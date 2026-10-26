"""A normalized convex transition from multi-LOD to finest-only supervision."""
def level_loss_weights(num_levels, decay, iteration, mode, ramp_start, ramp_end):
    raw = [decay ** (num_levels - 1 - i) for i in range(num_levels)]
    total = sum(raw)
    base = [w / total for w in raw]
    if mode in ('multi', 'detach', 'multiscale'):
        return base
    assert mode == 'anneal'
    progress = min(1.0, max(0.0, (iteration - ramp_start) / (ramp_end - ramp_start)))
    weights = [(1.0 - progress) * w for w in base]
    weights[-1] += progress
    return weights
