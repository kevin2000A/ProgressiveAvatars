"""Threshold-first triangle selection; limits never force additional splits."""
import torch

def select_split_nodes(leaves, levels, scores, threshold, count, growth_fraction, ceiling, max_depth, max_new_points):
    # leaves: (L,), levels/scores: (N,); each retained parent adds three children.
    eligible = leaves[(levels[leaves] < max_depth) & (scores[leaves] > threshold)]
    limit = min(int(count * growth_fraction) // 3, (ceiling - count) // 3, max_new_points // 3)
    ranked = torch.argsort(scores[eligible], descending=True, stable=True)
    return eligible[ranked[:limit]]
