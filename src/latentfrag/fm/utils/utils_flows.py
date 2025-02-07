import torch
import torch.nn.functional as F


def estimate_angular_scale_factor(embeddings, n_samples=1000):
    # Sample pairs of embeddings
    idx = torch.randperm(len(embeddings))[:n_samples]
    pairs = embeddings[idx]

    # Compute angular distances
    cos_sim = F.cosine_similarity(pairs[::2], pairs[1::2], dim=1)
    angles = torch.acos(torch.clamp(cos_sim, -1+1e-6, 1-1e-6))

    # Using median to be robust to outliers
    typical_angle = torch.median(angles)

    # Scale to keep tangent vectors reasonable (e.g. max angle π)
    scale = float(torch.pi / typical_angle)
    return round(scale, 1)