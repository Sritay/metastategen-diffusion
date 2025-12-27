from __future__ import annotations

from typing import Optional

import torch


def select_random(n_samples: int, k: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, n_samples)
    return torch.randperm(n_samples, generator=generator)[:k]


def select_uncertainty(scores: torch.Tensor, k: int) -> torch.Tensor:
    scores = scores.reshape(-1)
    if scores.numel() == 0:
        raise ValueError("scores is empty")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, scores.numel())
    return torch.topk(scores, k=k, largest=True).indices


def select_acquisition(
    scores: Optional[torch.Tensor],
    k: int,
    strategy: str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    strategy = strategy.lower()
    if strategy == "random":
        if scores is None:
            raise ValueError("scores must be provided for random selection to infer n_samples")
        return select_random(int(scores.numel()), k, generator=generator)
    if strategy == "uncertainty":
        if scores is None:
            raise ValueError("scores required for uncertainty selection")
        return select_uncertainty(scores, k)
    raise ValueError(f"Unknown acquisition strategy: {strategy}")
