from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Oracle(ABC):
    """Abstract oracle interface for relaxing candidate structures."""

    @abstractmethod
    def query(self, positions: torch.Tensor) -> torch.Tensor:
        """Return relaxed structures for each candidate positions [B,N,3]."""
        raise NotImplementedError
