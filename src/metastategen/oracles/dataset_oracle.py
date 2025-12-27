from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from metastategen.utils import get_logger
from .base import Oracle

log = get_logger("dataset_oracle")


class DatasetOracle(Oracle):
    """Oracle that snaps candidates to nearest neighbors in a reference pool."""

    def __init__(
        self,
        pool_path: str | Path,
        device: torch.device | str = "cpu",
        batch_size: int = 100,
    ) -> None:
        self.pool_path = Path(pool_path)
        if not self.pool_path.exists():
            raise FileNotFoundError(self.pool_path)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        data = torch.load(self.pool_path, map_location="cpu")
        if "positions" not in data or "atom_types" not in data:
            raise KeyError("pool data must contain positions and atom_types")

        self.atom_types = data["atom_types"]
        self.traj_id = data.get("traj_id")
        self.frame_id = data.get("frame_id")
        self.source_index = data.get("source_index")
        self.phi_psi = data.get("phi_psi")

        self.device = torch.device(device)
        self.batch_size = int(batch_size)

        pool_positions = data["positions"].to(dtype=torch.float32)
        self.pool_positions = pool_positions.to(self.device)
        self.pool_flat = self.pool_positions.reshape(self.pool_positions.shape[0], -1)

        self.last_indices: Optional[torch.Tensor] = None
        self.last_distances: Optional[torch.Tensor] = None

        log.info(
            "Loaded oracle pool: %s frames on %s",
            self.pool_positions.shape[0],
            self.device,
        )

    @property
    def pool_size(self) -> int:
        return int(self.pool_positions.shape[0])

    @torch.no_grad()
    def query(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.dim() != 3 or positions.shape[-1] != 3:
            raise ValueError(f"Expected positions [B,N,3], got {positions.shape}")
        if positions.shape[1:] != self.pool_positions.shape[1:]:
            raise ValueError(
                f"Candidate shape {positions.shape[1:]} != pool shape {self.pool_positions.shape[1:]}"
            )

        positions = positions.to(self.device, dtype=self.pool_positions.dtype)
        n = positions.shape[0]
        flat = positions.reshape(n, -1)

        indices = []
        distances = []
        for start in range(0, n, self.batch_size):
            end = min(n, start + self.batch_size)
            batch = flat[start:end]
            dists = torch.cdist(batch, self.pool_flat)
            min_d, min_idx = torch.min(dists, dim=1)
            indices.append(min_idx.cpu())
            distances.append(min_d.cpu())

        all_idx = torch.cat(indices, dim=0)
        all_dist = torch.cat(distances, dim=0)
        self.last_indices = all_idx
        self.last_distances = all_dist

        idx_device = all_idx.to(self.device)
        return self.pool_positions[idx_device]

    def get_metadata(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        idx = indices.to("cpu")
        out: dict[str, torch.Tensor] = {}
        if self.traj_id is not None:
            out["traj_id"] = self.traj_id[idx]
        if self.frame_id is not None:
            out["frame_id"] = self.frame_id[idx]
        if self.source_index is not None:
            out["source_index"] = self.source_index[idx]
        if self.phi_psi is not None:
            out["phi_psi"] = self.phi_psi[idx]
        return out
