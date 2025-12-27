from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset

from metastategen.utils import get_logger

log = get_logger("data_manager")


class PositionsDataset(Dataset):
    """Simple dataset wrapper for positions + atom types."""

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        if "positions" not in data or "atom_types" not in data:
            raise KeyError("data must include positions and atom_types")
        self.positions = data["positions"]
        self.atom_types = data["atom_types"]
        self.traj_id = data.get("traj_id")
        self.frame_id = data.get("frame_id")
        self.source_index = data.get("source_index")

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out = {"x": self.positions[idx], "a": self.atom_types}
        if self.traj_id is not None:
            out["t"] = self.traj_id[idx]
        if self.frame_id is not None:
            out["frame_id"] = self.frame_id[idx]
        if self.source_index is not None:
            out["source_index"] = self.source_index[idx]
        return out


def load_al_data(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu")
    if "positions" not in data or "atom_types" not in data:
        raise KeyError(f"{path} missing required keys: positions, atom_types")
    return data


def _concat_optional(keys: Iterable[str], datasets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [d[key] for d in datasets if key in d]
        if tensors and len(tensors) == len(datasets):
            out[key] = torch.cat(tensors, dim=0)
    return out


def merge_al_data(datasets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not datasets:
        raise ValueError("datasets must be non-empty")
    atom_types = datasets[0]["atom_types"]
    for d in datasets[1:]:
        if not torch.equal(atom_types, d["atom_types"]):
            raise ValueError("atom_types mismatch across datasets")

    merged = {
        "positions": torch.cat([d["positions"] for d in datasets], dim=0),
        "atom_types": atom_types,
    }
    optional = _concat_optional(["traj_id", "frame_id", "source_index", "phi_psi"], datasets)
    merged.update(optional)
    return merged


class ALDataManager:
    """Tracks cumulative AL data (seed + acquired) for training."""

    def __init__(self, seed_data: dict[str, torch.Tensor]) -> None:
        if "positions" not in seed_data or "atom_types" not in seed_data:
            raise KeyError("seed_data must include positions and atom_types")
        self.atom_types = seed_data["atom_types"]
        self._datasets: list[dict[str, torch.Tensor]] = [seed_data]

    def append(self, new_data: dict[str, torch.Tensor]) -> None:
        if not torch.equal(self.atom_types, new_data["atom_types"]):
            raise ValueError("atom_types mismatch in appended data")
        self._datasets.append(new_data)

    def cumulative_data(self) -> dict[str, torch.Tensor]:
        return merge_al_data(self._datasets)

    def dataset(self) -> PositionsDataset:
        return PositionsDataset(self.cumulative_data())

    def size(self) -> int:
        return int(sum(d["positions"].shape[0] for d in self._datasets))
