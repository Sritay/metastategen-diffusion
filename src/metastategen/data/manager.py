from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from metastategen.utils import get_logger

log = get_logger("data_manager")


class PositionsDataset(Dataset):
    """Simple dataset wrapper for positions + atom types."""

    def __init__(self, data: dict[str, torch.Tensor], scale_factor: float = 1.0) -> None:
        if "positions" not in data or "atom_types" not in data:
            raise KeyError("data must include positions and atom_types")
        self.positions = data["positions"]
        self.atom_types = data["atom_types"]
        self.traj_id = data.get("traj_id")
        self.frame_id = data.get("frame_id")
        self.source_index = data.get("source_index")
        self.scale_factor = scale_factor

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out = {"x": self.positions[idx] * self.scale_factor, "a": self.atom_types}
        if self.traj_id is not None:
            out["t"] = self.traj_id[idx]
        if self.frame_id is not None:
            out["frame_id"] = self.frame_id[idx]
        if self.source_index is not None:
            out["source_index"] = self.source_index[idx]
        return out




def load_al_data(path: Union[str, Path]) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu")
    if "positions" not in data or "atom_types" not in data:
        raise KeyError(f"{path} missing required keys: positions, atom_types")
    return data


def load_npz_as_al_data(npz_path: Path, pdb_path: Path, scale_factor: float = 1.0) -> dict[str, torch.Tensor]:
    """
    Loads data from a TimeWarp-style NPZ + PDB pair.
    NPZ expected keys: 'positions' or 'coords' [N, val_atoms, 3].
    PDB used for atom_types.
    """
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)
        
    # Load Topology for Atom Types
    try:
        import mdtraj as md
        traj = md.load(str(pdb_path))
        atom_types = []
        # Simple mapping (Same as toplogy.py) - refactor later to share?
        for atom in traj.topology.atoms:
            if atom.element.symbol == "H": continue
            s = atom.element.symbol
            if s == "C": atom_types.append(0)
            elif s == "N": atom_types.append(1)
            elif s == "O": atom_types.append(2)
            elif s == "S": atom_types.append(3)
            else: atom_types.append(4)
            
        atom_types = torch.tensor(atom_types, dtype=torch.long)
    except Exception as e:
        raise ValueError(f"Failed to infer atom types from PDB {pdb_path}: {e}")

    # Load NPZ
    try:
        data = np.load(npz_path)
        # Look for positions
        if "positions" in data:
            coords = data["positions"]
        elif "coords" in data:
            coords = data["coords"]
        else:
            raise KeyError(f"NPZ must contain 'positions' or 'coords'. Found: {list(data.keys())}")
            
        # TimeWarp data might include hydrogens or be heavy-only. 
        # We need to match the atom_types dimension.
        # If coords shape [N, n_all, 3] and atom_types [n_heavy], we need to slice.
        
        # Check dimensions
        n_atoms_data = coords.shape[1]
        n_atoms_topo = traj.n_atoms
        n_heavy = len(atom_types)
        
        positions = torch.tensor(coords, dtype=torch.float32)
        
        if n_atoms_data == n_atoms_topo and n_atoms_data != n_heavy:
             # NPZ has all atoms, we need to filter for heavy only to match AL pipeline expectations
             # Get heavy indices
             heavy_indices = [a.index for a in traj.topology.atoms if a.element.symbol != "H"]
             positions = positions[:, heavy_indices, :]
        elif n_atoms_data != n_heavy:
             log.warning(f"Mismatch: NPZ atoms {n_atoms_data} vs Inferred Heavy {n_heavy}. Assuming NPZ is already heavy-only or custom?")
             
        out = {
            "positions": positions,
            "atom_types": atom_types
        }
        return out
        
    except Exception as e:
        raise ValueError(f"Failed to load NPZ {npz_path}: {e}")


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

    def __init__(self, seed_data: dict[str, torch.Tensor], scale_factor: float = 1.0) -> None:
        if "positions" not in seed_data or "atom_types" not in seed_data:
            raise KeyError("seed_data must include positions and atom_types")
        self.atom_types = seed_data["atom_types"]
        self._datasets: list[dict[str, torch.Tensor]] = [seed_data]
        self.scale_factor = scale_factor

    def append(self, new_data: dict[str, torch.Tensor]) -> None:
        if not torch.equal(self.atom_types, new_data["atom_types"]):
            raise ValueError("atom_types mismatch in appended data")
        self._datasets.append(new_data)

    def cumulative_data(self) -> dict[str, torch.Tensor]:
        return merge_al_data(self._datasets)

    def dataset(self) -> PositionsDataset:
        return PositionsDataset(self.cumulative_data(), scale_factor=self.scale_factor)

    def size(self) -> int:
        return int(sum(d["positions"].shape[0] for d in self._datasets))
