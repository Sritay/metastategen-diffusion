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


def load_training_data(traj_path: Optional[Union[str, Path]] = None, topo_path: Optional[Union[str, Path]] = None, scale_factor: float = 1.0) -> dict[str, torch.Tensor]:
    """
    Generalized loader for training data (positions + atom types).
    
    Args:
        traj_path: Path to trajectory file (.npz, .pdb, .lammpstrj, .gro, .xyz).
                   If None, uses topo_path as trajectory (1 frame).
        topo_path: Path to topology file (.pdb, .psf). 
                   Required for atom types and constraints.
        scale_factor: Scaling factor for positions (default 1.0).
        
    Returns:
        Dict with 'positions' [N, Atoms, 3] and 'atom_types' [Atoms].
    """
    if topo_path is None:
        raise ValueError("topo_path is required to define atom types/topology.")
        
    topo_path = Path(topo_path)
    if not topo_path.exists():
        raise FileNotFoundError(f"Topology file not found: {topo_path}")

    # 1. Load Topology (Atom Types)
    try:
        import mdtraj as md
        # Load topology structure
        traj_top = md.load(str(topo_path))
        atom_types = []
        # Simple mapping (Same as toplogy.py)
        for atom in traj_top.topology.atoms:
            if atom.element.symbol == "H": continue
            s = atom.element.symbol
            if s == "C": atom_types.append(0)
            elif s == "N": atom_types.append(1)
            elif s == "O": atom_types.append(2)
            elif s == "S": atom_types.append(3)
            else: atom_types.append(4)
            
        atom_types = torch.tensor(atom_types, dtype=torch.long)
        heavy_indices = [a.index for a in traj_top.topology.atoms if a.element.symbol != "H"]
        
    except Exception as e:
        raise ValueError(f"Failed to infer atom types from Topology {topo_path}: {e}")

    # 2. Determine Trajectory Source
    if traj_path is None:
        log.info(f"No trajectory path provided. Using topology file {topo_path} as single-frame training data.")
        traj_source = topo_path
    else:
        traj_source = Path(traj_path)
        if not traj_source.exists():
             raise FileNotFoundError(f"Trajectory file not found: {traj_source}")

    # 3. Load Coordinates
    positions = None
    
    if traj_source.suffix == ".npz":
        # Fast path for NPZ
        try:
            data = np.load(traj_source)
            if "positions" in data:
                coords = data["positions"]
            elif "coords" in data:
                coords = data["coords"]
            else:
                raise KeyError(f"NPZ must contain 'positions' or 'coords'. Found: {list(data.keys())}")
            positions = torch.tensor(coords, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Failed to load NPZ {traj_source}: {e}")
            
    else:
        # General path (MDTraj) for PDB, LAMMPS, GRO, XYZ
        try:
            log.info(f"Loading trajectory from {traj_source} using MDTraj...")
            # Use topo_path as topology for loading (essential for LAMMPS/GRO without builtin topology)
            traj = md.load(str(traj_source), top=str(topo_path))
            positions = torch.tensor(traj.xyz, dtype=torch.float32) # [N, Atoms, 3] in nm
        except Exception as e:
             raise ValueError(f"Failed to load trajectory {traj_source}: {e}")

    # 4. Filter Heavy Atoms (if needed)
    n_atoms_data = positions.shape[1]
    n_heavy = len(atom_types)
    
    if n_atoms_data != n_heavy:
        if n_atoms_data == traj_top.n_atoms:
             # Full atoms -> Heavy only
             positions = positions[:, heavy_indices, :]
        else:
             # Mismatch that isn't simple hydrogen filtering
             log.warning(f"Atom count mismatch: Data={n_atoms_data}, Heavy={n_heavy}. Assuming data matches 'atom_types' logic or is custom.")

    _check_units(positions, atom_types)
    
    out = {
        "positions": positions,
        "atom_types": atom_types
    }
    return out

# Alias for backward compatibility
def load_npz_as_al_data(npz_path: Path, pdb_path: Path, scale_factor: float = 1.0) -> dict[str, torch.Tensor]:
    return load_training_data(traj_path=npz_path, topo_path=pdb_path, scale_factor=scale_factor)


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


def load_energy_data(source: Union[str, Path], recursive_coords: bool = False) -> dict[str, torch.Tensor]:
    """
    Generalized loader for energy/force datasets.
    Supported sources:
    - .npz file: Must contain 'positions'/'coords', 'forces', 'energies'.
    - Directory: Assumed to contain .pt shards (forces.pt, energies.pt, positions.pt).
    
    Args:
        source: Path to file or directory.
        recursive_coords: If True, tries to find positions.pt if not explicitly in source list (for legacy shards).

    Returns:
        Dict with keys: 'positions', 'forces', 'energies'.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Data source not found: {source}")

    if source.is_file():
        if source.suffix == ".npz":
            return _load_npz_energy_data(source)
        else:
            raise ValueError(f"Unsupported file format for energy data: {source.suffix}")
    
    elif source.is_dir():
        # Assume legacy PT shards style, or a directory of NPZs (not implemented yet)
        # For now, replicate the logic from train_pairwise for .pt files
        # Expect forces.pt and energies.pt and positions.pt
        return _load_pt_directory(source)
    
    else:
         raise ValueError(f"Invalid data source type: {source}")

def _load_npz_energy_data(path: Path) -> dict[str, torch.Tensor]:
    log.info(f"Loading energy data from NPZ: {path}")
    data = np.load(path)
    
    # Positions
    if "positions" in data:
        pos = data["positions"]
    elif "coords" in data:
        pos = data["coords"]
    else:
        raise KeyError(f"NPZ {path} missing 'positions' or 'coords'")
    
    # Forces
    if "forces" not in data:
         raise KeyError(f"NPZ {path} missing 'forces'")
    forces = data["forces"]
    
    # Energies
    if "energies" not in data:
         raise KeyError(f"NPZ {path} missing 'energies'")
    energies = data["energies"]
    if energies.ndim > 1:
        energies = energies[:, 0] # Assume col 0 is PE
        
    return {
        "positions": torch.from_numpy(pos).float(),
        "forces": torch.from_numpy(forces).float(),
        "energies": torch.from_numpy(energies).float()
    }

def _check_units(positions: torch.Tensor, atom_types: torch.Tensor, threshold_nm: float = 0.5) -> None:
    """
    Heuristic check for unit consistency (Angstrom vs nm).
    MetaStateGen assumes nm (approx 0.15 nm bond length).
    If average bond length is > 0.5, it's likely Angstroms (1.5 A).
    """
    # Quick heuristic: Sample first 100 frames
    N = min(100, positions.shape[0])
    if N == 0: return

    # Calculate pairwise distances for first frame
    pos = positions[0] # [N_atoms, 3]
    if pos.shape[0] < 2: return # Only 1 atom
    
    # Calculate all pairwise distances
    # We only care about bonded ones, but average nearest neighbor is a good proxy.
    dists = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0))[0] # [N, N]
    
    # Mask diagonal
    mask = torch.eye(pos.shape[0], device=pos.device).bool()
    dists = dists.masked_fill(mask, float('inf'))
    
    # Get nearest neighbor for each atom
    min_dists, _ = dists.min(dim=1)
    avg_nn = min_dists.mean().item()
    
    log.info(f"Unit Check: Avg NN distance = {avg_nn:.4f} (Raw input units)")
    
    if avg_nn > threshold_nm:
        log.warning(f"Suspected Angstrom units! Avg NN dist {avg_nn:.2f} > {threshold_nm}. "
                    f"Model expects nm (approx 0.15). Please check 'scale_factor' or convert input.")


def _load_pt_directory(path: Path) -> dict[str, torch.Tensor]:
    # Try to find specific filenames
    # Common convention from this project: forces.pt, energies.pt, positions.pt
    # Or shards like shard_*.pt containing dicts? 
    # train_pairwise previously assumed separate giant .pt files for forces/energies
    
    forces_path = path / "forces.pt"
    energies_path = path / "energies.pt"
    positions_path = path / "positions.pt"
    
    if not forces_path.exists() or not energies_path.exists():
        # Check for shards (e.g. data/processed/ala2/shards/*.pt)
        # Shards usually contain {'positions', 'atom_types', ...} but maybe not forces/energies?
        # The 'train_pairwise.py' original code took EXPLICIT paths for forces and energies.
        # So passing a directory implies standard filenames.
        # If files are named differently, user must point to file directly (which load_energy_data doesn't support for .pt yet? 
        # Actually simplest is to support explicit kwargs in loading if needed, or just standard names.
        raise FileNotFoundError(f"Directory {path} must contain forces.pt and energies.pt (and positions.pt)")

    log.info(f"Loading directory: {path}")
    return {
        "positions": torch.load(positions_path),
        "forces": torch.load(forces_path),
        "energies": torch.load(energies_path)
    }


