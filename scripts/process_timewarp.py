#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import glob
from metastategen.utils import get_logger, set_deterministic

log = get_logger("process_timewarp")

from typing import Tuple, List

def parse_heavy_atom_indices(pdb_path: Path) -> Tuple[List[int], List[str]]:
    """
    Parse PDB to find indices of heavy atoms (non-H).
    Returns (indices, element_symbols).
    """
    indices = []
    elements = []
    
    with open(pdb_path, 'r') as f:
        # PDB atom index is 1-based in file, we need 0-based for array
        # But we just iterate lines in order.
        idx_counter = 0
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Extract element. Last column or derived from name.
                # Timewarp pdb has element at column 77-78
                elem = line[76:78].strip()
                if not elem:
                     # Fallback to name
                     name = line[12:16].strip()
                     elem = "".join([c for c in name if c.isalpha()])[:1]
                
                elem = elem.upper()
                if elem != 'H':
                    indices.append(idx_counter)
                    elements.append(elem)
                
                idx_counter += 1
                
    return indices, elements

def save_shards(outdir, positions, atom_types, traj_id, shard_size=5000):
    outdir.mkdir(parents=True, exist_ok=True)
    n_frames = positions.shape[0]
    n_shards = (n_frames + shard_size - 1) // shard_size
    
    for s in range(n_shards):
        lo = s * shard_size
        hi = min((s + 1) * shard_size, n_frames)
        
        shard = {
            "positions": positions[lo:hi].clone(),
            "atom_types": atom_types.clone(),
            "traj_id": traj_id[lo:hi].clone(),
            # No phi_psi/frame_id for now unless we calculate them. 
            # Existing training checking only x, a, t.
            # But dataset.py loads 'traj_id' and 'positions'.
             "traj_id": traj_id[lo:hi].clone()
        }
        torch.save(shard, outdir / f"shard_{s:05d}.pt")
        
    log.info(f"Saved {n_shards} shards to {outdir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timewarp-dir", type=str, default="data/timewarp")
    parser.add_argument("--outdir", type=str, default="data/processed/ala2")
    parser.add_argument("--seed", type=int, default=42)
def parse_all_atoms(pdb_path: Path) -> Tuple[List[int], List[str]]:
    """Parse PDB to find all atoms."""
    indices = []
    elements = []
    idx_counter = 0
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                elem = line[76:78].strip()
                if not elem:
                     name = line[12:16].strip()
                     elem = "".join([c for c in name if c.isalpha()])[:1]
                elem = elem.upper()
                indices.append(idx_counter)
                elements.append(elem)
                idx_counter += 1
    return indices, elements

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timewarp-dir", type=str, default="data/timewarp")
    parser.add_argument("--outdir", type=str, default="data/processed/ala2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-heavy-only", action="store_true", default=False, help="Filter for only heavy atoms")
    
    args = parser.parse_args()
    set_deterministic(args.seed)
    
    tw_dir = Path(args.timewarp_dir)
    outdir = Path(args.outdir)
    
    # 1. Parse Topology
    # Find PDB
    pdb_files = list(tw_dir.glob("**/*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB found in {tw_dir}")
    pdb_path = pdb_files[0]
    log.info(f"Using PDB topology: {pdb_path}")
    
    heavy_idx, heavy_elems = parse_heavy_atom_indices(pdb_path)
    log.info(f"Found {len(heavy_idx)} heavy atoms: {heavy_elems}")
    
    # Atom types mapping
    # C=0, N=1, O=2, S=3, H=4, other=5
    mapping = {"C": 0, "N": 1, "O": 2, "S": 3, "H": 4}
    
    # Select Atoms
    if args.force_heavy_only:
        log.info("Filtering for Heavy Atoms (10)...")
        indices = heavy_idx
        elements = heavy_elems
    else:
        log.info("Using All Atoms (22) including Hydrogens...")
        indices = list(range(22)) # Assuming fixed size 22
        # Parse all elements
        all_indices, all_elements = parse_all_atoms(pdb_path)
        elements = all_elements
        
    atom_types_list = [mapping.get(e, 5) for e in elements]
    atom_types = torch.tensor(atom_types_list, dtype=torch.long)
    
    # 2. Load Data
    train_files = sorted(list((tw_dir / "train").glob("*.npz")))
    test_files = sorted(list((tw_dir / "test").glob("*.npz")))
    npz_files = train_files + test_files
    
    if not npz_files:
        raise FileNotFoundError(f"No NPZ found in {tw_dir}")
        
    all_pos, all_force, all_energy, all_traj = [], [], [], []
    
    for i, npz in enumerate(npz_files):
        d = np.load(npz)
        pos = d['positions'] # [T, 22, 3]
        frc = d['forces']     # [T, 22, 3]
        
        # Select atoms
        pos = pos[:, indices, :]
        frc = frc[:, indices, :]
        
        all_pos.append(torch.from_numpy(pos).float())
        all_force.append(torch.from_numpy(frc).float())
        
        ene = d['energies'][:, 0] # [T]
        all_energy.append(torch.from_numpy(ene).float())
        all_traj.append(torch.full((pos.shape[0],), i, dtype=torch.long))

    cat_pos = torch.cat(all_pos, dim=0)
    cat_frc = torch.cat(all_force, dim=0)
    cat_energy = torch.cat(all_energy, dim=0)
    cat_traj = torch.cat(all_traj, dim=0)
    
    # Save Shards
    shard_dir = outdir / "shards"
    if shard_dir.exists():
        import shutil
        shutil.rmtree(shard_dir)
        
    save_shards(shard_dir, cat_pos, atom_types, cat_traj)
    
    # Save Forces
    force_path = outdir / "al_forces_ref.pt"
    torch.save(cat_frc, force_path)
    log.info(f"Saved merged forces to {force_path}")
    
    # Save Energies
    energy_path = outdir / "al_energies_ref.pt"
    torch.save(cat_energy, energy_path)
    log.info(f"Saved merged energies to {energy_path}")
    
    # Save Meta
    meta = {
        "n_atoms": len(indices),
        "atom_types": atom_types,
        "source": "timewarp",
        "heavy_atoms_only": args.force_heavy_only,
        "total_frames": cat_pos.shape[0]
    }
    torch.save(meta, outdir / "meta.pt")
    log.info(f"Saved meta to {outdir / 'meta.pt'}")

if __name__ == "__main__":
    main()
