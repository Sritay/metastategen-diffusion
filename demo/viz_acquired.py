import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

def compute_phi_psi(samples: torch.Tensor, pdb_path: Path):
    if isinstance(samples, dict):
        # acquired.pt is a dict with 'positions'
        if 'positions' in samples:
            samples = samples['positions']
        else:
            return None
            
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs.numpy()

def main():
    root_dir = Path("runs/day8_9_al_3")
    iters = [1, 2, 3] # Iter 0 has no "acquired"
    pdb_path = Path("/Users/sritaymistry/projects/metastategen-diffusion/data/raw/alanine-dipeptide-nowater.pdb")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    for i, ax in zip(iters, axes):
        iter_dir = root_dir / f"iter_{i:02d}"
        
        # Load Acquired Points
        acq_path = iter_dir / "acquired.pt"
        if not acq_path.exists():
            continue
        acq_data = torch.load(acq_path, map_location='cpu')
        acq_phi_psi = compute_phi_psi(acq_data, pdb_path)
        
        # Load Candidates (Uncertainty pool)
        # candidates.pt is the pool from which we selected
        cand_path = iter_dir / "candidates.pt"
        cand_data = torch.load(cand_path, map_location='cpu')
        cand_phi_psi = compute_phi_psi(cand_data, pdb_path)

        ax.set_title(f"Iter {i} Acquisition")
        ax.set_xlabel("Phi")
        if i == 1:
            ax.set_ylabel("Psi")
            
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)

        # Plot ALL candidates as grey background
        if cand_phi_psi is not None:
            ax.scatter(cand_phi_psi[:, 0], cand_phi_psi[:, 1], c='lightgrey', s=5, alpha=0.5, label='Candidates')
            
        # Plot ACQUIRED points as Red
        if acq_phi_psi is not None:
            ax.scatter(acq_phi_psi[:, 0], acq_phi_psi[:, 1], c='red', s=20, alpha=1.0, label='Acquired', edgecolors='black', linewidth=0.5)

        if i == 1:
            ax.legend()
            
    plt.suptitle("Active Learning Acquisition Strategy: Red = Selected for Labeling")
    out_file = Path("demo") / "acquisition_strategy.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
