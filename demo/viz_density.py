import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

def compute_phi_psi(samples_path: Path, pdb_path: Path):
    if not samples_path.exists():
        return None
    samples = torch.load(samples_path, map_location='cpu')
    if isinstance(samples, dict):
        return None
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs.numpy()

def main():
    root_dir = Path("runs/day8_9_al_3")
    iters = [0, 1, 2, 3]
    pdb_path = Path("/Users/sritaymistry/projects/metastategen-diffusion/data/raw/alanine-dipeptide-nowater.pdb")
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    
    # Common settings for hexbin
    cmap = 'jet'
    gridsize = 50
    extent = [-180, 180, -180, 180]

    for i, ax in zip(iters, axes):
        iter_dir = root_dir / f"iter_{i:02d}"
        path = iter_dir / "eval_samples.pt"
        
        data = compute_phi_psi(path, pdb_path)
        
        ax.set_title(f"Iter {i} Density")
        ax.set_xlabel("Phi")
        if i == 0:
            ax.set_ylabel("Psi")
            
        if data is not None:
             # Hexbin plot for density
             hb = ax.hexbin(data[:, 0], data[:, 1], gridsize=gridsize, cmap=cmap, extent=extent, mincnt=1, bins='log')
             # Start marking regions
             # Alpha Basin approx
             rect = plt.Rectangle((-100, -70), 50, 40, linewidth=1, edgecolor='white', facecolor='none', linestyle='--')
             ax.add_patch(rect)
        else:
             ax.text(0, 0, "No Data", ha='center')
             
    plt.suptitle("Evolution of Generated Density (Log Scale)")
    out_file = Path("demo") / "evolution_density.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
