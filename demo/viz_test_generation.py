
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Ensure demo/ modules are importable
sys.path.append(str(Path(__file__).parent))
from regions import plot_regions
from metastategen.utils.geometry import compute_dihedrals
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

def main():
    path = "runs/test_local_generation_m002/refined_results.pt"
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    
    if not Path(path).exists():
        print(f"File not found: {path}")
        return

    data = torch.load(path, map_location="cpu")
    # Both initial and refined should be identical since steps=0
    pos = data["initial_positions"]
    
    print(f"Loaded {len(pos)} samples.")
    
    # Get Indices
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    
    # Compute
    angles = compute_dihedrals(pos, indices)
    phi = np.degrees(angles[:, 0].numpy())
    psi = np.degrees(angles[:, 1].numpy())
    
    # Plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Background
    try:
        plot_regions(ax)
    except Exception as e:
        print(f"Warning: regions plot failed: {e}")
        
    # Scatter
    plt.scatter(phi, psi, c='blue', alpha=0.5, s=10, label='Generated (Local Test)')
    
    plt.xlabel('Phi (degrees)')
    plt.ylabel('Psi (degrees)')
    plt.title('Local Generation Test (500 Samples)')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = "runs/test_local_generation/phi_psi_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
