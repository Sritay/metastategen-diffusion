
import torch
import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md
import sys
from pathlib import Path

# Ensure demo/ modules are importable
sys.path.append(str(Path(__file__).parent))
from regions import plot_regions
from metastategen.utils.geometry import compute_dihedrals
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

def main():
    path = "runs/loop_b_refinement_16/refined_results.pt"
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    
    if not Path(path).exists():
        print(f"File not found: {path}")
        return
        
    if not pdb_path.exists():
        print(f"PDB not found: {pdb_path}")
        return

    data = torch.load(path, map_location="cpu")
    init_pos = data["initial_positions"]
    ref_pos = data["refined_positions"]
    
    print(f"Initial (Generated): {init_pos.shape}")
    print(f"Refined (Final): {ref_pos.shape}")
    
    # Get Indices
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    
    # Compute and convert (result is radians, shape [B, 2])
    angles_init = compute_dihedrals(init_pos, indices)
    angles_ref = compute_dihedrals(ref_pos, indices)
    
    # Split
    phi_init, psi_init = angles_init[:, 0], angles_init[:, 1]
    phi_ref, psi_ref = angles_ref[:, 0], angles_ref[:, 1]
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # 0. Background (Ground Truth Regions)
    try:
        plot_regions(ax)
        print("Overlaid Ground Truth Regions.")
    except Exception as e:
        print(f"Warning: Could not plot regions: {e}")
    
    # 1. Generated (All) - Gray Scatter
    plt.scatter(phi_init, psi_init, c='gray', alpha=0.3, s=5, label='Generated (Raw)')
    
    # 2. Refined (Final) - Red Scatter
    plt.scatter(phi_ref, psi_ref, c='red', alpha=0.9, s=25, label='Refined (Minimization)')
    
    plt.xlabel('Phi (radians)')
    plt.ylabel('Psi (radians)')
    plt.title('Refinement Funnel: Generated vs Refined')
    plt.xlim(-180, 180) # Regions.py uses degrees [-180, 180]
    plt.ylim(-180, 180)
    
    # Adjust axes units to degrees because regions.py plots in degrees!
    # Our compute_dihedrals returns RADIANS. We must convert.
    # Wait, plot_regions logic:
    # "H, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=100, range=[[-180, 180], [-180, 180]])"
    # So plot_regions expects degrees.
    
    # Let's fix our data to degrees.
    phi_init_deg = np.degrees(phi_init)
    psi_init_deg = np.degrees(psi_init)
    phi_ref_deg = np.degrees(phi_ref)
    psi_ref_deg = np.degrees(psi_ref)
    
    # Clear previous scatter calls (since they were radians) and redo
    plt.cla()
    
    # Redo Plotting in Degrees
    plot_regions(ax)
    plt.scatter(phi_init_deg, psi_init_deg, c='gray', alpha=0.3, s=5, label='Generated (Raw)')
    plt.scatter(phi_ref_deg, psi_ref_deg, c='red', alpha=0.9, s=25, label='Refined (Minimization)')
    
    plt.xlabel('Phi (degrees)')
    plt.ylabel('Psi (degrees)')
    plt.title('Refinement Funnel: Generated vs Refined')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = "runs/loop_b_refinement_16/funnel_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
