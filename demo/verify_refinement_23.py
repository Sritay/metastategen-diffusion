
import torch
import numpy as np
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices
from metastategen.models.features import compute_chiral_volume_signal

def check_bonds(pos):
    # Check N-CA (6-8) and CA-C (8-14) for 22-atom Timewarp
    # Indices: 6, 8, 14
    # Wait, assuming 22-atom heavy indices?
    # No, indices in sample_refined.py were 6, 8, 14 for 22-atom tensor.
    # N=6, CA=8, C=14 (0-indexed).
    
    n = pos[:, 6]
    ca = pos[:, 8]
    c = pos[:, 14]
    
    d_n_ca = torch.norm(n - ca, dim=1)
    d_ca_c = torch.norm(ca - c, dim=1)
    
    print(f"Bond N-CA: Mean={d_n_ca.mean():.4f}, Std={d_n_ca.std():.4f}")
    print(f"Bond CA-C: Mean={d_ca_c.mean():.4f}, Std={d_ca_c.std():.4f}")
    
    return d_n_ca.std() < 1e-3 and d_ca_c.std() < 1e-3

def main():
    path = Path("runs/loop_b_refinement_23/refined_results.pt")
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    data = torch.load(path, map_location='cpu')
    
    initial = data['initial_positions']
    refined = data['refined_positions']
    
    print(f"Initial Samples: {initial.shape}")
    print(f"Refined Samples: {refined.shape}")
    
    # Check Bonds
    print("\n--- Checking Initial Bonds ---")
    check_bonds(initial)
    print("\n--- Checking Refined Bonds ---")
    valid_bonds = check_bonds(refined)
    
    # Check Chirality (Refined)
    # Refined is 22 atoms.
    # Need to map to 10-atom indices for features.py?
    # features.py uses N=3, CA=4, CB=5, C=6 (for 10-atom standard).
    # For 22-atom:
    # N is usually index 6. CA is 8. CB is 10. C is 14.
    # We must extract these to compute signal correctly.
    
    # Extract backbone+CB subset: [N, CA, CB, C] -> [6, 8, 10, 14]
    # We construct a synthetic 10-atom-like tensor or just call manual volume.
    # Manual Volume: V = (N-CA) . ((CB-CA) x (C-CA))
    
    idx_N = 6
    idx_CA = 8
    idx_CB = 10
    idx_C = 14
    
    def calc_vol(x):
        r_N = x[:, idx_N]
        r_CA = x[:, idx_CA]
        r_CB = x[:, idx_CB]
        r_C = x[:, idx_C]
        v1 = r_N - r_CA
        v2 = r_CB - r_CA
        v3 = r_C - r_CA
        cp = torch.cross(v2, v3, dim=-1)
        vol = torch.sum(v1 * cp, dim=-1)
        return vol * 1000.0
        
    print("\n--- Checking Refined Chirality ---")
    vols = calc_vol(refined)
    print(f"Volume Mean: {vols.mean():.4f}")
    print(f"Volume Max: {vols.max():.4f}")
    print(f"Volume Min: {vols.min():.4f}")
    
    n_D = (vols > 0).sum().item()
    print(f"D-Alanine (Vol > 0): {n_D} / {len(vols)}")
    
    if valid_bonds and n_D == 0:
        print("\nSUCCESS: Refined structures are valid.")
    else:
        print("\nWARNING: Issues detected.")

if __name__ == "__main__":
    main()
