
import torch
import numpy as np
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

def compute_bond_lengths(pos):
    # N-CA (3-4)
    # CA-C (4-6)
    # Return mean and std of deviations
    # Scaling factor 7.6 assumed if data is raw (usually eval samples are raw)
    # check magnitude first.
    # If magnitude is ~1, it is raw. ~7.6 is scaled.
    mean_mag = pos.abs().mean()
    scale = 1.0
    if mean_mag > 2.0:
        scale = 7.6
        pos = pos / scale
        
    n_ca = torch.norm(pos[:, 3] - pos[:, 4], dim=1)
    ca_c = torch.norm(pos[:, 4] - pos[:, 6], dim=1)
    return n_ca, ca_c, scale

def main():
    path = Path("runs/test_local_verify/iter_01/eval_samples.pt")
    if not path.exists():
        print(f"File not found: {path}")
        return

    # Load
    data = torch.load(path)
    # Handle dict
    if isinstance(data, dict):
         data = data["positions"] if "positions" in data else data["pos"]
    
    print(f"Loaded {data.shape} samples.")

    # 1. Bond Lengths
    n_ca, ca_c, scale = compute_bond_lengths(data)
    
    print(f"Detected Scale: {scale}")
    print(f"N-CA: {n_ca.mean():.4f} +/- {n_ca.std():.4f} (Target 0.146)")
    print(f"CA-C: {ca_c.mean():.4f} +/- {ca_c.std():.4f} (Target 0.151)")
    
    # 2. Chirality (Phi)
    # Need indices
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx], dtype=torch.long)
    
    # Needs to be 22 atoms for full indices?
    # Usually eval_samples in AL loop is [10 atoms + Reconstructed]?
    # Or just 10 atoms?
    # get_ala2_heavy_atom_indices returns [4, 6, 8, 14]... wait, those are for 22 atoms.
    # If we have 10 atoms, indices are different.
    # Main atoms:
    # 0: CH3, 1: C=O, 2: N, 3: H, 4: CA (idx 4? no)
    # 10-atom indices: N(1), CA(3), C(4)... need to check mapping.
    
    # But wait, run_al_loop uses _compute_phi_psi so it works.
    # Let's try standard compute_dihedrals if data is 10 atoms.
    # If data is 22, use helper.
    
    if data.shape[1] == 10:
         # Use custom indices for 10-atom backbone
         # N(1), CA(3), C(4) ...
         # Actually let's just look at sign of projection?
         # Or use the function from verify_loop17.py which worked on m000 output.
         # Phi: [1, 3, 4, 6]
         # Psi: [3, 4, 6, 8]
         indices = torch.tensor([[1, 3, 4, 6], [3, 4, 6, 8]], dtype=torch.long)
         pass
    else:
         indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)

    rads = compute_dihedrals(data, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    phi = degs[:, 0].numpy()
    psi = degs[:, 1].numpy()
    
    print(f"Phi Mean: {phi.mean():.2f}")
    print(f"Psi Mean: {psi.mean():.2f}")
    print(f"Phi > 0 Fraction: {(phi > 0).mean():.2f}")
    
    print("\nIndividual Samples (Phi, Psi):")
    for i in range(len(phi)):
        print(f"  {i}: ({phi[i]:.2f}, {psi[i]:.2f})")

if __name__ == "__main__":
    main()
