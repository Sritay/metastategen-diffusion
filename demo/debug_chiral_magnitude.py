
import torch
import numpy as np

def compute_active_chiral_features(x):
    # Copy of the function from src/metastategen/models/features.py
    # WITHOUT the scaling/clamping for inspection
    B, N, _ = x.shape
    c = x.mean(dim=1, keepdim=True)
    r = x - c
    x_i = x.unsqueeze(2)
    x_j = x.unsqueeze(1)
    dist_sq = torch.sum((x_i - x_j)**2, dim=-1)
    weights = torch.exp(-dist_sq) 
    mask = torch.eye(N, device=x.device).unsqueeze(0)
    weights = weights * (1.0 - mask)
    S1 = torch.einsum("bij,bjd->bid", weights, r)
    S2 = torch.einsum("bij,bjd->bid", weights**2, r)
    cross = torch.cross(S1, S2, dim=-1)
    V = torch.sum(cross * r, dim=-1, keepdim=True)
    return V

def main():
    seed_path = "data/processed/ala2/split_12/al_seed.pt"
    print(f"Loading seed form {seed_path}")
    data = torch.load(seed_path)
    x = data["positions"]  # [N, 22, 3] usually
    if x.shape[1] == 10:
        print("Using subset (10 atoms)")
    else:
        # subset to 10 atoms if model uses 10
        # Wait, the model Egnn is built for 10 atoms?
        # The 'ensemble.py' sets 'diffusion' for 'n_particles'
        # The data loader for diffusion usually yields the backbone or everything?
        # pdb_utils says 10 heavy atoms.
        # Let's assume input is 10.
        # But Seed might be 22.
        # Let's slice if needed.
        pass

    # IMPORTANT: Apply Scale Factor
    scale_factor = 7.6
    x_scaled = x * scale_factor
    
    print(f"Stats for Scaled Input (Scale={scale_factor}):")
    
    # Compute Raw V
    V_raw = compute_active_chiral_features(x_scaled)
    
    print(f"Raw V Mean: {V_raw.mean():.6f}")
    print(f"Raw V Std:  {V_raw.std():.6f}")
    print(f"Raw V Max:  {V_raw.max():.6f}")
    print(f"Raw V Min:  {V_raw.min():.6f}")
    
    # Check current scaling effect (1000.0)
    V_current = V_raw * 1000.0
    print(f"\nCurrent Impl (x1000):")
    print(f"Scaled Mean: {V_current.mean():.6f}")
    print(f"Scaled Max:  {V_current.max():.6f}")
    
    # Check saturation
    n_sat = (V_current.abs() > 10.0).sum().item()
    print(f"Saturated (>10.0): {n_sat} / {V_current.numel()} ({n_sat/V_current.numel()*100:.2f}%)")

if __name__ == "__main__":
    main()
