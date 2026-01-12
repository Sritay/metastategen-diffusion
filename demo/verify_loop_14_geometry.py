
import torch
from pathlib import Path
import numpy as np

def compute_chiral_volume(x):
    """
    Computes chiral volume for Alanine Dipeptide centering on CA.
    Indices: N=3, CA=4, CB=5, C=6.
    V = (r_N - r_CA) . [ (r_C - r_CA) x (r_CB - r_CA) ]
    Note: Order matters for sign, but we just check distribution separation.
    """
    # x: [B, 10, 3]
    # Vectors from CA
    v_n  = x[:, 3] - x[:, 4]
    v_c  = x[:, 6] - x[:, 4]
    v_cb = x[:, 5] - x[:, 4]
    
    # Cross product
    cross = torch.cross(v_c, v_cb, dim=-1)
    
    # Dot product
    vol = torch.sum(v_n * cross, dim=-1)
    return vol

def debug_connectivity(samples):
    print("\n--- Connectivity Debug (First Sample) ---")
    x = samples[0] # [10, 3]
    dists = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0) # [10, 10]
    
    # Print matrix
    print("      " + " ".join([f"{i:4d}" for i in range(10)]))
    for i in range(10):
        row_str = " ".join([f"{dists[i,j]:.2f}" for j in range(10)])
        print(f"{i:4d}: {row_str}")
    
    # Identify close pairs (potential bonds)
    print("\nPotential Bonds (dist < 0.20 nm):")
    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            d = dists[i,j].item()
            if d < 0.20:
                print(f"  {i}-{j}: {d:.4f} nm")
                pairs.append((i, j))
    return pairs

def analyze_geometry(run_dir="runs/day10_al_14_hpc", iter_idx=0):
    path = Path(run_dir) / f"iter_{iter_idx:02d}" / "eval_samples.pt"
    if not path.exists():
        fallback = Path(run_dir) / f"iter_{iter_idx:02d}" / "candidates.pt"
        if fallback.exists():
            print(f"eval_samples.pt not found, using candidates.pt from {fallback}")
            path = fallback
        else:
            print(f"No samples found at {path} or {fallback}")
            return

    print(f"Loading samples from {path}...")
    samples = torch.load(path, map_location="cpu")
    print(f"Loaded {samples.shape[0]} samples.")
    
    # Debug Connectivity
    debug_connectivity(samples)
    
    # 1. Bond Lengths
    # N(3)-CA(4)
    d_n_ca = torch.norm(samples[:, 3] - samples[:, 4], dim=-1)
    # CA(4)-C(6)
    d_ca_c = torch.norm(samples[:, 4] - samples[:, 6], dim=-1)
    
    print("\n--- Bond Length Analysis ---")
    print(f"N-CA Target: 0.146 nm")
    print(f"N-CA Mean:   {d_n_ca.mean():.4f} nm")
    print(f"N-CA Std:    {d_n_ca.std():.4f} nm")
    print(f"N-CA Range:  [{d_n_ca.min():.4f}, {d_n_ca.max():.4f}]")
    
    print(f"\nCA-C Target: 0.151 nm")
    print(f"CA-C Mean:   {d_ca_c.mean():.4f} nm")
    print(f"CA-C Std:    {d_ca_c.std():.4f} nm")
    print(f"CA-C Range:  [{d_ca_c.min():.4f}, {d_ca_c.max():.4f}]")
    
    # Check PASS/FAIL for Bonds
    bond_pass = abs(d_n_ca.mean() - 0.146) < 0.01 and abs(d_ca_c.mean() - 0.151) < 0.01
    print(f"\nBond Check: {'PASS' if bond_pass else 'FAIL'}")
    
    # 2. Chirality
    vols = compute_chiral_volume(samples)
    
    print("\n--- Chirality Analysis (Chiral Volume) ---")
    print(f"Mean Volume: {vols.mean():.4f} (Expected ~0 for racemic)")
    print(f"Abs Volume Mean: {vols.abs().mean():.4f}")
    print(f"Volume Std: {vols.std():.4f}")
    
    # Define planar threshold
    planar_thresh = 0.001
    n_planar = (vols.abs() < planar_thresh).sum().item()
    frac_planar = n_planar / samples.shape[0]
    
    print(f"Planar Samples (|V| < {planar_thresh}): {n_planar} ({frac_planar*100:.1f}%)")
    
    # Check PASS/FAIL for Chirality (Arbitrary < 10% planar is good improvement from 100%)
    chirality_pass = frac_planar < 0.10
    print(f"Chirality Check: {'PASS' if chirality_pass else 'FAIL'}")
    
    print("\n------------------------------")
    if bond_pass and chirality_pass:
        print("OVERALL RESULT: SUCCESS")
    else:
        print("OVERALL RESULT: FAILURE (See above)")

if __name__ == "__main__":
    analyze_geometry(run_dir="runs/debug_local_geom", iter_idx=1)
