
import torch
from pathlib import Path
import numpy as np

def compute_chiral_volume(x):
    # Indices: N=3, CA=4, CB=5, C=6
    v_n  = x[:, 3] - x[:, 4]
    v_c  = x[:, 6] - x[:, 4]
    v_cb = x[:, 5] - x[:, 4]
    cross = torch.cross(v_c, v_cb, dim=-1)
    vol = torch.sum(v_n * cross, dim=-1)
    return vol

def analyze_geometry(run_dir="runs/day10_al_15_hpc", iter_idx=20):
    path = Path(run_dir) / f"iter_{iter_idx:02d}" / "eval_samples.pt"
    if not path.exists():
        fallback = Path(run_dir) / f"iter_{iter_idx:02d}" / "candidates.pt"
        if fallback.exists():
            print(f"eval_samples.pt not found, using candidates.pt from {fallback}")
            path = fallback
        else:
            print(f"No samples found at {path} or {fallback}")
            return

    print(f"\nAnalyzing Iteration {iter_idx:02d} from {path}...")
    samples = torch.load(path, map_location="cpu")
    print(f"Loaded {samples.shape[0]} samples.")
    
    # 1. Bond Lengths
    d_n_ca = torch.norm(samples[:, 3] - samples[:, 4], dim=-1)
    d_ca_c = torch.norm(samples[:, 4] - samples[:, 6], dim=-1)
    
    print("--- Bond Length Analysis ---")
    print(f"N-CA Target: 0.1460 nm | Mean: {d_n_ca.mean():.4f} nm | Range: [{d_n_ca.min():.4f}, {d_n_ca.max():.4f}]")
    print(f"CA-C Target: 0.1510 nm | Mean: {d_ca_c.mean():.4f} nm | Range: [{d_ca_c.min():.4f}, {d_ca_c.max():.4f}]")
    
    # Check PASS/FAIL for Bonds
    bond_pass = abs(d_n_ca.mean() - 0.146) < 0.001 and abs(d_ca_c.mean() - 0.151) < 0.001
    print(f"Bond Check: {'PASS' if bond_pass else 'FAIL'}")
    
    # 2. Chirality
    vols = compute_chiral_volume(samples)
    
    print("--- Chirality Analysis ---")
    print(f"Mean Abs Volume: {vols.abs().mean():.4f}")
    
    # Define planar threshold
    planar_thresh = 0.001
    n_planar = (vols.abs() < planar_thresh).sum().item()
    frac_planar = n_planar / samples.shape[0]
    
    print(f"Planar Samples (< {planar_thresh}): {n_planar} ({frac_planar*100:.1f}%)")
    
    # Check PASS/FAIL for Chirality (Expect < 30% planar initially, improving over time)
    # Since we want to prove it's NOT collapsing to 100% planar like before.
    chirality_pass = frac_planar < 0.30
    print(f"Chirality Check: {'PASS' if chirality_pass else 'FAIL'}")
    
    if bond_pass and chirality_pass:
        print("OVERALL: SUCCESS")
    else:
        print("OVERALL: FAILURE")

if __name__ == "__main__":
    print("=== Verifying Loop 15 Geometry ===")
    analyze_geometry(iter_idx=0)
    analyze_geometry(iter_idx=20)
