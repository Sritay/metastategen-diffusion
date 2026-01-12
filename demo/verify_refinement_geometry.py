
import torch
from pathlib import Path
import numpy as np

def compute_chiral_volume(x):
    # Indices: N=3, CA=4, CB=5, C=6
    v_n  = x[:, 3] - x[:, 4]
    v_c  = x[:, 6] - x[:, 4]
    v_cb = x[:, 5] - x[:, 4]
    # Check shape [B, 3]
    if x.dim() == 2: x = x.unsqueeze(0)
    
    cross = torch.cross(v_c, v_cb, dim=-1)
    vol = torch.sum(v_n * cross, dim=-1)
    return vol

def analyze_bonds(samples, label):
    # 22 atom samples
    # N=1, CA=4, C=14 (from heavy indices mapping) or just calculate all?
    # Wait, the 22 atom indices for N, CA, C are:
    # N=1, CA=4, C=14? No.
    # Let's rely on standard AD-22 ordering:
    # 0: CH3 (ACE)
    # 1: C (ACE)
    # 2: O (ACE)
    # 3: N
    # 4: H (N)
    # 5: CA
    # 6: HA
    # 7: CB
    # 8-10: HB
    # 11: C
    # 12: O
    # 13: N (NME)
    
    # Correct Indices for Bonds (Timewarp):
    # N=6, CA=8, C=14
    
    idx_N = 6
    idx_CA = 8
    idx_C = 14
    
    d_n_ca = torch.norm(samples[:, idx_N] - samples[:, idx_CA], dim=-1)
    d_ca_c = torch.norm(samples[:, idx_CA] - samples[:, idx_C], dim=-1)
    
    print(f"--- {label} Bond Lengths ---")
    print(f"N-CA Target: 0.1460 nm | Mean: {d_n_ca.mean():.4f} | Std: {d_n_ca.std():.4f}")
    print(f"CA-C Target: 0.1510 nm | Mean: {d_ca_c.mean():.4f} | Std: {d_ca_c.std():.4f}")
    
    # Check deviations
    n_ca_err = (d_n_ca - 0.146).abs().mean()
    ca_c_err = (d_ca_c - 0.151).abs().mean()
    
    # Force model might relax bonds slightly, but shouldn't break them
    valid = n_ca_err < 0.005 and ca_c_err < 0.005
    print(f"Validity: {'PASS' if valid else 'WARN/FAIL'} (Err: {n_ca_err:.4f}, {ca_c_err:.4f})")
    
def analyze_chirality(samples, label):
    # For AD-22 (Timewarp):
    # N=6, CA=8, CB=10, C=14
    # Vector CA->N
    v_n = samples[:, 6] - samples[:, 8]
    # Vector CA->C
    v_c = samples[:, 14] - samples[:, 8]
    # Vector CA->CB
    v_cb = samples[:, 10] - samples[:, 8]
    
    cross = torch.cross(v_c, v_cb, dim=-1)
    vol = torch.sum(v_n * cross, dim=-1)
    
    abs_vol = vol.abs()
    planar_thresh = 0.001
    frac_planar = (abs_vol < planar_thresh).float().mean().item()
    
    print(f"--- {label} Chirality ---")
    print(f"Mean Abs Volume: {abs_vol.mean():.4f}")
    print(f"Planar Fraction: {frac_planar*100:.1f}%")
    
    return vol

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="runs/loop_b_refinement_16/refined_results.pt")
    args = parser.parse_args()
    
    print(f"Loading {args.path}...")
    
    if not Path(args.path).exists():
        print(f"File not found: {args.path}")
        return

    data = torch.load(args.path, map_location="cpu")
    init_pos = data["initial_positions"] # [N, 22, 3]
    ref_pos = data["refined_positions"]  # [N, 22, 3]
    
    print(f"Loaded {init_pos.shape[0]} samples.")
    
    print("\n=== Initial (Diffusion + Reconstruction) ===")
    analyze_bonds(init_pos, "Initial")
    analyze_chirality(init_pos, "Initial")
    
    print("\n=== Refined (Energy Minimization) ===")
    analyze_bonds(ref_pos, "Refined")
    analyze_chirality(ref_pos, "Refined")
    
    # Check movement
    disp = torch.norm(ref_pos - init_pos, dim=-1).mean()
    print(f"\nMean Atomic Displacement: {disp:.4f} nm")

if __name__ == "__main__":
    main()
