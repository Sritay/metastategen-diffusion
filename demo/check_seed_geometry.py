
import torch
from pathlib import Path

def compute_chiral_volume(x):
    # Indices: N=3, CA=4, CB=5, C=6
    v_n  = x[:, 3] - x[:, 4]
    v_c  = x[:, 6] - x[:, 4]
    v_cb = x[:, 5] - x[:, 4]
    cross = torch.cross(v_c, v_cb, dim=-1)
    vol = torch.sum(v_n * cross, dim=-1)
    return vol

def check_seed_geometry():
    seed_path = Path("data/processed/ala2/split_12/al_seed.pt")
    if not seed_path.exists():
        print(f"Seed data not found at {seed_path}")
        return

    print(f"Loading seed data from {seed_path}...")
    data = torch.load(seed_path)
    x = data["positions"]
    
    # ... Bond checks ...
    d_n_ca = torch.norm(x[:, 3] - x[:, 4], dim=-1)
    d_ca_c = torch.norm(x[:, 4] - x[:, 6], dim=-1)
    
    # Check Chirality
    vols = compute_chiral_volume(x)
    print("\n--- Seed Chirality Stats ---")
    print(f"Mean Volume: {vols.mean():.4f}")
    print(f"Abs Volume Mean: {vols.abs().mean():.4f}")
    print(f"Volume Std: {vols.std():.4f}")
    print(f"Volume Range: [{vols.min():.4f}, {vols.max():.4f}]")
    
    planar_thresh = 0.001
    n_planar = (vols.abs() < planar_thresh).sum().item()
    print(f"Planar Samples (|V| < {planar_thresh}): {n_planar} ({n_planar/len(x)*100:.1f}%)")

    # ... output bond stats ...
    print("\n--- Seed Data Bond Stats ---")
    print(f"N-CA Mean: {d_n_ca.mean():.4f} nm")
    # ... etc ...
    print(f"N-CA Mean: {d_n_ca.mean():.4f} nm")
    print(f"N-CA Std:  {d_n_ca.std():.4f} nm")
    print(f"CA-C Mean: {d_ca_c.mean():.4f} nm")
    print(f"CA-C Std:  {d_ca_c.std():.4f} nm")
    
    # Sanity check ranges
    print(f"N-CA Range: [{d_n_ca.min():.4f}, {d_n_ca.max():.4f}]")
    print(f"CA-C Range: [{d_ca_c.min():.4f}, {d_ca_c.max():.4f}]")
    
    target_n_ca = 0.146
    target_ca_c = 0.151
    
    if abs(d_n_ca.mean() - target_n_ca) < 0.005:
        print("Seed N-CA: PASS")
    else:
        print(f"Seed N-CA: FAIL (Diff {d_n_ca.mean() - target_n_ca:.4f})")
        
    if abs(d_ca_c.mean() - target_ca_c) < 0.005:
        print("Seed CA-C: PASS")
    else:
        print(f"Seed CA-C: FAIL (Diff {d_ca_c.mean() - target_ca_c:.4f})")

if __name__ == "__main__":
    check_seed_geometry()
