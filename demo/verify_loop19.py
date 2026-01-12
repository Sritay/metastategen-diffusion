
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

def compute_phi_psi(data, pdb_path):
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    rads = compute_dihedrals(data, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs[:, 0], degs[:, 1]

def main():
    print("Verifying Loop 19 Results (Randomized Conditioning Discovery)...")
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    
    # We use the path provided by the user (even if name is '18')
    run_dir = Path("runs/day11_al_18_hpc")
    iter_idx = 20 # Check final iteration (or latest available)
    
    # Check if iter_20 exists, else fallback
    sample_path = run_dir / f"iter_{iter_idx}/eval_samples.pt"
    if not sample_path.exists():
        print(f"Iter {iter_idx} not found. Checking available iters...")
        available = sorted([d.name for d in run_dir.glob("iter_*")])
        if not available:
            print("No iterations found!")
            return
        last_iter = available[-1]
        sample_path = run_dir / last_iter / "eval_samples.pt"
        print(f"Using {last_iter}")
    
    print(f"Loading samples from {sample_path}")
    data = torch.load(sample_path)
    pos = data["positions"] if isinstance(data, dict) else data
    
    # Compute Angles
    phi, psi = compute_phi_psi(pos, pdb_path)
    phi = phi.numpy()
    psi = psi.numpy()
    
    # Analyze Basins
    # Alpha-L: -90 < Psi < 0, Phi < 0
    # C7eq: 0 < Psi < 120, Phi < 0
    # Beta/C5: |Psi| > 120
    # Alpha-R: Phi > 0 (D-like)
    
    alpha_l = ((psi > -90) & (psi < 0) & (phi < 0)).sum()
    c7eq = ((psi > 0) & (psi < 120) & (phi < 0)).sum()
    beta = (np.abs(psi) > 120).sum()
    right_handed = (phi > 0).sum()
    
    total = len(phi)
    
    print("\n--- Basin Analysis (Discovery Check) ---")
    print(f"Total Samples: {total}")
    print(f"Alpha-L Basin: {alpha_l} ({alpha_l/total*100:.2f}%)   [Target: Discovery]")
    print(f"C7eq Basin:    {c7eq} ({c7eq/total*100:.2f}%)      [Target: Discovery]")
    print(f"Beta Basin:    {beta} ({beta/total*100:.2f}%)")
    print(f"Right-Handed:  {right_handed} ({right_handed/total*100:.2f}%)")
    
    if alpha_l > 50: # >5% assuming 1000 samples
        print("\n[SUCCESS] Alpha-L Basin Discovered!")
    else:
        print("\n[WARNING] Alpha-L Basin NOT significantly populated.")
    
    if c7eq > 50:
         print("[SUCCESS] C7eq Basin Discovered!")
    
    # Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(phi, psi, s=5, alpha=0.5, c='blue', label='Generated')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel('Phi')
    plt.ylabel('Psi')
    plt.grid(True)
    plt.title(f"Gen Structures (Loop 19?) - {sample_path.parent.name}")
    plt.legend()
    plt.savefig("demo/verify_loop19.png")
    print("Saved plot to demo/verify_loop19.png")

if __name__ == "__main__":
    main()
