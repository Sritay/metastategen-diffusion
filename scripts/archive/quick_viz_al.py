import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg

def main():
    # Hardcoded indices (from previous logs) for when PDB is missing
    # Frame 0: [HC1, C, C, H, N, CA, HA, C, O, N, H, C, HC1, HC2, HC3]... typical AD 
    # Actually, let's use the indices seen in logs:
    # Phi: [1, 3, 4, 6]
    # Psi: [3, 4, 6, 8]
    phi_idx = [1, 3, 4, 6]
    psi_idx = [3, 4, 6, 8]
    
    sample_path = "runs/day8_9_al_3/iter_03/eval_samples.pt"
    if not Path(sample_path).exists():
        print(f"File not found: {sample_path}")
        return

    print(f"Loading {sample_path}...")
    samples = torch.load(sample_path, map_location="cpu")
    print(f"Shape: {samples.shape}")

    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    
    print("Computing dihedrals...")
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    phi_psi = degs.numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(phi_psi[:, 0], phi_psi[:, 1], s=1, alpha=0.5, label="Generated")
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel("Phi")
    plt.ylabel("Psi")
    plt.title("Ramachandran Plot (Loop 3 - Success)")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='k', alpha=0.1)
    plt.axvline(0, color='k', alpha=0.1)
    
    out_path = "runs/day8_9_al_3/iter_03/phi_psi_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    csv_path = "runs/day8_9_al_3/iter_03/phi_psi_values.csv"
    np.savetxt(csv_path, phi_psi, delimiter=",", header="phi,psi", comments="")
    print(f"Saved raw values to {csv_path}")

if __name__ == "__main__":
    main()
