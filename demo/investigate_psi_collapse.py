
import torch
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from pathlib import Path
from metastategen.models.features import compute_chiral_volume_signal
from metastategen.utils.geometry import compute_dihedrals, rad2deg

def main():
    print("Investigating Psi Collapse...")
    
    # 1. Load Reference Data (MD Trajectory or Seed)
    # We use the raw PDB but better to use the training seed if possible.
    # Let's use the actual AL seed file.
    seed_path = Path("data/processed/ala2/split_12/al_seed.pt")
    if not seed_path.exists():
        print("Seed not found, using huge MDshare data (might be slow)...")
        # Fallback to PDB for structure then generate random valid batch? 
        # Or load raw?
        import mdshare
        pdb_path = "data/raw/alanine-dipeptide-nowater.pdb"
        # Dummy: Just use PDB and Perturb?
        # Better: Load the validation set for a representative distribution.
        val_path = Path("data/processed/ala2/split_12/val_set.pt")
        data = torch.load(val_path)["positions"]
    else:
        print(f"Loading Seed: {seed_path}")
        data = torch.load(seed_path)["positions"]
        
    print(f"Loaded {len(data)} frames.")
    
    # 2. Compute Chiral Signal (Raw scale assumed)
    # data is usually raw in .pt files
    # compute_chiral_volume_signal expects raw unless scaled.
    # Pass scale_factor=1.0 just to be sure.
    
    signals = compute_chiral_volume_signal(data, scale_factor=1.0).mean(dim=1) # [B]
    
    # 3. Compute Psi
    # Need indices.
    from metastategen.utils.pdb import get_ala2_heavy_atom_indices
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    rads = compute_dihedrals(data, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    
    phi = degs[:, 0].numpy()
    psi = degs[:, 1].numpy()
    sig = signals.cpu().numpy()
    
    # 4. Analysis
    # A) Correlation Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(psi, sig, s=5, alpha=0.1)
    plt.xlabel("Psi (degrees)")
    plt.ylabel("Chiral Signal Strength")
    plt.title("Chiral Signal vs Psi")
    plt.grid(True, alpha=0.3)
    
    # Plot the Target Value used in loop (approx 0.10)
    target_val = sig.mean() # Assuming we conditioned on mean of seed
    plt.axhline(target_val, color='r', linestyle='--', label=f"Target (~{target_val:.3f})")
    plt.legend()
    
    # B) Seed Distribution
    plt.subplot(1, 2, 2)
    plt.hist(psi, bins=60, range=(-180, 180), density=True, alpha=0.7)
    plt.xlabel("Psi (degrees)")
    plt.ylabel("Density")
    plt.title("Seed Data Psi Distribution")
    plt.xlim(-180, 180)
    
    plt.tight_layout()
    plt.savefig("demo/investigate_psi_collapse.png")
    print("Saved plot to demo/investigate_psi_collapse.png")
    
    # Stats
    print("\n--- Statistics ---")
    print(f"Target Signal Mean: {target_val:.4f}")
    
    # Check signal variation across regions
    # Define Regions
    # Beta/C5: |Psi| > 120
    # C7eq: 0 < Psi < 120 (rough)
    # Alpha: -120 < Psi < 0 (rough)
    
    mask_beta = np.abs(psi) > 120
    mask_c7eq = (psi > 0) & (psi < 120)
    mask_alpha = (psi > -90) & (psi < 0) # Rough definition of Alpha region
    
    sig_beta = sig[mask_beta]
    sig_c7eq = sig[mask_c7eq]
    sig_alpha = sig[mask_alpha]
    
    print(f"Beta (|Psi|>120) Count: {len(sig_beta)}")
    print(f"Beta Signal Mean: {sig_beta.mean():.4f} +/- {sig_beta.std():.4f}")
    
    print(f"C7eq (0<Psi<120) Count: {len(sig_c7eq)}")
    print(f"C7eq Signal Mean: {sig_c7eq.mean():.4f} +/- {sig_c7eq.std():.4f}")
    
    print(f"Alpha (-90<Psi<0) Count: {len(sig_alpha)}")
    if len(sig_alpha) > 0:
        print(f"Alpha Signal Mean: {sig_alpha.mean():.4f} +/- {sig_alpha.std():.4f}")
    else:
        print("Alpha Signal Mean: N/A (No points in seed)")
    
    diff = abs(sig_beta.mean() - sig_c7eq.mean())
    print(f"Signal Difference: {diff:.4f}")
    
    if diff > 0.05: # Arbitrary threshold
        print("\n[CONCLUSION] Chiral Signal varies significantly with Psi!")
        print("Conditions on 'Mean Signal' might favor one basin over another.")
    else:
        print("\n[CONCLUSION] Chiral Signal is relatively independent of Psi.")

if __name__ == "__main__":
    main()
