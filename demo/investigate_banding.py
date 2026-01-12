
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from metastategen.models.features import compute_chiral_volume_signal
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
    print("Investigating Phi Banding at Psi=180...")
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    
    # 1. Load Data
    seed_path = Path("data/processed/ala2/split_12/al_seed.pt")
    gen_path = Path("runs/day11_al_23_hpc/iter_20/eval_samples.pt")
    
    seed = torch.load(seed_path)["positions"]
    # Gen might be dict or tensor
    gen_data = torch.load(gen_path)
    gen = gen_data["positions"] if isinstance(gen_data, dict) else gen_data
    
    # 2. Compute Angles
    phi_seed, psi_seed = compute_phi_psi(seed, pdb_path)
    phi_gen, psi_gen = compute_phi_psi(gen, pdb_path)
    
    # 3. Filter for Band (|Psi| > 150)
    mask_seed = torch.abs(psi_seed) > 150
    mask_gen = torch.abs(psi_gen) > 150
    
    phi_seed_band = phi_seed[mask_seed]
    phi_gen_band = phi_gen[mask_gen]
    
    print(f"Seed Points in band: {len(phi_seed_band)} / {len(seed)}")
    print(f"Gen Points in band:  {len(phi_gen_band)} / {len(gen)}")
    
    # 4. Compute Signals for these points
    # Need to be careful about scaling. Seed is raw (1.0). Gen might be raw or scaled?
    # Eval samples are usually raw.
    # We calculate signal assuming Raw.
    
    sig_seed = compute_chiral_volume_signal(seed, scale_factor=1.0).mean(dim=1)
    sig_gen = compute_chiral_volume_signal(gen, scale_factor=1.0).mean(dim=1)
    
    print(f"Seed Signal Stats: Mean={sig_seed.mean():.4f}, Std={sig_seed.std():.4f}, Min={sig_seed.min():.4f}, Max={sig_seed.max():.4f}")
    print(f"Gen  Signal Stats: Mean={sig_gen.mean():.4f}, Std={sig_gen.std():.4f}, Min={sig_gen.min():.4f}, Max={sig_gen.max():.4f}")

    sig_seed_band = sig_seed[mask_seed]
    sig_gen_band = sig_gen[mask_gen]
    
    # 5. Plot Comparison
    plt.figure(figsize=(15, 6))
    
    # Histogram of Phi in Band
    plt.subplot(1, 3, 1)
    plt.hist(phi_seed_band.numpy(), bins=50, density=True, alpha=0.6, label='Seed', range=(-180, 180))
    plt.hist(phi_gen_band.numpy(), bins=50, density=True, alpha=0.6, label='Gen (Loop 18)', range=(-180, 180))
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Density")
    plt.title("Phi Distribution @ Psi ~ 180")
    plt.legend()
    
    # Scatter: Signal vs Phi (Seed)
    plt.subplot(1, 3, 2)
    plt.scatter(phi_seed_band.numpy(), sig_seed_band.numpy(), s=5, alpha=0.3)
    plt.xlabel("Phi")
    plt.ylabel("Chiral Signal")
    plt.title("Seed: Signal vs Phi")
    plt.axhline(0, color='k', linestyle='--')
    plt.grid(True)
    
    # Scatter: Signal vs Phi (Gen)
    plt.subplot(1, 3, 3)
    plt.scatter(phi_gen_band.numpy(), sig_gen_band.numpy(), s=5, alpha=0.3, color='orange')
    plt.xlabel("Phi")
    plt.ylabel("Chiral Signal")
    plt.title("Gen: Signal vs Phi")
    plt.axhline(0, color='k', linestyle='--')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("demo/investigate_banding.png")
    print("Saved plot to demo/investigate_banding.png")
    
    # 6. Analysis
    # Check if High Signal implies Specific Phi
    # In Seed, high signal (>0.05) likely correlates with Phi ~ -160 (L-Ala Beta)
    # If Gen has High Signal but Phi > 0, that's impossible physically for chiral volume.
    # Unless the chiral volume sign flip is failing?
    
    phi_gen_band_np = phi_gen_band.detach().cpu().numpy().flatten()
    sig_gen_band_np = sig_gen_band.detach().cpu().numpy().flatten()
    
    high_sig_mask = sig_gen_band_np > 0.04
    biased_phi = phi_gen_band_np[high_sig_mask]
    print("\n--- Statistics for Gen Band Points with High Signal (> 0.04) ---")
    print(f"Count: {len(biased_phi)}")
    print(f"Phi Mean: {biased_phi.mean():.2f}")
    print(f"Fraction with Phi > 0: {(biased_phi > 0).astype(float).mean():.2f}")
    
    if (biased_phi > 0).astype(float).mean() > 0.1:
        print("[CRITICAL] Found points with High Positive Signal but Positive Phi (D-Ala region?)")
        print("This suggests Chiral Volume metric might be symmetric/broken for this region?")
    else:
        print("[OK] High signal correctly implies L-Ala Phi (Negative).")
        print("The banding might be low-signal points that the model fails to filter?")

    # --- Added Seed Statistics ---
    print("\n--- Statistics for Seed Band Points ---")
    phi_seed_band_np = phi_seed_band.detach().cpu().numpy().flatten()
    sig_seed_band_np = sig_seed_band.detach().cpu().numpy().flatten()
    
    high_sig_seed = sig_seed_band_np > 0.04
    biased_phi_seed = phi_seed_band_np[high_sig_seed]
    print(f"Count > 0.04: {len(biased_phi_seed)} / {len(phi_seed_band_np)}")
    if len(biased_phi_seed) > 0:
        print(f"Phi Mean: {biased_phi_seed.mean():.2f}")
        print(f"Fraction with Phi > 0: {(biased_phi_seed > 0).astype(float).mean():.2f}")
    
    # Check if there are ANY positive phi points in seed with high signal
    pos_phi_high_sig = (biased_phi_seed > 0).sum()
    print(f"Total Seed points with Phi > 0 and Signal > 0.04: {pos_phi_high_sig}")

if __name__ == "__main__":
    main()
