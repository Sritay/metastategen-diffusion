
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    f_path = "data/processed/ala2/al_forces_ref.pt"
    e_path = "data/processed/ala2/al_energies_ref.pt"
    
    print(f"Loading {f_path}...")
    forces = torch.load(f_path) # [T, N, 3]
    print(f"Loading {e_path}...")
    energies = torch.load(e_path) # [T]
    
    print(f"Forces shape: {forces.shape}")
    print(f"Energies shape: {energies.shape}")
    
    # 1. Basic Stats
    f_flat = forces.flatten()
    print(f"Force: Mean={f_flat.mean():.4f}, Std={f_flat.std():.4f}, Min={f_flat.min():.4f}, Max={f_flat.max():.4f}")
    
    e_flat = energies.flatten()
    print(f"Energy: Mean={e_flat.mean():.4f}, Std={e_flat.std():.4f}, Min={e_flat.min():.4f}, Max={e_flat.max():.4f}")
    
    # 2. Check for outliers
    # Forces > 5 * std?
    limit = 5 * f_flat.std()
    outliers = (f_flat.abs() > limit).sum()
    print(f"Forces > 5*sigma ({limit:.2f}): {outliers} / {len(f_flat)} ({outliers/len(f_flat)*100:.2f}%)")
    
    # 3. Check Consistency?
    # Hard without positions.
    
    # 4. Save Histograms
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(f_flat.numpy(), bins=100, log=True)
    plt.title("Forces (Log Scale)")
    
    plt.subplot(1, 2, 2)
    plt.hist(e_flat.numpy(), bins=100)
    plt.title("Energies")
    
    plt.savefig("debug_data_hist.png")
    print("Saved debug_data_hist.png")

if __name__ == "__main__":
    main()
