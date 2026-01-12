
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# Ensure demo/ modules are importable
sys.path.append(str(Path(__file__).parent))
from regions import plot_regions
from metastategen.utils.geometry import compute_dihedrals
from metastategen.utils.pdb import get_ala2_heavy_atom_indices
from analyze_clusters import get_cluster_centers

def analyze_subset(pos_tensor, label_name, pdb_path):
    print(f"\n--- Analyzing {label_name} ({len(pos_tensor)} samples) ---")
    
    # Compute Angles
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    angles = compute_dihedrals(pos_tensor, indices)
    
    # Convert to Degrees
    phi = np.degrees(angles[:, 0].numpy())
    psi = np.degrees(angles[:, 1].numpy())
    
    # Cluster
    # Use same eps as standard analysis (0.28 rad approx / or whatever get_cluster_centers uses).
    # get_cluster_centers uses degrees internally.
    top_clusters, labels = get_cluster_centers(phi, psi, eps=0.28, min_samples=20 if len(pos_tensor) > 1000 else 5)
    
    print(f"{'Rank':<5} | {'Phi':>7} | {'Psi':>7} | {'Count':>5} | {'Description'}")
    print("-" * 55)
    
    for rank, c in enumerate(top_clusters):
        cx, cy = c['phi'], c['psi']
        
        # Determine Region Description
        desc = "Unknown"
        # High Energy Barrier (Phi ~ 0)
        if -25 <= cx <= 25:
            desc = "Barrier (Phi~0)"
        # Alpha Right
        elif -100 <= cx <= -40 and -80 <= cy <= -10: 
            desc = "Alpha_R"
        # Beta / C5 / Extended
        elif (cx <= -100 or cx >= 150) and (cy >= 90 or cy <= -90):
            desc = "Beta/C5"
        # C7eq
        elif -120 <= cx <= -40 and 0 <= cy <= 100:
            desc = "C7eq"
        # C7ax / Alpha Left
        elif 30 <= cx <= 90 and -90 <= cy <= -30:
            desc = "C7ax"
        elif 30 <= cx <= 90 and 0 <= cy <= 80:
            desc = "Alpha_L"
            
        print(f"#{rank+1:<4} | {cx:>7.1f} | {cy:>7.1f} | {int(c['count']):>5} | {desc}")
        
    return phi, psi, top_clusters, labels

def main():
    path = "runs/loop_b_refinement_16/refined_results.pt"
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    
    if not Path(path).exists():
        print(f"File not found: {path}")
        return

    data = torch.load(path, map_location="cpu")
    init_pos = data["initial_positions"]
    ref_pos = data["refined_positions"]
    
    # Plot Setup
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    # Analyze Generated
    phi_gen, psi_gen, clust_gen, labels_gen = analyze_subset(init_pos, "Generated (All)", pdb_path)
    
    # Plot Generated
    ax = axes[0]
    plot_regions(ax) # Background
    # Plot Noise
    mask_noise = labels_gen == -1
    ax.scatter(phi_gen[mask_noise], psi_gen[mask_noise], c='lightgrey', s=5, alpha=0.1)
    # Plot Clusters
    unique_valid = sorted(list(set(labels_gen) - {-1}))
    cmap = plt.cm.get_cmap('tab10', len(unique_valid) if unique_valid else 1)
    for lab in unique_valid:
        mask = labels_gen == lab
        ax.scatter(phi_gen[mask], psi_gen[mask], s=5, alpha=0.3)
    ax.set_title(f"Generated ({len(init_pos)})")
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    
    # Analyze Refined
    phi_ref, psi_ref, clust_ref, labels_ref = analyze_subset(ref_pos, "Refined (Top 1%)", pdb_path)
    
    # Plot Refined
    ax = axes[1]
    plot_regions(ax) # Background
    # Plot Noise
    mask_noise = labels_ref == -1
    ax.scatter(phi_ref[mask_noise], psi_ref[mask_noise], c='lightgrey', s=20, alpha=0.3)
    # Plot Clusters
    unique_valid = sorted(list(set(labels_ref) - {-1}))
    cmap = plt.cm.get_cmap('tab10', len(unique_valid) if unique_valid else 1)
    for lab in unique_valid:
        mask = labels_ref == lab
        ax.scatter(phi_ref[mask], psi_ref[mask], s=20, alpha=0.6)
    ax.set_title(f"Refined ({len(ref_pos)})")
    ax.set_xlabel("Phi")
    
    plt.suptitle("Refinement Cluster Analysis")
    out_path = "runs/loop_b_refinement_16/cluster_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

if __name__ == "__main__":
    main()
