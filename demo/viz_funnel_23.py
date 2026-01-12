
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from regions import plot_regions

def compute_phi_psi_22(samples, device='cpu'):
    # Indices for 22-atom Timewarp
    # Phi: C_prev(4), N(6), CA(8), C(14)
    # Psi: N(6), CA(8), C(14), N_next(16)
    
    samples = samples.to(device)
    phi_idx = [4, 6, 8, 14]
    psi_idx = [6, 8, 14, 16]
    
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long, device=device)
    
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs.cpu().numpy()

def main():
    path = Path("runs/loop_b_refinement_23/refined_results.pt")
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    data = torch.load(path, map_location='cpu')
    
    initial = data['initial_positions']
    refined = data['refined_positions']
    
    print("Computing Dihedrals...")
    # Initial (Generated)
    pp_init = compute_phi_psi_22(initial)
    
    # Refined
    pp_ref = compute_phi_psi_22(refined)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Setup axis
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    ax.set_title("Loop 23 Funnel Plot: Generation to Refinement")
    
    # 0. Background Regions
    plot_regions(ax)
    
    # 1. Generated (Initial) - Gray
    # Subsample if too many
    if len(pp_init) > 5000:
        indices = np.random.choice(len(pp_init), 5000, replace=False)
        pp_init_plot = pp_init[indices]
    else:
        pp_init_plot = pp_init
        
    ax.scatter(pp_init_plot[:, 0], pp_init_plot[:, 1], 
               c='gray', s=10, alpha=0.2, label='Generated (Initial)')
               
    # 2. Refined - Red
    ax.scatter(pp_ref[:, 0], pp_ref[:, 1], 
               c='red', s=40, alpha=0.9, edgecolors='white', linewidth=0.5, label='Refined (Final)')
               
    # Arrows? (Connecting Refined back to their initial state?)
    # We don't have the indices. So we just show the distributions.
    
    ax.legend(loc='upper right')
    
    out_file = Path("demo") / "funnel_plot_23.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved {out_file}")
    
    # PDF
    out_pdf = Path("demo") / "funnel_plot_23.pdf"
    plt.savefig(out_pdf, format='pdf')
    print(f"Saved {out_pdf}")

if __name__ == "__main__":
    main()
