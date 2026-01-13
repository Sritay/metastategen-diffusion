import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Ensure we can import regions
sys.path.append(str(Path(__file__).parent))
from regions import plot_regions

try:
    from metastategen.utils.geometry import compute_dihedrals, rad2deg
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
    from metastategen.utils.geometry import compute_dihedrals, rad2deg

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
    parser = argparse.ArgumentParser(description="Visualize Refinement Funnel Plot")
    parser.add_argument("--run", type=str, required=True, help="Path to refinement run directory (containing refined_results.pt)")
    parser.add_argument("--outdir", type=str, default="analysis_outputs", help="Directory to save plots")
    
    args = parser.parse_args()
    run_dir = Path(args.run)
    out_dir = Path(args.outdir)
    
    results_path = run_dir / "refined_results.pt"
    if not results_path.exists():
        print(f"Error: File {results_path} not found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {results_path}...")
    try:
        data = torch.load(results_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {results_path}: {e}")
        return
    
    initial = data['initial_positions']
    refined = data['refined_positions']
    
    print("Computing Dihedrals...")
    pp_init = compute_phi_psi_22(initial)
    pp_ref = compute_phi_psi_22(refined)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    ax.set_title(f"Funnel Plot: Generation to Refinement ({run_dir.name})")
    
    # Background Regions
    plot_regions(ax)
    
    # Generated (Initial) - Gray
    if len(pp_init) > 5000:
        indices = np.random.choice(len(pp_init), 5000, replace=False)
        pp_init_plot = pp_init[indices]
    else:
        pp_init_plot = pp_init
        
    ax.scatter(pp_init_plot[:, 0], pp_init_plot[:, 1], 
               c='gray', s=10, alpha=0.2, label='Generated (Initial)')
               
    # Refined - Red
    ax.scatter(pp_ref[:, 0], pp_ref[:, 1], 
               c='red', s=40, alpha=0.9, edgecolors='white', linewidth=0.5, label='Refined (Final)')
    
    ax.legend(loc='upper right')
    
    out_png = out_dir / "funnel_plot.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")
    
    out_pdf = out_dir / "funnel_plot.pdf"
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved {out_pdf}")

if __name__ == "__main__":
    main()
