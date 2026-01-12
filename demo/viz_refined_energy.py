
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.models.pairwise import PairwiseEnergyModel
from regions import plot_regions

def compute_phi_psi_22(samples, device='cpu'):
    # Indices for 22-atom Timewarp
    # Phi: 4, 6, 8, 14
    # Psi: 6, 8, 14, 16
    
    samples = samples.to(device)
    phi_idx = [4, 6, 8, 14]
    psi_idx = [6, 8, 14, 16]
    
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long, device=device)
    
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs.cpu().numpy()

from matplotlib.colors import LogNorm

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="runs/loop_b_refinement_23/refined_results.pt")
    parser.add_argument("--force-ckpt", type=str, default="runs/energy_pairwise/best_model.pt")
    args = parser.parse_args()

    res_path = Path(args.results)
    if not res_path.exists():
        print(f"File not found: {res_path}")
        return

    print(f"Loading results from {res_path}...")
    data = torch.load(res_path, map_location='cpu')
    refined = data['refined_positions'] # [N, 22, 3]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading Energy Model from {args.force_ckpt}...")
    model = PairwiseEnergyModel(n_atoms=22).to(device)
    ckpt = torch.load(args.force_ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    
    # Scaling stats
    e_mean = ckpt['e_mean'].to(device)
    e_std = ckpt['e_std'].to(device)
    
    # Compute Energies
    print("Computing Energies...")
    refined = refined.to(device)
    batch_size = 100
    energies = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(refined), batch_size):
            batch = refined[i:i+batch_size]
            e_norm = model(batch)
            e_val = e_norm * e_std + e_mean
            energies.append(e_val.cpu())
            
    energies = torch.cat(energies, dim=0).squeeze().numpy()
    
    # Compute Dihedrals
    print("Computing Dihedrals...")
    pp = compute_phi_psi_22(refined, device=device)
    phi = pp[:, 0]
    psi = pp[:, 1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Background
    plot_regions(ax)
    
    # Log Scale Setup
    # Shift to positive range relative to min
    e_min = energies.min()
    e_plot = energies - e_min + 1.0 # Base 1.0 for log
    
    # Filter high energy outliers for plot clarity (Top 5%)
    limit_idx = int(len(energies) * 0.95)
    limit_val = np.sort(e_plot)[limit_idx]
    
    mask = e_plot < limit_val
    
    sc = ax.scatter(phi[mask], psi[mask], c=e_plot[mask], cmap='viridis_r', 
                    norm=LogNorm(vmin=e_plot[mask].min(), vmax=e_plot[mask].max()),
                    s=40, alpha=0.9, edgecolors='black', linewidth=0.5, zorder=10)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"Relative Energy (Log Scale) [Min E = {e_min:.1f}]")
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    ax.set_title(f"Refined Structures Colored by Log Relative Energy\n(Loop 23, {mask.sum()} structures)")
    
    out_file = Path("demo") / "refined_energy_plot_log.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved {out_file}")
    
    out_pdf = Path("demo") / "refined_energy_plot_log.pdf"
    plt.savefig(out_pdf, format='pdf')
    print(f"Saved {out_pdf}")

if __name__ == "__main__":
    main()
