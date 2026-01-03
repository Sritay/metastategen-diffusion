import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices
from regions import plot_regions

def compute_phi_psi(samples: torch.Tensor, pdb_path: Path):
    if isinstance(samples, dict):
        if 'positions' in samples: samples = samples['positions']
        else: return None
            
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs.numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="runs/day10_al_5_hpc")
    args = parser.parse_args()

    root_dir = Path(args.run)
    # Visualize Early/Mid/Late uncertainty
    iters = [1, 5, 9]
    pdb_path = Path("/Users/sritaymistry/projects/metastategen-diffusion/data/raw/alanine-dipeptide-nowater.pdb")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    
    # Row 1: Hexbin of Mean Uncertainty
    # Row 2: Histogram of Uncertainty Values
    
    global_max_unc = 0
    all_rows = []
    
    for idx, i in enumerate(iters):
        iter_dir = root_dir / f"iter_{i:02d}"
        
        cand_path = iter_dir / "candidates.pt"
        unc_path = iter_dir / "uncertainty.pt"
        
        if not cand_path.exists() or not unc_path.exists():
            continue
            
        cand_data = torch.load(cand_path, map_location='cpu')
        uncertainty = torch.load(unc_path, map_location='cpu').numpy()
        phi_psi = compute_phi_psi(cand_data, pdb_path)
        
        # Collect for CSV
        for j in range(len(uncertainty)):
             all_rows.append({
                 "Iter": i,
                 "Phi": phi_psi[j, 0],
                 "Psi": phi_psi[j, 1],
                 "Uncertainty": uncertainty[j]
             })
        
        global_max_unc = max(global_max_unc, uncertainty.max())

        # Plot 1: Uncertainty Map
        ax_map = axes[0, idx]
        hb = ax_map.hexbin(phi_psi[:, 0], phi_psi[:, 1], C=uncertainty, reduce_C_function=np.mean, gridsize=40, cmap='inferno', extent=[-180, 180, -180, 180], mincnt=1)
        ax_map.set_title(f"Iter {i} Uncertainty Map")
        plot_regions(ax_map)
        if idx == 0: ax_map.set_ylabel("Psi")
        
        # Plot 2: Distribution
        ax_dist = axes[1, idx]
        ax_dist.hist(uncertainty, bins=30, color='orange', edgecolor='black', alpha=0.7)
        ax_dist.set_title(f"Iter {i} Unc. Dist (Mean: {uncertainty.mean():.4f})")
        ax_dist.set_xlabel("Uncertainty")
        if idx == 0: ax_dist.set_ylabel("Count")

    plt.suptitle("Uncertainty Landscape & Reduction")
    
    # Save PNG
    out_file_png = Path("demo") / "uncertainty_map.png"
    plt.savefig(out_file_png, dpi=150)
    print(f"Saved {out_file_png}")

    # Save PDF
    out_file_pdf = Path("demo") / "uncertainty_map.pdf"
    plt.savefig(out_file_pdf, dpi=300, format='pdf')
    print(f"Saved {out_file_pdf}")
    
    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        csv_file = Path("demo") / "uncertainty_map.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved {csv_file}")

if __name__ == "__main__":
    main()
