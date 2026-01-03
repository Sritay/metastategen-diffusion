import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices
from regions import plot_regions

def compute_phi_psi(samples_path: Path, pdb_path: Path):
    if not samples_path.exists():
        return None
    samples = torch.load(samples_path, map_location='cpu')
    if isinstance(samples, dict):
        return None
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
    # Plot Iter 0 (Init), 3 (Early), 6 (Mid), 10 (Final)
    iters = [0, 3, 6, 10]
    pdb_path = Path("/Users/sritaymistry/projects/metastategen-diffusion/data/raw/alanine-dipeptide-nowater.pdb")
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    
    # Common settings for hexbin
    cmap = 'jet'
    gridsize = 50
    extent = [-180, 180, -180, 180]

    # Collect data for CSV
    all_rows = []

    for i, ax in zip(iters, axes):
        iter_dir = root_dir / f"iter_{i:02d}"
        path = iter_dir / "eval_samples.pt"
        
        data = compute_phi_psi(path, pdb_path)
        
        ax.set_title(f"Iter {i} Density")
        ax.set_xlabel("Phi")
        if i == iters[0]:
            ax.set_ylabel("Psi")
            
        if data is not None:
             # Hexbin plot for density with fixed scale (vmin=1, vmax=100)
             hb = ax.hexbin(data[:, 0], data[:, 1], gridsize=gridsize, cmap=cmap, extent=extent, mincnt=1, bins='log', vmin=1, vmax=100)
             
             # Overlay Regions
             plot_regions(ax)
             
             # CSV collection
             for row in data:
                 all_rows.append({"Iter": i, "Phi": row[0], "Psi": row[1]})
        else:
             ax.text(0, 0, "No Data", ha='center')
             
    plt.suptitle("Evolution of Generated Density (Log Scale)")
    
    # Save PNG
    out_file_png = Path("demo") / "evolution_density.png"
    plt.savefig(out_file_png, dpi=150)
    print(f"Saved {out_file_png}")
    
    # Save PDF
    out_file_pdf = Path("demo") / "evolution_density.pdf"
    plt.savefig(out_file_pdf, dpi=300, format='pdf')
    print(f"Saved {out_file_pdf}")

    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        csv_file = Path("demo") / "evolution_density.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved {csv_file}")

    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        csv_file = Path("demo") / "evolution_density.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved {csv_file}")

if __name__ == "__main__":
    main()
