import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Ensure we can import from local directory (for regions.py)
sys.path.append(str(Path(__file__).parent))

try:
    from metastategen.utils.geometry import compute_dihedrals, rad2deg
    from metastategen.utils.pdb import get_ala2_heavy_atom_indices
except ImportError:
    # Try adding project root to path if running as script
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
    from metastategen.utils.geometry import compute_dihedrals, rad2deg
    from metastategen.utils.pdb import get_ala2_heavy_atom_indices

from regions import plot_regions

def compute_phi_psi(samples_path: Path, pdb_path: Path):
    if not samples_path.exists():
        print(f"File not found: {samples_path}")
        return None
    try:
        samples = torch.load(samples_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {samples_path}: {e}")
        return None
        
    if isinstance(samples, dict):
        print(f"Warning: Sample file {samples_path.name} is a dict (likely keys: {samples.keys()}). Expected tensor.")
        return None
    
    samples = samples.cpu()
    
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs.numpy()

def main():
    parser = argparse.ArgumentParser(description="Visualize Evolution Density from AL Run")
    parser.add_argument("--run", type=str, required=True, help="Path to AL run directory (e.g., runs/day11_al_23_hpc)")
    parser.add_argument("--pdb", type=str, default="data/raw/alanine-dipeptide-nowater.pdb", help="Path to reference PDB")
    parser.add_argument("--outdir", type=str, default="analysis_outputs", help="Directory to save plots and CSV")
    parser.add_argument("--iters", type=str, default="0,5,10,15,20", help="Comma-separated list of iterations to plot")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    pdb_path = Path(args.pdb)
    out_dir = Path(args.outdir)
    iters = [int(i) for i in args.iters.split(",")]
    
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist.")
        return

    if not pdb_path.exists():
         # Try relative to project root
         project_root = Path(__file__).resolve().parent.parent.parent
         pdb_path = project_root / args.pdb
         if not pdb_path.exists():
            print(f"Error: PDB file {pdb_path} not found.")
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    
    num_plots = len(iters)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharex=True, sharey=True)
    if num_plots == 1:
        axes = [axes]
    
    # Common settings
    cmap = 'Blues'
    gridsize = 50
    extent = [-180, 180, -180, 180]
    
    all_rows = []

    for i, ax in zip(iters, axes):
        iter_dir = run_dir / f"iter_{i:02d}"
        path = iter_dir / "eval_samples.pt"
        
        print(f"Processing Iter {i}: {path}")
        data = compute_phi_psi(path, pdb_path)
        
        ax.set_aspect('equal')
        
        if data is not None:
             # Linear scale hexbin
             hb = ax.hexbin(data[:, 0], data[:, 1], gridsize=gridsize, cmap=cmap, extent=extent, mincnt=1)
             
             # Overlay Regions
             plot_regions(ax)
             
             # Collect for CSV
             for row in data:
                 all_rows.append({"Iter": i, "Phi": row[0], "Psi": row[1]})
        else:
             ax.text(0, 0, "No Data", ha='center')
             hb = None

        ax.set_title(f"Iter {i}")
        ax.set_xlabel("Phi")
        if i == iters[0]:
            ax.set_ylabel("Psi")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        
        # Colorbar specific to this axis? Or shared?
        # Usually shared is better if scale is same, but counts might vary wildly.
        # Let's add individual colorbars for accuracy.
        if hb is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(hb, cax=cax)
            cb.ax.tick_params(labelsize=8)

    plt.suptitle(f"Evolution of Generated Density ({run_dir.name})")
    
    # Save Plots
    out_png = out_dir / "evolution_density.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")
    
    out_pdf = out_dir / "evolution_density.pdf"
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved {out_pdf}")
    
    # Save Data for Cluster Analysis
    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_out = out_dir / "evolution_data.csv"
        df.to_csv(csv_out, index=False)
        print(f"Saved Data to {csv_out}")

if __name__ == "__main__":
    main()
