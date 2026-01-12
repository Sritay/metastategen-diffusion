import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices
from regions import plot_regions

def compute_phi_psi(samples: torch.Tensor, pdb_path: Path):
    if isinstance(samples, dict):
        if 'pos' in samples: samples = samples['pos']
        elif 'data' in samples: samples = samples['data']
        elif 'positions' in samples: samples = samples['positions']
        
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs.numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="runs/day11_al_23_hpc")
    args = parser.parse_args()
    
    root_dir = Path(args.run)
    if not root_dir.exists():
        print(f"Run directory not found: {root_dir}")
        return

    # Check for pdb file
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    if not pdb_path.exists():
        print("PDB file not found.")
        return

    iters = [1, 10, 20]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    all_rows = []
    
    print(f"{'Iter':<5} | {'Acquired':<10} | {'Training Set':<15}")
    print("-" * 40)
    
    # Validation / Seed Path (Hardcoded for Demo Loop 5)
    seed_path = Path("data/processed/ala2/split_12/al_seed.pt")
    
    for idx, i in enumerate(iters):
        ax = axes[idx]
        iter_dir = root_dir / f"iter_{i:02d}"
        
        # 1. Load Seed (Black)
        # Always exists
        seed_data = compute_phi_psi(torch.load(seed_path), pdb_path)
        
        # 2. Load Previous Acquisitions (Green)
        # Iterate from j=1 to i-1
        prev_acq_list = []
        for j in range(1, i):
            prev_acq_path = root_dir / f"iter_{j:02d}" / "acquired.pt"
            if prev_acq_path.exists():
                data = torch.load(prev_acq_path)
                # Handle dict structure
                if isinstance(data, dict):
                    if 'pos' in data: data = data['pos']
                    elif 'data' in data: data = data['data']
                    elif 'positions' in data: data = data['positions']
                prev_acq_list.append(data)
                
        if prev_acq_list:
            prev_acq_cat = torch.cat(prev_acq_list, dim=0)
            prev_acq_data = compute_phi_psi(prev_acq_cat, pdb_path)
        else:
            prev_acq_data = None
            
        # 3. Load New Acquisitions (Red)
        acq_path = iter_dir / "acquired.pt"
        new_data = None
        if acq_path.exists():
            new_data = compute_phi_psi(torch.load(acq_path), pdb_path)
            
        ax.set_title(f"Iter {i} Acquisition")
        ax.set_xlabel("Phi")
        if idx == 0:
            ax.set_ylabel("Psi")
            
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        
        # Plot Layers (Order matters for visibility)
        
        # Layer 1: Seed (Black) - Base knowlege
        if seed_data is not None:
             # Subsample
             if len(seed_data) > 5000:
                 import numpy as np
                 indices = np.random.choice(len(seed_data), 5000, replace=False)
                 seed_plot = seed_data[indices]
             else:
                 seed_plot = seed_data
            
             ax.scatter(seed_plot[:, 0], seed_plot[:, 1], c='black', s=2, alpha=0.3, label='Seed Data')
             
        # Layer 2: Previous Acquisitions (Green) - Gained knowledge
        if prev_acq_data is not None:
             ax.scatter(prev_acq_data[:, 0], prev_acq_data[:, 1], c='green', s=4, alpha=0.5, label='Prev. Acquired')
             
        # Layer 3: New Batch (Red) - New Discovery
        if new_data is not None:
             ax.scatter(new_data[:, 0], new_data[:, 1], c='red', s=15, alpha=0.9, edgecolors='white', linewidth=0.3, label='New Batch')
             
             # Stats
             n_seed = len(seed_data)
             n_prev = len(prev_acq_data) if prev_acq_data is not None else 0
             n_new = len(new_data)
             print(f"{i:<5} | {n_new:<10} | Seed: {n_seed}, Prev: {n_prev}")
             
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9, markerscale=2.0)
            
        # Overlay Regions (Ground Truth Background)
        plot_regions(ax)
            
    plt.suptitle("Active Learning Strategy: Filling in the Unknown")
    
    # Save PNG
    out_file_png = Path("demo") / "acquisition_strategy.png"
    plt.savefig(out_file_png, dpi=150)
    print(f"\nSaved {out_file_png}")

    # Save PDF
    out_file_pdf = Path("demo") / "acquisition_strategy.pdf"
    plt.savefig(out_file_pdf, dpi=300, format='pdf')
    print(f"Saved {out_file_pdf}")
    
    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        csv_file = Path("demo") / "acquired_points.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved {csv_file}")

if __name__ == "__main__":
    main()
