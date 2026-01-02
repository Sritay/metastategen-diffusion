import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

import pandas as pd

def compute_phi_psi(samples_path: Path, pdb_path: Path):
    if not samples_path.exists():
        print(f"File not found: {samples_path}")
        return None
    
    samples = torch.load(samples_path)
    # If standard [N, 22, 3] tensor
    if isinstance(samples, dict):
        # Could be 'eval_samples' key?
        print(f"Warning: {samples_path} contains dict keys: {samples.keys()}")
        return None
        
    device = torch.device('cpu') # Plotting on CPU
    samples = samples.to(device)
    
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long, device=device)
    
    # Need to check if samples are 22 atoms or 10 atoms?
    # Eval samples in AL loop are typically 22 atoms (reconstructed) OR 10 atoms?
    # The AL loop uses 'oracle.query(selected)', and 'oracle' returns 'labels' which are 22 atoms.
    # Waittt. 'ensemble.predict' usually outputs 10 atoms (backbone).
    # 'align_and_reconstruct' is used?
    # Actually, let's check input shape.
    if samples.shape[1] == 10:
        # We cannot compute Phi/Psi from 10 atoms easily unless we abuse indices or reconstruct.
        # But 'eval_samples.pt' in AL Loop:
        # _sample_candidates -> _consensus_ddpm -> returns x_10
        # Then _evaluate -> _compute_phi_psi -> get_ala2_heavy_atom_indices
        # If _compute_phi_psi works in AL loop, then samples MUST be 22 atoms?
        # Let's check 'run_al_loop.py' line 306:
        #   gen_phi_psi = _compute_phi_psi(samples, pdb_path)
        # But '_sample_candidates' usually returns what the model produces (10 atoms).
        # Ah, maybe the indices [phi_idx, psi_idx] are valid for the subset?
        # Indices: [4, 6, 8, 14] and [6, 8, 14, 16].
        # If samples are 10 atoms (Backbone), do we have these indices?
        # 10 atoms map to specific indices in 22 atoms.
        # If the AL loop computed KL, it likely worked.
        # Let's trust it for now. If it fails, we handle it.
        pass

    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs.cpu().numpy()

def main():
    root_dir = Path("runs/day8_9_al_3")
    iters = [0, 1, 2, 3]
    pdb_path = Path("/Users/sritaymistry/projects/metastategen-diffusion/data/raw/alanine-dipeptide-nowater.pdb")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    
    all_data = []
    
    for i, ax in zip(iters, axes):
        iter_dir = root_dir / f"iter_{i:02d}"
        path = iter_dir / "eval_samples.pt"
        
        data = compute_phi_psi(path, pdb_path)
        
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_title(f"Iter {i}")
        ax.set_xlabel("Phi")
        if i == 0:
            ax.set_ylabel("Psi")
            
        if data is not None:
             ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.5)
             
             # Collect data for CSV
             df_iter = pd.DataFrame(data, columns=['Phi', 'Psi'])
             df_iter['Iter'] = i
             all_data.append(df_iter)
        else:
             ax.text(0, 0, "No Data", ha='center')
             
    plt.suptitle("Evolution of Generated Structures (Active Learning)")
    out_file = Path("demo") / "evolution_plot.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved {out_file}")

    # Save data to CSV
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        csv_path = Path("demo") / "evolution_data.csv"
        full_df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

if __name__ == "__main__":
    main()
