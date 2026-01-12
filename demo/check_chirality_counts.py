
import torch
import numpy as np
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices
from metastategen.models.features import compute_chiral_volume_signal

def main():
    path = Path("runs/day11_al_23_hpc/iter_20/eval_samples.pt")
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Loading: {path}")
    samples = torch.load(path)
    if isinstance(samples, dict):
        if 'pos' in samples: samples = samples['pos']
        elif 'positions' in samples: samples = samples['positions']
        
    device = torch.device('cpu')
    samples = samples.to(device)
    
    # Compute Signal
    signal_tensor = compute_chiral_volume_signal(samples, scale_factor=1.0)
    # signal_tensor might be [B, 1] or [B, N, 1].
    if signal_tensor.dim() == 3:
        signal = signal_tensor[:, 0, 0].numpy()
    else:
        signal = signal_tensor.numpy().flatten()
    
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long, device=device)
    
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    
    phi = degs[:, 0].numpy()
    
    n_total = len(phi)
    
    # Analysis of Phi > 0
    mask_D_phi = phi > 0
    n_D = np.sum(mask_D_phi)
    
    print("-" * 30)
    print(f"Total Samples: {n_total}")
    print(f"Points with Phi > 0: {n_D} ({n_D/n_total*100:.2f}%)")
    
    if n_D > 0:
        phi_pos = phi[mask_D_phi]
        sig_pos = signal[mask_D_phi]
        
        print("\n--- Stats for Phi > 0 Points ---")
        print(f"Phi Range: [{phi_pos.min():.2f}, {phi_pos.max():.2f}]")
        print(f"Signal Range: [{sig_pos.min():.4f}, {sig_pos.max():.4f}]")
        print(f"Signal Mean: {sig_pos.mean():.4f}")
        
        # Check consistency
        n_pos_sig = np.sum(sig_pos > 0)
        print(f"Points with Phi > 0 AND Signal > 0 (True D-Ala?): {n_pos_sig}")
        print(f"Points with Phi > 0 AND Signal < 0 (L-Ala in Positive Phi region?): {np.sum(sig_pos <= 0)}")
    
    print("-" * 30)


if __name__ == "__main__":
    main()
