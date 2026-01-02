import torch
import numpy as np
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

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
    path = Path("runs/day8_9_al_3/iter_03/eval_samples.pt")
    pdb = Path("/Users/sritaymistry/projects/metastategen-diffusion/data/raw/alanine-dipeptide-nowater.pdb")
    
    data = compute_phi_psi(path, pdb)
    if data is None:
        print("Error: Could not load data.")
        return

    # Approximate Alpha Basin Limits
    # Phi: -100 to -50
    # Psi: -70 to -30
    mask = (data[:, 0] > -100) & (data[:, 0] < -50) & (data[:, 1] > -70) & (data[:, 1] < -30)
    count = np.sum(mask)
    total = len(data)
    
    print(f"Total Samples: {total}")
    print(f"Samples in Alpha Basin Region: {count}")
    print(f"Percentage: {count/total*100:.2f}%")

if __name__ == "__main__":
    main()
