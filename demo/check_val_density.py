import torch
import matplotlib.pyplot as plt
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

def main():
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    val_path = Path("data/processed/ala2/split_4/al_val.pt")
    
    val_data = torch.load(val_path)
    pos = val_data["positions"]
    
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    rads = compute_dihedrals(pos, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    data = degs.numpy()
    
    plt.figure(figsize=(6, 6))
    plt.hexbin(data[:, 0], data[:, 1], gridsize=50, cmap='jet', mincnt=1, bins='log', extent=[-180, 180, -180, 180])
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.title(f"Validation Set Density (Traj 1)\n{len(data)} frames")
    plt.xlabel("Phi")
    plt.ylabel("Psi")
    plt.savefig("demo/val_density_check.png")
    print("Saved demo/val_density_check.png")

if __name__ == "__main__":
    main()
