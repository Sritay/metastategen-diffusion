import torch
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

def check_seed_coverage():
    seed_path = Path("data/processed/ala2/split_5/al_seed.pt")
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    
    if not seed_path.exists():
        print("Seed path not found")
        return

    data = torch.load(seed_path)
    if isinstance(data, dict):
        if 'pos' in data: data = data['pos']
        elif 'data' in data: data = data['data']
        elif 'positions' in data: data = data['positions']

    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    rads = compute_dihedrals(data, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    
    # Alpha_L / C7ax are generally in the Phi > 0 region (Right half of plot)
    # Specifically Alpha_L is approx (50, 50)
    
    mask_right_half = degs[:, 0] > 0
    count_right = mask_right_half.sum().item()
    total = len(degs)
    
    print(f"Total Seed Points: {total}")
    print(f"Points in Right Half (Phi > 0): {count_right} ({count_right/total:.2%})")
    
    if count_right > 0:
        print("\nSample points in Right Half:")
        right_points = degs[mask_right_half][:10]
        print(right_points)

if __name__ == "__main__":
    check_seed_coverage()
