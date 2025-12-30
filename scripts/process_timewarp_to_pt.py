import numpy as np
import torch
from pathlib import Path

def process(split):
    base_dir = Path(f"data/timewarp/{split}")
    npz_path = base_dir / "ad1-traj-arrays.npz"
    
    if not npz_path.exists():
        print(f"Skipping {split}: {npz_path} not found")
        return

    print(f"Processing {split}...")
    data = np.load(npz_path)
    
    # Positions
    pos = torch.from_numpy(data['positions']).float()
    out_pos = base_dir / "positions.pt"
    torch.save(pos, out_pos)
    print(f"Saved {out_pos}: {pos.shape}")
    
    # Forces
    force = torch.from_numpy(data['forces']).float()
    out_force = base_dir / "forces.pt"
    torch.save(force, out_force)
    print(f"Saved {out_force}: {force.shape}")
    
    # Energies (Select Col 0 = Potential?)
    # Based on earlier inspection: Col 0 mean=-85, Col 1 mean=81.
    # Typically PE is negative. We use Col 0.
    ene = torch.from_numpy(data['energies'][:, 0]).float()
    out_ene = base_dir / "energies.pt"
    torch.save(ene, out_ene)
    print(f"Saved {out_ene}: {ene.shape}")

def main():
    process("train")
    process("test")

if __name__ == "__main__":
    main()
