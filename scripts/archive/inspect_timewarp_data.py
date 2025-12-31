import numpy as np
import os

data_dir = "data/timewarp/train"
npz_path = os.path.join(data_dir, "ad1-traj-arrays.npz")
pdb_path = os.path.join(data_dir, "ad1-traj-state0.pdb")

print(f"Inspecting {npz_path}...")
try:
    data = np.load(npz_path)
    print("Keys:", list(data.keys()))
    for k in data.keys():
        print(f"{k}: shape={data[k].shape}, dtype={data[k].dtype}")
        
    if 'energies' in data:
        e = data['energies']
        print(f"Energies shape: {e.shape}")
        print(f"Col 0: mean={np.mean(e[:,0]):.4f}, std={np.std(e[:,0]):.4f}, min={np.min(e[:,0]):.4f}, max={np.max(e[:,0]):.4f}")
        print(f"Col 1: mean={np.mean(e[:,1]):.4f}, std={np.std(e[:,1]):.4f}, min={np.min(e[:,1]):.4f}, max={np.max(e[:,1]):.4f}")
    
    if 'forces' in data:
        f = data['forces']
        f_norm = np.linalg.norm(f, axis=-1)
        print(f"Force magnitudes: mean={np.mean(f_norm):.4f}, max={np.max(f_norm):.4f}")

    if 'positions' in data:
        p = data['positions']
        # Check standard deviation of positions to see spatial extent
        print(f"Position stats: mean={np.mean(p):.2f}, std={np.std(p):.2f}")

except Exception as e:
    print(f"Error loading npz: {e}")

print(f"\nInspecting {pdb_path}...")
try:
    atom_count = 0
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_count += 1
    print(f"Atom count in PDB: {atom_count}")
except Exception as e:
    print(f"Error loading pdb: {e}")
