
import torch
import numpy as np
import os

def check_stats(name, data, is_force=False):
    """
    Data shape expected: [N_frames, N_atoms, 3] or similar.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # Flatten just to get global stats
    flat = data.flatten()
    print(f"--- {name} ---")
    print(f"Shape: {data.shape}")
    print(f"Min: {np.min(flat):.4f}")
    print(f"Max: {np.max(flat):.4f}")
    print(f"Mean: {np.mean(flat):.4f}")
    print(f"Std: {np.std(flat):.4f}")
    
    if not is_force:
        # Calculate heuristics for lengths
        # Assuming data is [B, N, 3]
        if data.ndim == 3:
            # Calculate some interatomic distances for the first frame
            pos = data[0] # [N, 3]
            # Just take distance between atom 0 and 1 (approx bond or non-bonded)
            dist = np.linalg.norm(pos[0] - pos[1])
            print(f"Dist(atom0, atom1) sample: {dist:.4f}")
            
            # Average nearest neighbor dist estimation
            from scipy.spatial.distance import pdist
            dists = pdist(pos)
            print(f"Avg Pairwise Dist in Frame 0: {np.mean(dists):.4f}")
            print(f"Min Pairwise Dist in Frame 0: {np.min(dists):.4f}")
            
            if np.min(dists) > 0.8 and np.min(dists) < 2.0:
                 print("-> CONFIDENCE: Likely ANGSTROMS (bonds ~1-1.5)")
            elif np.min(dists) > 0.08 and np.min(dists) < 0.2:
                 print("-> CONFIDENCE: Likely NANOMETERS (bonds ~0.1-0.15)")
            else:
                 print("-> CONFIDENCE: Unknown scale")

def main():
    print("Checking MDSHARE / PROCESSED Data...")
    try:
        # Load one of the PT files
        path = "data/processed/ala2/al_pool_ref.pt"
        if os.path.exists(path):
            data = torch.load(path)
            # Inspect structure. It might be a dict or a tensor.
            print(f"Loaded {path}, type: {type(data)}")
            if isinstance(data, dict):
               if 'positions' in data:
                   check_stats("MDShare Processed (al_pool_ref['positions'])", data['positions'])
               else:
                   print("Key 'positions' not found in dict keys:", data.keys())
            else:
               check_stats("MDShare Processed (Tensor)", data)
        else:
            print(f"File not found: {path}")
    except Exception as e:
        print(f"Error reading mdshare data: {e}")

    print("\nChecking TIMEWARP Data...")
    try:
        path = "data/timewarp/train/ad1-traj-arrays.npz"
        if os.path.exists(path):
            data = np.load(path)
            print(f"Keys in npz: {list(data.keys())}")
            
            if 'positions' in data:
                check_stats("TimeWarp positions", data['positions'])
            elif 'coords' in data:
                 check_stats("TimeWarp coords", data['coords'])
            
            if 'forces' in data:
                check_stats("TimeWarp forces", data['forces'], is_force=True)
        else:
            print(f"File not found: {path}")

    except Exception as e:
        print(f"Error reading timewarp data: {e}")

if __name__ == "__main__":
    main()
