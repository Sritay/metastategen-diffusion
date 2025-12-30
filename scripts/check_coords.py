
import torch
from pathlib import Path

def main():
    shard_path = "data/processed/ala2/shards/shard_00000.pt"
    if not Path(shard_path).exists():
        print(f"{shard_path} not found.")
        return
        
    d = torch.load(shard_path)
    pos = d['positions'] # [N, 22, 3]
    print(f"Pos shape: {pos.shape}")
    
    # Calculate pairwise distances for first frame
    p = pos[0] # [22, 3]
    dists = torch.cdist(p.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
    
    # Get non-zero distances
    mask = dists > 0
    valid_dists = dists[mask]
    
    print(f"Min Distance: {valid_dists.min().item():.4f}")
    print(f"Mean Distance: {valid_dists.mean().item():.4f}")
    print(f"Max Distance: {valid_dists.max().item():.4f}")
    
    # Check bond-like distances (e.g. < 0.2 if nm, < 2.0 if A)
    n_short = (valid_dists < 0.2).sum().item()
    n_long = (valid_dists > 2.0).sum().item()
    
    print(f"Count < 0.2: {n_short}")
    print(f"Count > 2.0: {n_long}")
    
    if n_short > 0 and n_long == 0:
        print("Likely Nanometers (nm)")
    elif n_short == 0 and n_long > 0:
        print("Likely Angstroms (A)")
    else:
        print("Ambiguous units or mixed?")

if __name__ == "__main__":
    main()
