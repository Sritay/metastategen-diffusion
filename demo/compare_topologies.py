
import torch
import numpy as np
from pathlib import Path

def get_bonds(pos, thresh=0.16):
    n = pos.shape[0]
    bonds = []
    dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0))[0]
    for i in range(n):
        for j in range(i+1, n):
            if dist[i,j] < thresh:
                bonds.append((i, j, dist[i,j].item()))
    return bonds

def main():
    print("=== 1. MDShare (Diffusion Data, 10 atoms) ===")
    shard_path = next(Path("data/processed/ala2/shards").glob("*.pt"))
    data = torch.load(shard_path)
    # Load first frame
    pos10 = data['positions'][0]
    types10 = data['atom_types']
    print(f"Atom Types (0=C, 1=N, 2=O, 3=H? No H in 10-atom): {types10.tolist()}")
    
    bonds10 = get_bonds(pos10)
    print("Bonds:")
    for b in bonds10: print(f"  {b[0]}-{b[1]}: {b[2]:.3f}")
    
    print("\n=== 2. Timewarp (Refinement Template, 22 atoms) ===")
    path22 = "data/timewarp/train/positions.pt"
    pos22 = torch.load(path22)[0]
    print(f"Shape: {pos22.shape}")
    
    bonds22 = get_bonds(pos22)
    print("Bonds (Heavy-Heavy only candidates):")
    
    # We want to identify which indices in 22 correspond to the 10 in MDShare
    # Strategy: Find chain of heavy atoms.
    # MDShare is ACE-ALA-NME backbone.
    # Usually: CH3(1)-C(2)=O(3) -- N(4)-CA(5)-C(6)=O(7) -- N(8)-CH3(9) ... 
    # Let's inspect the connectivity to match them.
    
    for b in bonds22:
        print(f"  {b[0]}-{b[1]}: {b[2]:.3f}")
        
if __name__ == "__main__":
    main()
