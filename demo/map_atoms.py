
import torch
import numpy as np

def main():
    path = "data/timewarp/train/positions.pt"
    pos = torch.load(path)[0] # [22, 3]
    
    n_atoms = 22
    dist = torch.cdist(pos, pos)
    
    print("--- Finding Bonds (< 0.16 nm) ---")
    bonds = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            d = dist[i, j].item()
            if d < 0.16:
                bonds.append((i, j, d))
                print(f"{i}-{j}: {d:.4f}")
                
    print("\n--- Deducing Types ---")
    # C=O ~ 0.123
    # N-H, C-H ~ 0.10-0.11
    # C-N (peptide) ~ 0.133
    # N-CA ~ 0.146
    # CA-C ~ 0.151
    # CA-CB ~ 0.153
    # C-C (methyl) ~ 0.150
    
    # Identify Hydrogens (only 1 neighbor usually)
    neighbors = {i: [] for i in range(n_atoms)}
    for i, j, d in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
        
    hydrogens = [i for i, n in neighbors.items() if len(n) == 1]
    heavies = [i for i, n in neighbors.items() if len(n) > 1] # Technically terminal methyl carbons have 4 neighbors (3H + 1C)
    
    print(f"Leaf Nodes (Hydrogens/Terminal?): {hydrogens}")
    # Methyl C has 4 neighbors. H has 1.
    real_hydrogens = []
    for h in hydrogens: 
        # Double check distance
        n = neighbors[h][0]
        d = dist[h, n].item()
        if d < 0.115:
            real_hydrogens.append(h)
            
    print(f"True Hydrogens (dist < 0.115): {real_hydrogens}")
    real_heavies = sorted(list(set(range(n_atoms)) - set(real_hydrogens)))
    print(f"Heavy Atoms ({len(real_heavies)}): {real_heavies}")
    
    # Filter bonds for heavy-heavy only
    print("\n--- Heavy-Heavy Bonds ---")
    for i in real_heavies:
        for j in real_heavies:
            if i < j:
                d = dist[i, j].item()
                if d < 0.16:
                    print(f"{i}-{j}: {d:.4f}")

if __name__ == "__main__":
    main()
