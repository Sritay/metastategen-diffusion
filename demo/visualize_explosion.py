
import torch
import numpy as np
from pathlib import Path

def write_pdb_frame(f, pos, model_num=1):
    f.write(f"MODEL     {model_num}\n")
    types = ["C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "H", "H"]
    
    for i, p in enumerate(pos):
        elem = types[i] if i < len(types) else "X"
        name = f"{elem}{i+1}"
        # Convert nm to Angstrom
        x, y, z = p[0]*10, p[1]*10, p[2]*10 
        f.write(f"ATOM  {i+1:>5} {name:<4} ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2}\n")
    
    f.write("ENDMDL\n")

def main():
    path = Path("runs/loop_b_refinement_23/refined_results.pt")
    print(f"Loading {path}...")
    data = torch.load(path, map_location='cpu')
    initial = data['initial_positions'] # [N, 22, 3]
    
    # Calculate Max Coordinate Magnitude per structure
    # max(|x|, |y|, |z|)
    mags = torch.abs(initial).amax(dim=(1, 2)) # [N]
    
    # Sort descending
    vals, indices = torch.sort(mags, descending=True)
    
    out_pdb = "demo/exploded_mess.pdb"
    print(f"Writing top 10 worst structures to {out_pdb}")
    
    with open(out_pdb, "w") as f:
        for i in range(10):
            idx = indices[i]
            val = vals[i]
            print(f"Rank {i+1}: Index {idx}, MaxCoord {val:.4f} nm")
            
            # Write to PDB
            write_pdb_frame(f, initial[idx], model_num=i+1)
            
    print("Done.")

if __name__ == "__main__":
    main()
