import torch
import numpy as np
from pathlib import Path

def write_pdb_model(f, pos, model_num=1):
    f.write(f"MODEL     {model_num}\n")
    # Ala 2 (22 atoms) standard names
    # Index map based on previous checks:
    # 0: CH3 (ACE)
    # ...
    # Let's use the list from `demo/recover_trajectory.py` or standard CHARMM/AMBER names if possible.
    # Updated based on ad1-traj-state0.pdb
    # 0: H1, 1: CH3, 2: H2, 3: H3, 4: C, 5: O
    types = ["H", "C", "H", "H", "C", "O", "N", "H", "C", "H", "C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "H", "H"]
    names = [
        "H1", "CH3", "H2", "H3", "C", "O",  # ACE (0-5)
        "N", "H", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O", # ALA (6-15)
        "N", "H", "C", "H1", "H2", "H3" # NME (16-21)
    ]
    resnames = ["ACE"]*6 + ["ALA"]*10 + ["NME"]*6
    resids = [1]*6 + [2]*10 + [3]*6
    
    for i, p in enumerate(pos):
        atom_name = names[i]
        res_name = resnames[i]
        res_id = resids[i]
        elem = types[i]
        
        # nm to Angstrom
        x, y, z = p[0]*10, p[1]*10, p[2]*10
        
        # Format: ATOM <id> <name> <res> <chain> <resid> <x> <y> <z> <occ> <temp> <elem>
        # Note: PDB format is strict.
        f.write(f"ATOM  {i+1:>5} {atom_name:<4} {res_name} A{res_id:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2}\n")
        
    f.write("ENDMDL\n")

def main():
    pt_path = Path("runs/loop_b_refinement_23_fixed/refined_results.pt")
    out_path = Path("demo/refined_structures_23_fixed.pdb")
    
    print(f"Loading {pt_path}...")
    data = torch.load(pt_path, map_location="cpu")
    refined = data['refined_positions'] # [N, 22, 3]
    
    print(f"Exporting {len(refined)} structures to {out_path}...")
    
    with open(out_path, "w") as f:
        for i, pos in enumerate(refined):
            write_pdb_model(f, pos, model_num=i+1)
            
    print("Done.")

if __name__ == "__main__":
    main()
