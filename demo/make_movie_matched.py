import torch
import numpy as np

def compute_rmsd(pos_a, pos_b):
    # pos_a: [22, 3]
    # pos_b: [N, 22, 3]
    diff = pos_a.unsqueeze(0) - pos_b
    dist_sq = torch.sum(diff**2, dim=-1)
    mean_dist_sq = torch.mean(dist_sq, dim=-1)
    rmsd = torch.sqrt(mean_dist_sq)
    return rmsd

def write_pdb_model(f, pos, model_num=1, title=""):
    f.write(f"MODEL     {model_num}\n")
    if title:
        f.write(f"REMARK    {title}\n")
    
    # Corrected Order (H1, CH3...)
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
        f.write(f"ATOM  {i+1:>5} {atom_name:<4} {res_name} A{res_id:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2}\n")
        
    f.write("ENDMDL\n")

def main():
    path = "runs/loop_b_refinement_23_fixed/refined_results.pt"
    print(f"Loading {path}...")
    data = torch.load(path, map_location="cpu")
    
    init_all = data['initial_positions']   # [10000, 22, 3]
    refined_all = data['refined_positions'] # [100, 22, 3]
    
    out_pdb = "demo/refinement_movie_fixed.pdb"
    print(f"Matching parents and writing to {out_pdb}...")
    
    with open(out_pdb, "w") as f:
        # For each refined structure
        for i in range(len(refined_all)):
            ref = refined_all[i]
            
            # Find closest parent in init_all
            # Since init structures are now physical (fixed), RMSD should be small.
            dists = compute_rmsd(ref, init_all)
            best_idx = torch.argmin(dists).item()
            best_rmsd = dists[best_idx].item()
            parent = init_all[best_idx]
            
            print(f"Refined {i}: Matched Parent {best_idx} with RMSD {best_rmsd:.4f} nm")
            
            # Write Parent (Odd Frame)
            write_pdb_model(f, parent, model_num=2*i+1, title=f"Generated (Parent {best_idx})")
            
            # Write Refined (Even Frame)
            write_pdb_model(f, ref, model_num=2*i+2, title=f"Refined {i} (RMSD {best_rmsd:.4f})")
            
    print("Done.")

if __name__ == "__main__":
    main()
