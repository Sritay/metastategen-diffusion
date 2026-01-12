
import torch
import numpy as np
from pathlib import Path

def rmsd(x, y):
    # x: [B, N, 3], y: [N, 3] or [1, N, 3]
    # returns [B]
    d = x - y
    d2 = torch.sum(d**2, dim=-1) # [B, N]
    mean_d2 = torch.mean(d2, dim=-1) # [B]
    return torch.sqrt(mean_d2)

def write_pdb_frame(f, pos, atom_types, model_num=1):
    f.write(f"MODEL     {model_num}\n")
    # Atom types map?
    # 22 atoms Timewarp.
    # We can just write generically C, H, O, N based on simple heuristics or just atoms.
    # Indices:
    # 0: CH3 (C)
    # 1: H
    # ...
    # Let's just use "Ar" (Argon) if types unknown, but we want pretty.
    # Timewarp types:
    # Usually: C, H, H, H, C, O, N, H, C, H, CB, H, H, H, C, O, N, H, C, H, H, H
    # We can just use "C" for all or guess.
    # Better: Use a predefined list for Ala2 (22 atoms).
    
    types = ["C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "H", "H"]
    
    for i, p in enumerate(pos):
        # ATOM id name resname chain resseq x y z occ temp element
        # name: C, CA, etc.
        elem = types[i] if i < len(types) else "X"
        name = f"{elem}{i+1}"
        x, y, z = p[0]*10, p[1]*10, p[2]*10 # nm to Angstrom
        f.write(f"ATOM  {i+1:>5} {name:<4} ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2}\n")
    
    f.write("ENDMDL\n")

def main():
    path = Path("runs/loop_b_refinement_23/refined_results.pt")
    out_pdb = Path("demo") / "refinement_movie.pdb"
    
    batch_size = 1000 # Known from script
    
    print(f"Loading {path}...")
    data = torch.load(path, map_location='cpu')
    
    initial_all = data['initial_positions'] # [10000, 22, 3]
    refined_all = data['refined_positions'] # [100, 22, 3]
    
    # Check counts
    n_init = len(initial_all)
    n_ref = len(refined_all)
    n_batches = n_init // batch_size
    
    print(f"Total Initial: {n_init}, Refined: {n_ref}")
    print(f"Batches: {n_batches} (Size {batch_size})")
    
    # Refined are stored sequentially by batch?
    # sample_refined.py appends filtered output batch by batch.
    # But filters are variable size? No, `keep_percent` creates fixed size usually?
    # args.keep_percent = 0.01. Batch=1000. Keep=10.
    # So exactly 10 refined per batch.
    
    expected_per_batch = int(batch_size * 0.01)
    print(f"Expected per batch: {expected_per_batch}")
    
    if n_ref != n_batches * expected_per_batch:
        print("WARNING: Refined count does not match expected batch structure.")
        # We might have dropped some if filtering logic dynamic.
        # But let's assume it IS structured.
        
    pairs = []
    
    curr_ref_idx = 0
    
    for b in range(n_batches):
        start_init = b * batch_size
        end_init = start_init + batch_size
        batch_init = initial_all[start_init:end_init]
        
        # How many refined in this batch?
        # We assume `expected_per_batch`.
        # To be safe, we could check if index is valid.
        
        batch_refined = refined_all[curr_ref_idx : curr_ref_idx + expected_per_batch]
        curr_ref_idx += expected_per_batch
        
        # Match each refined to its parent in batch_init
        for i, ref in enumerate(batch_refined):
            # Compute distances to ALL in batch_init
            dists = rmsd(batch_init, ref)
            best_idx = torch.argmin(dists)
            best_dist = dists[best_idx]
            
            # Parent
            parent = batch_init[best_idx]
            
            pairs.append((parent, ref))
            
            if i == 0 and b == 0:
                print(f"Match 0: RMSD {best_dist:.4f} nm")
                
    # Write PDB
    print(f"Writing {len(pairs)} pairs to {out_pdb}...")
    with open(out_pdb, 'w') as f:
        model_cnt = 1
        for parent, ref in pairs:
            # Frame 1: Unrefined
            write_pdb_frame(f, parent, None, model_cnt)
            model_cnt += 1
            # Frame 2: Refined
            write_pdb_frame(f, ref, None, model_cnt)
            model_cnt += 1
            
    print("Done.")

if __name__ == "__main__":
    main()
