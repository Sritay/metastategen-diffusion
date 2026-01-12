
import torch
import numpy as np
from pathlib import Path
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

def check_bond_lengths(samples_path: str):
    print(f"Checking bond lengths for: {samples_path}")
    samples = torch.load(samples_path) # [N, 10, 3]
    
    # Ala2 Heavy Atoms:
    # 0: CH3 (ACE)
    # 1: C (ACE)
    # 2: O (ACE)
    # 3: N (ALA)
    # 4: CA (ALA)
    # 5: CB (ALA)
    # 6: C (ALA)
    # 7: O (ALA)
    # 8: N (NME)
    # 9: CH3 (NME)
    
    # Key Bonds:
    # N-CA (3-4) ~ 0.146 nm
    # CA-C (4-6) ~ 0.151 nm
    # CA-CB (4-5) ~ 0.153 nm
    
    pairs = [
        (3, 4, "N-CA"),
        (4, 6, "CA-C"),
        (4, 5, "CA-CB")
    ]
    
    for idx1, idx2, name in pairs:
        p1 = samples[:, idx1, :]
        p2 = samples[:, idx2, :]
        dist = torch.norm(p1 - p2, dim=-1) # [N]
        
        mean = dist.mean().item()
        std = dist.std().item()
        
        print(f"Bond {name}: Mean = {mean:.4f} nm, Std = {std:.4f} nm")
        
        if std > 0.05:
            print("  -> WARNING: High variance! Bond breaking?")
        if mean < 0.1 or mean > 0.2:
            print("  -> WARNING: Unphysical length!")

if __name__ == "__main__":
    check_bond_lengths("runs/day11_al_23_hpc/iter_20/eval_samples.pt")
