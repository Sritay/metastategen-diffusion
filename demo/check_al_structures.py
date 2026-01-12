
import torch
import numpy as np
import sys
from pathlib import Path

def analyze_structure(file_path):
    print(f"Loading {file_path}")
    data = torch.load(file_path)
    
    if isinstance(data, dict):
        if 'positions' in data:
            pos = data['positions']
        else:
            print(f"Unknown dictionary keys: {data.keys()}")
            return
    else:
        pos = data

    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    
    print(f"Shape: {pos.shape}")
    
    n_samples, n_atoms, _ = pos.shape
    
    print(f"Number of samples: {n_samples}")
    print(f"Number of atoms per sample: {n_atoms}")

    if n_atoms == 10:
        print("Detected 10-atom backbone structure.")
        # Topology assumption for 10-atom heavy atom Ala2
        # Based on typical mdshare order:
        # 0: CH3 (C)
        # 1: C (carbonyl)
        # 2: O (carbonyl oxygen)
        # 3: N (amide)
        # 4: CA (alpha carbon)
        # 5: CB (beta carbon)
        # 6: C (carbonyl)
        # 7: O (carbonyl oxygen)
        # 8: N (amide)
        # 9: CH3 (C)
        
        bonds = [
            (0, 1, "C-C (first cap)"),
            (1, 2, "C=O (first)"),
            (1, 3, "C-N (peptide)"),
            (3, 4, "N-CA"),
            (4, 5, "CA-CB"),
            (4, 6, "CA-C"),
            (6, 7, "C=O (second)"),
            (6, 8, "C-N (peptide)"),
            (8, 9, "N-C (last cap)")
        ]
        
    elif n_atoms == 22:
        print("Detected 22-atom full structure.")
        # TODO: Add full atom topology if needed
        print("Skipping detailed bond analysis for 22 atoms, please verify indices.")
        return
    else:
        print(f"Unexpected atom count: {n_atoms}")
        return

    print("\n--- Bond Length Analysis (nm) ---")
    
    total_valid = 0
    total_bonds = 0

    for i, j, name in bonds:
        dist = np.linalg.norm(pos[:, i] - pos[:, j], axis=1)
        mean_d = np.mean(dist)
        std_d = np.std(dist)
        min_d = np.min(dist)
        max_d = np.max(dist)
        median_d = np.median(dist)
        p95_d = np.percentile(dist, 95)
        p99_d = np.percentile(dist, 99)
        
        valid_count = np.sum((dist > 0.1) & (dist < 0.2))
        valid_frac = valid_count / n_samples
        
        print(f"{name:15s} ({i}-{j}): Mean={mean_d:.4f} Med={median_d:.4f} P95={p95_d:.4f} P99={p99_d:.4f} [Min={min_d:.4f}, Max={max_d:.4f}]")
        print(f"    Valid (0.1-0.2nm): {valid_count}/{n_samples} ({valid_frac*100:.1f}%)")
        
        if mean_d > 0.2 or mean_d < 0.1:
             print(f"  WARNING: {name} mean length seems off!")

if __name__ == "__main__":
    # 1. Check Generated (Cluster AL Loop 11)
    base_dir = "runs/day10_al_11_hpc"
    iters = ["00", "05", "10"] 
    
    for it in iters:
        path = f"{base_dir}/iter_{it}/eval_samples.pt"
        if Path(path).exists():
            print(f"\n\n=== Analyzing Iteration {it} ===")
            analyze_structure(path)
        else:
            print(f"File not found: {path}")
