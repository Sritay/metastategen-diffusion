import torch
import numpy as np
from pathlib import Path

def compute_chirality(positions):
    # expect [B, 22, 3]
    # Indices for Ala2 22-atom: N=6, CA=8, CB=10, C=14
    n = positions[:, 6]
    ca = positions[:, 8]
    cb = positions[:, 10]
    c = positions[:, 14]
    
    # Vectors
    v_n = n - ca
    v_cb = cb - ca
    v_c = c - ca
    
    # Triple scalar product: (v_n x v_c) . v_cb
    # or similar definition.
    # Standard: H = (n - ca) . cross( (c - ca), (cb - ca) ) ? 
    # Let's match typical definition: CA is center.
    # Cross product of two plane vectors, dot with third.
    
    cross = torch.cross(v_n, v_c, dim=-1)
    vol = torch.sum(cross * v_cb, dim=-1)
    
    # Sign: L-Ala should be negative? Or positive?
    # We'll check the distribution.
    return vol

def main():
    # 1. Load Template to define Topology
    templ_path = "data/timewarp/train/positions.pt"
    try:
        templ_all = torch.load(templ_path)[0] # [22, 3]
    except Exception as e:
        print(f"Error loading template: {e}")
        return

    # Infer bonds (cutoff 0.17 nm)
    dists = torch.cdist(templ_all.unsqueeze(0), templ_all.unsqueeze(0))[0]
    bonds = []
    bond_expected = []
    for i in range(22):
        for j in range(i + 1, 22):
            d = dists[i, j].item()
            if d < 0.17:
                bonds.append((i, j))
                bond_expected.append(d)
                
    print(f"Topology: {len(bonds)} bonds inferred from template.")
    
    # Check Template Chirality
    templ_vol = compute_chirality(templ_all.unsqueeze(0)).item()
    print(f"Template Chirality Volume: {templ_vol:.4f}")
    if templ_vol < 0:
        print("Template is 'L-like' (by script convention)")
    else:
        print("Template is 'D-like' (by script convention)")
    
    # 2. Load Results
    res_path = "runs/loop_b_refinement_23_fixed/refined_results.pt"
    print(f"Loading {res_path}...")
    data = torch.load(res_path, map_location='cpu')
    
    # We have 'initial_positions' (Reconstructed from Diffusion) and 'refined_positions'
    initial = data['initial_positions']
    refined = data['refined_positions']
    
    def analyze_set(name, pos):
        print(f"\n=== {name} Analysis ({pos.shape[0]} structures) ===")
        
        # A. Coordinates
        min_c = pos.min().item()
        max_c = pos.max().item()
        print(f"Coordinates Range: [{min_c:.4f}, {max_c:.4f}] nm")
        if max(abs(min_c), abs(max_c)) > 1.0:
            print("  !! WARNING: Coordinates Exploding !!")
            
        # B. Bond Lengths
        violation_counts = 0
        worst_bond = 0.0
        worst_bond_idx = -1
        
        # Check all bonds
        p1 = pos[:, [b[0] for b in bonds]]
        p2 = pos[:, [b[1] for b in bonds]]
        d = torch.norm(p1 - p2, dim=-1) # [B, n_bonds]
        
        # Compare to expected
        ref = torch.tensor(bond_expected, device=pos.device).unsqueeze(0)
        
        # Deviation
        diff = torch.abs(d - ref)
        max_diff = diff.max(dim=0)[0] # Max deviation per bond type
        mean_diff = diff.mean(dim=0)
        
        # Report
        global_max_diff = diff.max().item()
        print(f"Max Bond Deviation: {global_max_diff:.4f} nm")
        
        # Check specific bad bonds
        bad_mask = diff > 0.05 # 0.5 Angstrom tolerance
        bad_count = bad_mask.sum().item()
        total_checks = diff.numel()
        print(f"Bonds > 0.05nm deviation: {bad_count} / {total_checks} ({bad_count/total_checks*100:.4f}%)")
        
        # Top 5 offending bonds
        if bad_count > 0:
            print("  Worst Bonds:")
            vals, indices = torch.sort(max_diff, descending=True)
            for k in range(min(5, len(bonds))):
                idx = indices[k]
                if vals[k] > 0.02:
                    i, j = bonds[idx]
                    print(f"    Bond {i}-{j} (Ref {bond_expected[idx]:.3f}): Max Dev {vals[k]:.4f} nm")
                    
        # C. Chirality
        vols = compute_chirality(pos)
        n_L = (vols < 0).sum().item()
        n_D = (vols > 0).sum().item()
        print(f"Chirality: L-like (<0): {n_L}, D-like (>0): {n_D} (Ratio L: {n_L/pos.shape[0]:.2f})")
        print(f"  Volume Stats: Mean={vols.mean():.4f}, Std={vols.std():.4f}")

    analyze_set("Initial (Diffusion+Recon)", initial)
    analyze_set("Refined (Post-Force)", refined)

if __name__ == "__main__":
    main()
