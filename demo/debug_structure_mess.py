import torch
import numpy as np

def main():
    # 1. Load Template
    templ_path = "/Users/sritaymistry/projects/metastategen-diffusion/data/timewarp/train/positions.pt"
    try:
        templ_all = torch.load(templ_path)[0] # [22, 3]
        print(f"Loaded template from {templ_path}")
    except Exception as e:
        print(f"Error loading template: {e}")
        return

    # 2. Infer Topology from Template
    # Calculate all pairwise distances
    # Ensure units are handled (assuming nm based on project docs)
    dists = torch.cdist(templ_all.unsqueeze(0), templ_all.unsqueeze(0))[0]
    
    # Cutoff: 0.17nm covers all covalent bonds (C-C ~0.15, C-H ~0.11)
    n_atoms = 22
    bonds = []
    bond_types = [] # Record approximate length to guess type
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = dists[i, j].item()
            if d < 0.17:
                bonds.append((i, j))
                bond_types.append(d)
                
    print(f"Inferred {len(bonds)} bonds from template.")
    
    # 3. Load Results
    res_path = "/Users/sritaymistry/projects/metastategen-diffusion/runs/loop_b_refinement_23/refined_results.pt"
    try:
        data = torch.load(res_path)
        initial = data['initial_positions'] # [N, 22, 3]
        refined = data['refined_positions'] # [N, 22, 3]
        print(f"Loaded results from {res_path}. Shape: {initial.shape}")
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    # 4. Analyze
    def analyze(name, positions, bond_list):
        print(f"\n--- {name} ---")
        # Gather all bond lengths
        p1 = positions[:, [b[0] for b in bond_list]]
        p2 = positions[:, [b[1] for b in bond_list]]
        lengths = torch.norm(p1 - p2, dim=-1) # [N, n_bonds]
        
        # Stats
        mean_l = lengths.mean(dim=0)
        max_l = lengths.max(dim=0)[0]
        min_l = lengths.min(dim=0)[0]
        std_l = lengths.std(dim=0)
        
        # Check coordinate range
        min_c = positions.min().item()
        max_c = positions.max().item()
        mean_c = positions.mean().item()
        std_c = positions.std().item()
        print(f"Coords: Min={min_c:.4f}, Max={max_c:.4f}, Mean={mean_c:.4f}, Std={std_c:.4f}")
        
        # Check for broken bonds
        
        # Check for broken bonds
        # Using 0.20 nm as a loose "broken" threshold
        broken_threshold = 0.20
        total_bonds = lengths.numel()
        broken_count = (lengths > broken_threshold).sum().item()
        
        print(f"Total bonds evaluated: {total_bonds}")
        print(f"Broken bonds (>{broken_threshold}nm): {broken_count} ({broken_count/total_bonds*100:.2f}%)")
        
        # Detailed stats per bond
        print("\nPer-Bond Statistics (Top 10 worst max lengths):")
        
        # Sort by max length
        sorted_indices = torch.argsort(max_l, descending=True)
        
        for k in sorted_indices[:10]:
            i, j = bond_list[k]
            print(f"Bond {i:2d}-{j:2d} (Ref ~{bond_types[k]:.3f}): Mean={mean_l[k]:.4f}, Min={min_l[k]:.4f}, Max={max_l[k]:.4f}, Std={std_l[k]:.4f}")

    analyze("Initial (Reconstructed)", initial, bonds)
    
    # Analyze Heavy-Heavy only to check Diffusion Quality
    heavy_indices = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
    heavy_bonds = [b for b in bonds if b[0] in heavy_indices and b[1] in heavy_indices]
    analyze("Initial (Heavy-Heavy Only / Diffusion Output)", initial, heavy_bonds)
    
    analyze("Refined (Post-Force)", refined, bonds)

if __name__ == "__main__":
    main()
