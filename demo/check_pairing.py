import torch
import numpy as np

def main():
    path = "runs/loop_b_refinement_23_fixed/refined_results.pt"
    print(f"Loading {path}...")
    data = torch.load(path, map_location="cpu")
    
    init = data['initial_positions']
    ref = data['refined_positions']
    
    print(f"Initial Shape: {init.shape}")
    print(f"Refined Shape: {ref.shape}")
    
    if init.shape != ref.shape:
        print("Shapes mismatch! Cannot assume 1-to-1 pairing.")
        return

    # Compute per-atom distance between i-th initial and i-th refined
    # [N, 22, 3] -> [N, 22] -> [N]
    diff = init - ref
    dists = torch.norm(diff, dim=-1).mean(dim=-1) # Mean displacement per structure
    
    print(f"Mean Displacement between indices: {dists.mean().item():.4f} nm")
    print(f"Max Displacement: {dists.max().item():.4f} nm")
    print(f"Min Displacement: {dists.min().item():.4f} nm")
    
    # If aligned, displacement should be small-ish (refinement movement).
    # If random, it would be larger (dist between random conformers).
    
    trunc_len = min(10, len(init))
    print("First 10 displacements:", dists[:trunc_len])

if __name__ == "__main__":
    main()
