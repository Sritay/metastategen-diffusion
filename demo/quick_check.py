import torch
import numpy as np

def main():
    res_path = "runs/test_constraints_final/refined_results.pt"
    try:
        data = torch.load(res_path)
        initial = data['initial_positions'] # [N, 22, 3]
        print(f"Loaded results from {res_path}. Shape: {initial.shape}")
        
        # Check coordinate range
        min_c = initial.min().item()
        max_c = initial.max().item()
        print(f"Coords: Min={min_c:.4f}, Max={max_c:.4f}")
        
        if max(abs(min_c), abs(max_c)) > 2.0: # 2nm is huge
             print("FAIL: Coordinates still exploding!")
        else:
             print("PASS: Coordinates within physical range.")
        
        # Check bonds if pass
        if max(abs(min_c), abs(max_c)) <= 2.0:
            # 1-4 (CH3-C)
            # diffusion indices 0-1
            # reconstruction aligns 22-atom template.
            # But wait, did I fix the diffusion model or the reconstruction?
            # I fixed diffusion. So the 10-atom backbone coming out of diffusion should be good.
            # Then reconstruction places the rest. 
            pass

    except Exception as e:
        print(f"Error loading results: {e}")

if __name__ == "__main__":
    main()
