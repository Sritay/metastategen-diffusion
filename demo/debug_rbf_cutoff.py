
import torch
import numpy as np

def check_noise_distances(n_atoms=10, batch_size=10000, cutoff=2.0):
    # Simulate initial noise state: Standard Normal N(0,1)
    # Shape: [B, N, 3]
    noise = torch.randn(batch_size, n_atoms, 3)
    
    # Calculate pairwise distances
    # x_i: [B, N, 1, 3]
    # x_j: [B, 1, N, 3]
    dist = torch.norm(noise[:, :, None, :] - noise[:, None, :, :], dim=-1) # [B, N, N]
    
    # Flatten and look at unique pairs (upper triangle)
    mask = torch.triu(torch.ones(n_atoms, n_atoms), diagonal=1).bool()
    distances = dist[:, mask] # [B, N*(N-1)/2]
    distances = distances.flatten()
    
    mean_d = distances.mean().item()
    std_d = distances.std().item()
    max_d = distances.max().item()
    min_d = distances.min().item()
    
    in_range = (distances <= cutoff).float().mean().item()
    
    print(f"--- Noise Distance Analysis (N={n_atoms}) ---")
    print(f"Mean Distance: {mean_d:.4f}")
    print(f"Std Dev:       {std_d:.4f}")
    print(f"Min / Max:     {min_d:.4f} / {max_d:.4f}")
    print(f"RBF Cutoff:    {cutoff}")
    print(f"Fraction Inside Cutoff: {in_range*100:.2f}%")
    
    if in_range < 0.5:
        print("\nCRITICAL: Majority of initial noise edges are INVISIBLE to the model!")
        print("The model cannot see the structure to denoise it.")

if __name__ == "__main__":
    check_noise_distances(cutoff=2.0)
