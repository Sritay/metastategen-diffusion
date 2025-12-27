import torch
import numpy as np

def main():
    forces_path = "data/processed/ala2/al_forces_ref.pt"
    print(f"Loading {forces_path}...")
    try:
        f = torch.load(forces_path)
    except FileNotFoundError:
        print("File not found. Please ensure data is downloaded.")
        return

    print(f"Shape: {f.shape}")
    
    # Statistics
    f_flat = f.view(-1)
    mean = f_flat.mean().item()
    std = f_flat.std().item()
    min_val = f_flat.min().item()
    max_val = f_flat.max().item()
    
    print(f"Mean: {mean:.4f}")
    print(f"Std:  {std:.4f}")
    print(f"Min:  {min_val:.4f}")
    print(f"Max:  {max_val:.4f}")
    
    # Norms per atom
    norms = torch.norm(f, dim=-1) # [M, N]
    max_norm = norms.max().item()
    print(f"Max Force Norm (per atom): {max_norm:.4f}")
    
    # Check for outliers (> 3 std, > 5 std, > 10 std)
    count_3std = (torch.abs(f_flat - mean) > 3*std).sum().item()
    count_5std = (torch.abs(f_flat - mean) > 5*std).sum().item()
    count_10std = (torch.abs(f_flat - mean) > 10*std).sum().item()
    
    total = f_flat.numel()
    print(f"Outliers > 3 std: {count_3std} ({count_3std/total*100:.2f}%)")
    print(f"Outliers > 5 std: {count_5std} ({count_5std/total*100:.2f}%)")
    print(f"Outliers > 10 std: {count_10std} ({count_10std/total*100:.2f}%)")

if __name__ == "__main__":
    main()
