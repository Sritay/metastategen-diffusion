
import torch
import numpy as np

def check_stats(path):
    print(f"Loading {path}...")
    data = torch.load(path)
    pos = data["positions"]
    
    # recenter if needed (usually handled by diffusion model, but good to check raw distribution)
    pos_centered = pos - pos.mean(dim=1, keepdim=True)
    
    std = pos_centered.std()
    mean_abs = pos_centered.abs().mean()
    max_val = pos_centered.abs().max()
    
    print(f"Data Shape: {pos.shape}")
    print(f"Global STD: {std:.4f}")
    print(f"Global MeanAbs: {mean_abs:.4f}")
    print(f"Global Max: {max_val:.4f}")
    
    target_std = 1.0
    suggested_scale = target_std / std
    print(f"Suggested Scale Factor (to reach std=1.0): {suggested_scale:.4f}")

if __name__ == "__main__":
    check_stats("data/processed/ala2/split_5/al_pool_ref.pt")
