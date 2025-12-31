import torch
import numpy as np

path = "runs/loop_b_final/refined_samples.pt"
print(f"Loading {path}...")
x = torch.load(path) # [B, 22, 3]

print(f"Shape: {x.shape}")
print(f"Contains NaNs: {torch.isnan(x).any()}")
print(f"Contains Infs: {torch.isinf(x).any()}")

full_mean = x.mean().item()
full_std = x.std().item()
full_min = x.min().item()
full_max = x.max().item()

print(f"Mean: {full_mean:.4f}")
print(f"Std:  {full_std:.4f}")
print(f"Min:  {full_min:.4f}")
print(f"Max:  {full_max:.4f}")

# Check heavy atom distances (C-N bond, approx 1.33 A)
# Index 4 (C) to 6 (N)
dist_cn = torch.norm(x[:, 4] - x[:, 6], dim=-1)
print(f"C-N Distance: Mean={dist_cn.mean().item():.4f}, Std={dist_cn.std().item():.4f}")
