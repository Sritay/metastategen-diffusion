
import torch
from pathlib import Path

def main():
    forces_path = "data/processed/ala2_all_atom/al_forces_ref.pt"
    if not Path(forces_path).exists():
        print("Forces file not found.")
        return
        
    print(f"Loading {forces_path}...")
    forces = torch.load(forces_path) # [N, atoms, 3]
    
    # Calculate norms
    norms = torch.norm(forces, dim=-1) # [N, atoms]
    max_force_per_frame = norms.max(dim=1).values # [N]
    
    global_max = max_force_per_frame.max().item()
    mean = norms.mean().item()
    std = norms.std().item()
    
    print(f"Force Norm Stats:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std:  {std:.4f}")
    print(f"  Max:  {global_max:.4f}")
    
    # Count outliers > 5 sigma
    threshold = mean + 5 * std
    n_outliers = (max_force_per_frame > threshold).sum().item()
    print(f"  Outliers (> {threshold:.4f}): {n_outliers} / {len(forces)}")
    
    if global_max > 1e5:
        print("CRITICAL: Massive singularity found.")
    else:
        print("Data seems within reasonable physical bounds (no singularities).")

if __name__ == "__main__":
    main()
