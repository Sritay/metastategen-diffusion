import torch
from typing import Tuple, List

def align_and_reconstruct(
    x_gen: torch.Tensor, 
    x_template: torch.Tensor, 
    heavy_indices: List[int]
) -> torch.Tensor:
    """
    Aligns a 22-atom template to a 10-atom generated backbone and returns the reconstructed 22-atom structure.
    
    Args:
        x_gen: [B, 10, 3] Generated heavy atoms
        x_template: [22, 3] Reference structure (All atoms)
        heavy_indices: List of indices in template corresponding to heavy atoms (0..21)
        
    Returns:
        x_recon: [B, 22, 3] Reconstructed all-atom structures
    """
    B = x_gen.shape[0]
    device = x_gen.device
    
    # 1. Prepare Template
    # Template Heavy: [10, 3]
    templ_heavy = x_template[heavy_indices].to(device)
    # Template All: [22, 3]
    templ_all = x_template.to(device)
    
    # Center Template Heavy at origin
    t_mean = templ_heavy.mean(dim=0, keepdim=True) # [1, 3]
    templ_heavy_centered = templ_heavy - t_mean
    templ_all_centered = templ_all - t_mean
    
    # 2. Iterate over batch (Vectorized Kabsch is possible but loop is safer for clarity first)
    # TODO: Vectorize if slow. B is usually 100-500.
    
    # Target Heavy: [B, 10, 3]
    # Center Target
    x_mean = x_gen.mean(dim=1, keepdim=True) # [B, 1, 3]
    x_centered = x_gen - x_mean
    
    # Kabsch Algorithm: R that minimizes RMSD(X * R^T, Y)
    # H = X^T @ Y
    # U, S, Vt = SVD(H)
    # R = Vt.T @ U.T
    
    x_recon_list = []
    
    for i in range(B):
        # P: Template [10, 3], Q: Target [10, 3]
        P = templ_heavy_centered
        Q = x_centered[i]
        
        # Covariance matrix
        H = P.transpose(0, 1) @ Q
        
        U, S, V = torch.svd(H)
        
        # Rotation
        d = torch.det(V @ U.t())
        
        # Handle reflection
        E = torch.eye(3, device=device)
        if d < 0:
            E[2, 2] = -1
            
        R = V @ E @ U.t()
        
        # Apply to ALL template atoms
        # X_new = (X_old - center) @ R + new_center
        # Note: P @ R ~ Q
        
        # templ_all_centered: [22, 3]
        # Rotated: [22, 3]
        atoms_rot = templ_all_centered @ R.t() # or matmul(R, P.T).T
        
        # Translate to target center
        atoms_final = atoms_rot + x_mean[i]
        
        x_recon_list.append(atoms_final)
        
    return torch.stack(x_recon_list, dim=0)
