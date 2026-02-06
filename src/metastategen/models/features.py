import torch
from typing import Optional, List, Dict

def compute_active_chiral_features(x: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Computes "Active" Chiral Features for each node using Moment Invariants.
    
    Args:
        x: [B, N, 3]
        scale_factor: Data scale factor. If > 1, x is unscaled internally for feature computation
                      to match the "expected" physical scale (~0.15nm bonds).
        
    Returns:
        h_chiral: [B, N, 1]
    """
    B, N, _ = x.shape
    
    # 0. Handle Scaling
    if scale_factor != 1.0:
        x_in = x / scale_factor
    else:
        x_in = x
    
    # 1. Center of Mass (on UN-SCALED x_in)
    # Note: Chiral features are translation invariant, so centering works on x_in.
    c = x_in.mean(dim=1, keepdim=True)
    r = x_in - c # [B, N, 3] (Real scale ~ Angstrom/nm)
    
    # 2. Distances for weights
    # d_sq[b, i, j] = |x_i - x_j|^2
    x_i = x_in.unsqueeze(2) # [B, N, 1, 3]
    x_j = x_in.unsqueeze(1) # [B, 1, N, 3]
    dist_sq = torch.sum((x_i - x_j)**2, dim=-1) # [B, N, N]
    
    # Weights: exp(-dist_sq)
    # Average bond ~0.15nm. d^2 ~ 0.02. exp(-0.02) ~ 1.
    # Sensitivity range ~0.3-0.5nm.
    weights = torch.exp(-dist_sq) 
    
    # Mask self-loops
    mask = torch.eye(N, device=x.device).unsqueeze(0) # [1, N, N]
    weights = weights * (1.0 - mask)
    
    # 3. Compute S1 (Linear Weights)
    # [B, N, N] * [B, 1, N, 3] -> [B, N, 3]
    S1 = torch.einsum("bij,bjd->bid", weights, r)
    
    # 4. Compute S2 (Quadratic Weights - sharper profile)
    S2 = torch.einsum("bij,bjd->bid", weights**2, r)
    
    # 5. Cross Product S1 x S2
    cross = torch.cross(S1, S2, dim=-1) # [B, N, 3]
    
    # 6. Dot with r_i (Vector to COM)
    # V_i = (S1 x S2) . r_i
    V = torch.sum(cross * r, dim=-1, keepdim=True) # [B, N, 1]
    
    # 7. Normalization
    # Magnitudes observed ~ 0.0015 (Raw).
    # Since we unscaled, V is in raw units (L^3).
    # So the scaling factor 5000 is still appropriate.
    V = V * 5000.0
    
    # Clamp for stability
    V = torch.clamp(V, -5.0, 5.0)
    
    return V


def compute_chiral_volume_signal(x: torch.Tensor, scale_factor: float = 1.0, chirality_config: Optional[List[Dict]] = None) -> torch.Tensor:
    """
    Computes the EXPLICIT Scalar Triple Product (Signed Volume) for Chiral Centers.
    
    If chirality_config is provided, computes volume for each specified center.
    If NOT provided, attempts to fallback to Alanine Dipeptide hardcoded logic (idx 4,3,5,6).
    
    Args:
        x: [B, N_atoms, 3]
        scale_factor: Data scale factor for un-scaling.
        chirality_config: List of dicts, each with keys:
                          "center_idx": int
                          "neighbors": List[int] [nA, nB, nC]
        
    Returns:
        signal: [B, N_atoms, 1] 
                Populates the volume at the 'center_idx' node.
                Other nodes may be zero or receive a broadcasted value depending on strategy.
    """
    B, N, _ = x.shape
    device = x.device
    
    # Operations on unscaled data for physical consistency
    if scale_factor != 1.0:
        x_in = x / scale_factor
    else:
        x_in = x
        
    signal = torch.zeros(B, N, 1, device=device)
    
    if chirality_config is None:
        # Backward compatibility / Fallback for Ala2
        # N(3), CA(4), CB(5), C(6)
        if N > 6:
            chirality_config = [{
                "center_idx": 4,
                "neighbors": [3, 5, 6]
            }]
        else:
            return signal

    for cfg in chirality_config:
        c_idx = cfg["center_idx"]
        n_idxs = cfg["neighbors"] # [n1, n2, n3]
        
        if c_idx >= N or any(ni >= N for ni in n_idxs):
            continue

        r_CA = x_in[:, c_idx]    # [B, 3]
        r_n1 = x_in[:, n_idxs[0]]
        r_n2 = x_in[:, n_idxs[1]]
        r_n3 = x_in[:, n_idxs[2]]
        
        # Vectors from CA
        v1 = r_n1 - r_CA
        v2 = r_n2 - r_CA
        v3 = r_n3 - r_CA
        
        # Scalar Triple Product: V = v1 . (v2 x v3)
        cross_product = torch.cross(v2, v3, dim=-1) # [B, 3]
        volume = torch.sum(v1 * cross_product, dim=-1, keepdim=True) # [B, 1]
        
        # Normalize: * 1000.0 (roughly nm^3 -> unit range)
        val = volume * 1000.0
        
        # Assign to the Center Node
        signal[:, c_idx, :] = val
        
        # Optional: diffuse/broadcast to neighbors? 
        # For simple AL logic, we often just want a global "chiral state" indicator.
        # If we have multiple centers, local assignment is better.
        # If we have only one (Ala2), previous logic broadcasted to ALL.
        if len(chirality_config) == 1:
             signal = val.unsqueeze(1).expand(-1, N, -1)

    return signal

