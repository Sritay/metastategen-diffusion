
import torch

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


def compute_chiral_volume_signal(x: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Computes the EXPLICIT Scalar Triple Product (Signed Volume) for the Chiral Center.
    
    Definition:
        Center: CA (Idx 4)
        Neighbors: N (3), CB (5), C (6)
        
        V = (r_N - r_CA) . [ (r_CB - r_CA) x (r_C - r_CA) ]
        
    This geometric invariant strictly flips sign under mirror reflection (Enantiomers),
    robustly distinguishing L-Ala (Phi < 0) from D-Ala (Phi > 0).
    
    Args:
        x: [B, N_atoms, 3]
        scale_factor: Data scale factor for un-scaling.
        
    Returns:
        signal: [B, N_atoms, 1] 
                (Note: We return per-node shape for compatibility, but the signal is 
                 identical for all nodes in the molecule or localized to CA).
    """
    B, N, _ = x.shape
    
    # Indices for Alanine Dipeptide
    # N: 3, CA: 4, CB: 5, C: 6
    idx_CA = 4
    idx_N = 3
    idx_CB = 5
    idx_C = 6
    
    if N <= 6:
        # Fallback for subsets or debugging
        return torch.zeros(B, N, 1, device=x.device)

    # Operations on unscaled data for physical consistency
    if scale_factor != 1.0:
        x_in = x / scale_factor
    else:
        x_in = x
        
    r_CA = x_in[:, idx_CA] # [B, 3]
    r_N = x_in[:, idx_N]
    r_CB = x_in[:, idx_CB]
    r_C = x_in[:, idx_C]
    
    # Vectors from CA
    v1 = r_N - r_CA
    v2 = r_CB - r_CA
    v3 = r_C - r_CA
    
    # Scalar Triple Product
    # V = v1 . (v2 x v3)
    cross_product = torch.cross(v2, v3, dim=-1) # [B, 3]
    volume = torch.sum(v1 * cross_product, dim=-1, keepdim=True) # [B, 1]
    
    # Normalize magnitude to be ~O(1) for the diffusion model conditioning.
    # Typical volume ~ 0.003 (nm^3).
    # We multiply by 1000 to bring it to ~3.0, which is a good range for NN inputs.
    
    signal = volume * 1000.0 # [B, 1]
    
    # Broadcast to all nodes [B, N, 1]
    # Ideally simpler models might only need it on CA, 
    # but the architecture expects per-node features.
    signal = signal.unsqueeze(1).expand(-1, N, -1)
    
    return signal

