
import torch
from metastategen.models.features import compute_active_chiral_features

def debug_active_chirality_internals():
    print("Debugging Active Chirality Internals...")
    
    # Create a simplified Chiral Tetrahedron
    # CA at origin
    x = torch.zeros(1, 4, 3) # ONLY 4 atoms to keep it simple
    x[0, 0] = torch.tensor([0.0, 0.0, 0.0]) # Center (CA) -> Index 0
    x[0, 1] = torch.tensor([1.0, 0.0, 0.0]) # Right (N) -> Index 1
    x[0, 2] = torch.tensor([0.0, 1.2, 0.0]) # Up (C) -> Index 2 (Diff length)
    x[0, 3] = torch.tensor([0.0, 0.0, 0.8]) # Forward (CB) -> Index 3 (Diff length)
    
    # COM will be (0.25, 0.25, 0.25)
    
    # Let's inspect line by line (copying logic from features.py)
    B, N, _ = x.shape
    
    # 1. Center of Mass
    c = x.mean(dim=1, keepdim=True)
    r = x - c 
    print(f"COM: {c}")
    print(f"r:\n{r}")
    
    # 2. Distances
    x_i = x.unsqueeze(2) 
    x_j = x.unsqueeze(1) 
    dist_sq = torch.sum((x_i - x_j)**2, dim=-1) 
    weights = torch.exp(-dist_sq) 
    mask = torch.eye(N).unsqueeze(0)
    weights = weights * (1.0 - mask)
    print(f"Weights (row 0):\n{weights[0,0]}")
    
    # New Formulation: Moment Invariants
    # S1 = Sum_j w_ij * r_j
    S1 = torch.einsum("bij,bjd->bid", weights, r)
    
    # S2 = Sum_j (w_ij ** 2) * r_j  (Different radial profile)
    S2 = torch.einsum("bij,bjd->bid", weights**2, r)
    
    print(f"S1 (Index 0):\n{S1[0,0]}")
    print(f"S2 (Index 0):\n{S2[0,0]}")
    
    # Cross product S1 x S2
    cross = torch.cross(S1, S2, dim=-1)
    
    # Dot with r_i (Vector from COM to node i)
    # V = (S1 x S2) . r_i
    V = torch.sum(cross * r, dim=-1, keepdim=True)
    
    print(f"V (Index 0): {V[0,0].item()}")
    
    # Scale?
    print(f"V scaled (x1000): {V[0,0].item() * 1000}")
    
    return V

if __name__ == "__main__":
    debug_active_chirality_internals()
