from __future__ import annotations
import torch

def compute_dihedrals(pos: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Computes batch dihedral angles using cross-product geometry.
    
    Formula:
        angle = atan2( (n1 x n2) . u, n1 . n2 )
    
    Where:
        u is the bond vector b1 normalized.
        n1 is normal to plane (b0, b1) -> b0 x b1
        n2 is normal to plane (b1, b2) -> b1 x b2
        
    Args:
        pos: Coordinates [B, N, 3] or [N, 3]
        indices: Indices of 4 atoms [K, 4] defining K dihedrals.
                 (atom i, j, k, l) -> b0=j-i, b1=k-j, b2=l-k
                 
    Returns:
        angles: [B, K] in radians, in range [-pi, pi].
    """
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    
    # pos: [B, N, 3]
    # indices: [K, 4]
    
    # Gather atom positions: [B, K, 3]
    p0 = pos[:, indices[:, 0]]
    p1 = pos[:, indices[:, 1]]
    p2 = pos[:, indices[:, 2]]
    p3 = pos[:, indices[:, 3]]
    
    # Bond vectors
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    
    # Normals to the planes defined by pairs of bond vectors
    # n1 = b0 x b1
    # n2 = b1 x b2
    n1 = torch.cross(b0, b1, dim=-1)
    n2 = torch.cross(b1, b2, dim=-1)
    
    # Normalize b1 to get unit vector u along the rotation axis
    # Add epsilon to avoid div by zero (though bond length 0 is unphysical)
    b1_norm = torch.norm(b1, dim=-1, keepdim=True) + 1e-8
    u = b1 / b1_norm
    
    # Torsion angle formula: atan2( (n1 x n2) . u,  n1 . n2 )
    # Term 1 (y): (n1 x n2) . u
    # Term 2 (x): n1 . n2
    
    m1 = torch.cross(n1, n2, dim=-1)
    y = torch.sum(m1 * u, dim=-1)
    x = torch.sum(n1 * n2, dim=-1)
    
    angles = torch.atan2(y, x)
    return angles
