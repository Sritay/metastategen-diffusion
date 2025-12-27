import torch

def compute_dihedrals(pos: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Compute dihedral angles for a batch of coordinates.
    
    Args:
        pos: [B, N, 3] or [N, 3] Coordinates
        indices: [K, 4] Indices of atoms (p0, p1, p2, p3) for K dihedrals.
        
    Returns:
        angles: [B, K] in radians, range [-pi, pi]
    """
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    
    # pos: [B, N, 3]
    # Gather atoms: [B, K, 4, 3]
    # We want p0, p1, p2, p3 for each dihedral
    p0 = pos[:, indices[:, 0]] # [B, K, 3]
    p1 = pos[:, indices[:, 1]]
    p2 = pos[:, indices[:, 2]]
    p3 = pos[:, indices[:, 3]]
    
    # Vectors
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    
    # Normalize b1 so that it does not influence magnitude of vector rejections
    b1_norm = torch.norm(b1, dim=-1, keepdim=True) + 1e-7
    b1_u = b1 / b1_norm
    
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 - (b0 . b1_u) * b1_u
    v = b0 - torch.sum(b0 * b1_u, dim=-1, keepdim=True) * b1_u
    
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 - (b2 . b1_u) * b1_u
    w = b2 - torch.sum(b2 * b1_u, dim=-1, keepdim=True) * b1_u
    
    # Angle between v and w
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1_u, v, dim=-1) * w, dim=-1)
    
    return torch.atan2(y, x)

def rad2deg(x: torch.Tensor) -> torch.Tensor:
    return x * 180.0 / 3.1415926535
