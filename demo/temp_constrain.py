
def constrain_bonds(x: torch.Tensor) -> torch.Tensor:
    """
    Projects backbone bonds (N-CA, CA-C) to target lengths.
    Indices: N=3, CA=4, C=6.
    """
    # Valid only for Ala2 (10 atoms)
    if x.shape[1] != 10:
        return x
    
    # Constraints
    # (idx1, idx2, length)
    # N(3)-CA(4): 0.146 nm
    # CA(4)-C(6): 0.151 nm
    constraints = [
        (3, 4, 0.146),
        (4, 6, 0.151)
    ]
    
    # Iterative projection
    for _ in range(5):
        for i1, i2, dist_target in constraints:
            p1 = x[:, i1]
            p2 = x[:, i2]
            diff = p2 - p1
            dist = torch.norm(diff, dim=1, keepdim=True) + 1e-8
            
            # Correction
            delta = diff * (dist_target / dist - 1.0)
            
            x[:, i1] -= 0.5 * delta
            x[:, i2] += 0.5 * delta
    return x
