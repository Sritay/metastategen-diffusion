import torch
from typing import Tuple, List

def _build_robust_frame(v1, v2):
    """
    Build an orthonormal frame from two direction vectors.
    
    Handles the degenerate case where v1 ≈ v2 (nearly parallel),
    which would make cross(v1, v2) ≈ 0 and destroy the frame.
    
    When degenerate, uses the least-aligned cardinal axis as a 
    fallback to construct a valid perpendicular vector.
    
    Args:
        v1: Primary axis [..., 3] (assumed unit-length)
        v2: Secondary direction [..., 3] (assumed unit-length)
    
    Returns:
        Frame matrix [..., 3, 3] with orthonormal columns [u1, u2, u3]
    """
    u1 = v1
    u3 = torch.cross(v1, v2, dim=-1)
    u3_norm = u3.norm(dim=-1, keepdim=True)
    
    # Detect degeneracy: cross product too small
    DEGEN_THRESH = 0.1
    is_degen = (u3_norm < DEGEN_THRESH).squeeze(-1)  # [...] bool
    
    if is_degen.any():
        # Find the cardinal axis least aligned with v1 to guarantee a good cross product
        # For each degenerate vector, pick the axis with smallest |dot(v1, axis)|
        v1_abs = v1.abs()
        
        if v1.dim() == 1:
            # Single vector case (template)
            min_idx = v1_abs.argmin()
            fallback = torch.zeros(3, device=v1.device)
            fallback[min_idx] = 1.0
            u3 = torch.cross(v1, fallback, dim=-1)
        else:
            # Batched case (generated)
            min_idx = v1_abs.argmin(dim=-1)  # [B]
            fallback = torch.zeros_like(v1)  # [B, 3]
            fallback.scatter_(-1, min_idx.unsqueeze(-1), 1.0)
            u3_fallback = torch.cross(v1, fallback, dim=-1)
            # Only replace degenerate entries
            u3 = torch.where(is_degen.unsqueeze(-1), u3_fallback, u3)
    
    u3 = u3 / (u3.norm(dim=-1, keepdim=True) + 1e-8)
    u2 = torch.cross(u3, u1, dim=-1)
    return torch.stack([u1, u2, u3], dim=-1)



def align_and_reconstruct(
    x_gen: torch.Tensor, 
    x_template: torch.Tensor, 
    heavy_indices: List[int],
    topology = None # Optional arg, but we need it now.
) -> torch.Tensor:
    """
    Reconstructs all-atom structure from heavy-atom backbone using Local Frame Alignment.
    
    Args:
        x_gen: [B, N_heavy, 3] Generated heavy atoms
        x_template: [N_total, 3] Reference structure (All atoms)
        heavy_indices: List of indices in template corresponding to heavy atoms
        topology: MoleculeTopology object (Optional, but required for correct local reconstruction)
        
    Returns:
        x_recon: [B, N_total, 3]
    """
    B, N_heavy, _ = x_gen.shape
    N_total = x_template.shape[0]
    device = x_gen.device
    
    # Initialize output with template (replicated)
    x_recon = x_template.unsqueeze(0).repeat(B, 1, 1).to(device) # [B, N_all, 3]
    
    # 1. Place Heavy Atoms (Direct Copy)
    x_recon[:, heavy_indices, :] = x_gen
    
    if topology is None:
        # Fallback to Global Kabsch if no topology provided (Legacy behavior)
        from metastategen.utils import get_logger
        log = get_logger("reconstruct")
        log.warning("No topology provided to align_and_reconstruct! Using Global Kabsch (Bad for flexible molecules).")
        
        # Global align template to x_gen[i]
        templ_heavy = x_template[heavy_indices].to(device) # [N_heavy, 3]
        templ_center = templ_heavy.mean(dim=0)
        
        for i in range(B):
            # Center both
            tgt = x_gen[i] # [N_heavy, 3]
            tgt_center = tgt.mean(dim=0)
            
            P = templ_heavy - templ_center
            Q = tgt - tgt_center
            
            H = P.T @ Q
            U, S, V = torch.svd(H) # deprecated but works
            d = torch.det(V @ U.T)
            E = torch.eye(3, device=device)
            if d < 0: E[2, 2] = -1
            R = V @ E @ U.T
            
            # Transform ALL atoms of template
            all_rot = (x_template.to(device) - templ_center) @ R + tgt_center
            
            # Mask for hydrogens
            all_indices = set(range(N_total))
            heavy_set = set(heavy_indices)
            h_indices = list(all_indices - heavy_set)
            
            x_recon[i, h_indices] = all_rot[h_indices]
            
        return x_recon

    # 2. Local Alignment for Hydrogens
    # Mapping: Global Index -> Index in Heavy Tensor (0..N_heavy-1)
    global_to_local_heavy = {idx: i for i, idx in enumerate(heavy_indices)}
    
    # Build Adjacency Graph from Topology
    adj = {i: [] for i in range(N_total)}
    for bond in topology.traj.topology.bonds:
        adj[bond.atom1.index].append(bond.atom2.index)
        adj[bond.atom2.index].append(bond.atom1.index)
        
    for p_global in heavy_indices:
        p_idx = global_to_local_heavy[p_global]
        
        # Identify children (H) and reference neighbors (Heavy)
        children_h = []
        refs_heavy = []
        
        for neighbor_idx in adj[p_global]:
            if neighbor_idx in global_to_local_heavy:
                refs_heavy.append(neighbor_idx)
            else:
                # Assume it's a hydrogen if not in heavy_indices
                children_h.append(neighbor_idx)
        
        if not children_h:
            continue # No hydrogens to place for this atom
            
        # Define Local Frame
        # Coordinates in Template
        p_tpl = x_template[p_global].to(device) # [3]
        
        # Coordinates in Gen [B, 3]
        p_gen = x_gen[:, p_idx, :]
        
        if not refs_heavy:
            # Floating heavy atom with hydrogens (e.g. Methane CH4, Water OH2)
            # Local translation is enough for 0th order.
            delta = p_gen - p_tpl # [B, 3]
            for h_idx in children_h:
                h_tpl = x_template[h_idx].to(device)
                x_recon[:, h_idx, :] = h_tpl + delta
            continue

        # We have at least 1 heavy neighbor.
        ref1 = refs_heavy[0]
        r1_idx = global_to_local_heavy[ref1]
        
        # Vector P->R1
        v1_tpl = x_template[ref1].to(device) - p_tpl
        v1_tpl = v1_tpl / (v1_tpl.norm() + 1e-8)
        
        v1_gen = x_gen[:, r1_idx, :] - p_gen # [B, 3]
        v1_gen = v1_gen / (v1_gen.norm(dim=1, keepdim=True) + 1e-8)
        
        # Get second direction vector
        v2_tpl = None
        v2_gen = None
        
        if len(refs_heavy) >= 2:
            ref2 = refs_heavy[1]
            r2_idx = global_to_local_heavy[ref2]
            v2_tpl = x_template[ref2].to(device) - p_tpl
            v2_tpl = v2_tpl / (v2_tpl.norm() + 1e-8)
            v2_gen = x_gen[:, r2_idx, :] - p_gen
            v2_gen = v2_gen / (v2_gen.norm(dim=1, keepdim=True) + 1e-8)
        else:
            # Only 1 neighbor. Use grandparent direction.
            parent_neighbors = adj[ref1]
            grandparents = [n for n in parent_neighbors if n != p_global and n in global_to_local_heavy]
            if grandparents:
                gp = grandparents[0]
                gp_idx = global_to_local_heavy[gp]
                v2_tpl = x_template[gp].to(device) - x_template[ref1].to(device)
                v2_tpl = v2_tpl / (v2_tpl.norm() + 1e-8)
                v2_gen = x_gen[:, gp_idx, :] - x_gen[:, r1_idx, :]
                v2_gen = v2_gen / (v2_gen.norm(dim=1, keepdim=True) + 1e-8)

        if v2_tpl is not None and v2_gen is not None:
            F_tpl = _build_robust_frame(v1_tpl, v2_tpl)         # [3, 3]
            F_gen = _build_robust_frame(v1_gen, v2_gen)          # [B, 3, 3]
            R = F_gen @ F_tpl.T.unsqueeze(0)                     # [B, 3, 3]
        else:
            # No second vector available at all — identity fallback
            R = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)

        # Apply R to Hydrogens
        for h_idx in children_h:
            h_tpl = x_template[h_idx].to(device)
            rel_vec = h_tpl - p_tpl
            
            # R [B, 3, 3], rel [3] -> [B, 3]
            rel_rot = (R @ rel_vec.unsqueeze(-1)).squeeze(-1)
            
            x_recon[:, h_idx, :] = p_gen + rel_rot

    return x_recon
