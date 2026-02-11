import torch
import numpy as np
from typing import Dict
from metastategen.utils import get_logger

log = get_logger("augmentation")

def augment_with_noise_and_rotations(
    data: Dict[str, torch.Tensor],
    n_copies: int = 1000,
    noise_scale: float = 0.05
) -> Dict[str, torch.Tensor]:
    """
    Augments a single-frame (or small) dataset by replicating it N times,
    adding Gaussian thermal noise, and applying random 3D rotations.

    Args:
        data: Dict containing 'positions' [N_frames, N_atoms, 3] and 'atom_types'.
        n_copies: Total number of frames to generate (approx). If input has M frames, 
                  we generate ~n_copies total by replicating each input frame n_copies/M times.
        noise_scale: Standard deviation of Gaussian noise in Angstroms (before scaling).
                     Note: If positions are already scaled (e.g. by 7.0), this should be adjusted or apply noise pre-scaling?
                     Ideally apply noise in original space? 
                     BUT the input 'data' to this function usually comes AFTER loading, so checks are needed.
                     
                     Assuming input `data` has positions in whatever unit. We apply noise of magnitude `noise_scale`.

    Returns:
        Augmented data dict.
    """
    
    pos = data["positions"] # [M, N_atoms, 3]
    m_frames = pos.shape[0]
    
    if m_frames >= n_copies:
        log.info(f"Dataset has {m_frames} frames, sufficient for requested {n_copies}. Skipping augmentation.")
        return data

    copies_per_frame = max(1, n_copies // m_frames)
    total_new = m_frames * copies_per_frame
    
    log.info(f"Augmenting {m_frames} frames to ~{total_new} frames (Copies: {copies_per_frame}, Noise: {noise_scale})")
    
    new_pos = []
    new_types = []
    
    # Replicate atom_types
    # atom_types is usually [N_atoms] (1D) for static topology or [M, N_atoms] if varying?
    # Our pipeline usually assumes fixed topology, so atom_types is 1D tensor [N_atoms] in ALDataManager,
    # but in data loader it might be returned as part of dict. 
    # Let's check input format. train.py loads it. 
    # Usually 'atom_types' is a single tensor.
    
    atom_types = data["atom_types"]
    
    for i in range(m_frames):
        base_pos = pos[i] # [N_atoms, 3]
        
        # 1. Replicate
        # [K, N, 3]
        replicated = base_pos.unsqueeze(0).repeat(copies_per_frame, 1, 1)
        
        # 2. Add Noise
        noise = torch.randn_like(replicated) * noise_scale
        noisy = replicated + noise
        
        # 3. Random Rotations
        # Generate K random rotation matrices
        rot_mats = _random_rotations(copies_per_frame, dtype=noisy.dtype, device=noisy.device) # [K, 3, 3]
        
        # Apply: (R @ x.T).T  => x @ R.T
        # [K, N, 3] @ [K, 3, 3] -> [K, N, 3]
        # We need to treat each frame separately or batch multiply
        rot_mats_t = rot_mats.transpose(1, 2)
        rotated = torch.bmm(noisy, rot_mats_t)
        
        new_pos.append(rotated)
    
    augmented_pos = torch.cat(new_pos, dim=0)
    
    out_data = {
        "positions": augmented_pos,
        "atom_types": atom_types # Keep original, Dataset wrapper handles it
    }
    
    # Propagate other keys if they exist and match M dimension?
    # e.g. energies.
    # For now, simplistic approach: only augment positions.
    # If energies present, they are invalid for noisy/rotated structures unless we recompute!
    # BUT diffusion training doesn't strictly need energies unless we do guided.
    # We should DROP energies if they exist, because the noise invalidates them.
    
    if "energies" in data:
        log.warning("Dropping 'energies' from augmented data as thermal noise invalidates them.")
        
    return out_data


def _random_rotations(n: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Generates N random 3D rotation matrices using Gram-Schmidt (QR decomposition).
    This ensures uniform sampling from SO(3).
    """
    # 1. Random Gaussian matrices [N, 3, 3]
    h = torch.randn(n, 3, 3, dtype=dtype, device=device)
    
    # 2. QR Decomposition
    q, r = torch.linalg.qr(h)
    
    # 3. Ensure proper rotation (det=1) and uniqueness of QR
    # QR is unique if diagonal of R is positive.
    d = torch.diagonal(r, dim1=-2, dim2=-1) # [N, 3]
    sign = d.sign() # [N, 3]
    
    # Multiply columns of Q by sign of R diagonal to fix uniqueness
    q = q * sign.unsqueeze(1)
    
    # 4. Correct determinant to +1
    det = torch.linalg.det(q) # [N]
    
    # If det is -1, flip sign of *all* columns (reflection -> rotation)
    # wait, flipping all columns flips det ONLY if dimension is odd (3x3).
    # (-1)^3 = -1. Correct.
    mask = (det < 0).float().view(n, 1, 1) # 1 where det=-1
    # q = (1 - 2*mask) * q # If mask=1 -> -q
    
    # Simpler: If det < 0, swap two columns or flip one column?
    # Flipping one column changes det sign.
    # q[:, :, 0] *= det.sign().view(n, 1) # Multiply col 0 by sign(det)
    
    # Let's verify standard approach:
    # If det(Q) < 0, negate the first column.
    q[:, :, 0] *= det.sign().unsqueeze(1)
    
    return q
