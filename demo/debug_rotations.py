
import torch
import math

def random_rotation_matrices(batch: int, device: torch.device) -> torch.Tensor:
    """Generates random 3x3 rotation matrices."""
    # Random quaternions
    u1 = torch.rand(batch, device=device)
    u2 = torch.rand(batch, device=device)
    u3 = torch.rand(batch, device=device)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)

    # Convert to rotation matrix
    R = torch.zeros((batch, 3, 3), device=device)
    
    R[:, 0, 0] = 1 - 2 * (q3**2 + q4**2)
    R[:, 0, 1] = 2 * (q2*q3 - q1*q4)
    R[:, 0, 2] = 2 * (q2*q4 + q1*q3)
    
    R[:, 1, 0] = 2 * (q2*q3 + q1*q4)
    R[:, 1, 1] = 1 - 2 * (q2**2 + q4**2)
    R[:, 1, 2] = 2 * (q3*q4 - q1*q2)
    
    R[:, 2, 0] = 2 * (q2*q4 - q1*q3)
    R[:, 2, 1] = 2 * (q3*q4 + q1*q2)
    R[:, 2, 2] = 1 - 2 * (q2**2 + q3**2)
    return R

def check_det():
    R = random_rotation_matrices(1000, torch.device("cpu"))
    dets = torch.det(R)
    print(f"Mean Det: {dets.mean().item():.4f}")
    print(f"Min  Det: {dets.min().item():.4f}")
    print(f"Max  Det: {dets.max().item():.4f}")
    
    if (dets < 0).any():
        print("CRITICAL: Found matrices with negative determinant (Reflections)!")
    else:
        print("OK: All matrices are proper rotations.")

if __name__ == "__main__":
    check_det()
