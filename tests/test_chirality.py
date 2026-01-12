
import torch
import numpy as np
from metastategen.models.features import compute_chiral_edge_features
from metastategen.models.egnn import EGNN, EGNNConfig

def test_chiral_features_properties():
    print("Testing Chiral Features Properties...")
    # Create random coordinates [B=1, N=5, 3]
    B, N = 1, 5
    x = torch.randn(B, N, 3)
    
    # Compute Features
    f_orig = compute_chiral_edge_features(x)
    
    # 1. Translation Invariance
    x_trans = x + torch.tensor([10.0, -5.0, 3.0])
    f_trans = compute_chiral_edge_features(x_trans)
    assert torch.allclose(f_orig, f_trans, atol=1e-5), "Failed Translation Invariance"
    print("Passed Translation Invariance")
    
    # 2. Rotation Invariance
    # Rotate 90 deg around Z
    R = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    x_rot = x @ R.T
    f_rot = compute_chiral_edge_features(x_rot)
    assert torch.allclose(f_orig, f_rot, atol=1e-5), f"Failed Rotation Invariance: diff {torch.max(torch.abs(f_orig - f_rot))}"
    print("Passed Rotation Invariance")
    
    # 3. Reflection Equivariance (Anti-symmetry)
    # Mirror z -> -z
    M = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])
    x_mirror = x @ M.T
    f_mirror = compute_chiral_edge_features(x_mirror)
    
    # Should be -f_orig
    assert torch.allclose(f_mirror, -f_orig, atol=1e-5), f"Failed Reflection Equivariance: Expected -F, got F_mirror. Error: {torch.max(torch.abs(f_mirror + f_orig))}"
    print("Passed Reflection Equivariance")
    
    # 4. Check Non-Zero for Chiral structure
    if torch.allclose(f_orig, torch.zeros_like(f_orig), atol=1e-6):
        print("WARNING: Features are all zero! (Planar or Broken)")
    else:
        print(f"Features are non-zero. Range: {f_orig.min():.4f} to {f_orig.max():.4f}")

def test_egnn_integration():
    print("\nTesting EGNN Integration...")
    cfg = EGNNConfig(use_chiral_features=True, hidden_dim=16)
    model = EGNN(n_atom_types=5, hidden_dim=16, n_layers=2, time_emb_dim=16, cfg=cfg)
    
    B, N = 2, 5
    x = torch.randn(B, N, 3)
    h = torch.randint(0, 5, (B, N))
    t = torch.randint(0, 100, (B,))
    
    try:
        out = model(x, h, t)
        print(f"Forward Pass Success. Output shape: {out.shape}")
        assert out.shape == (B, N, 3)
    except Exception as e:
        print(f"Forward Pass Failed: {e}")
        raise e

if __name__ == "__main__":
    test_chiral_features_properties()
    test_egnn_integration()
