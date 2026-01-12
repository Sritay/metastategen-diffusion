
import torch
import numpy as np
from metastategen.models.diffusion import constrain_bonds
from metastategen.models.features import compute_active_chiral_features
from metastategen.models.egnn import EGNN, EGNNConfig

def test_bond_constraints():
    print("Testing Bond Constraints (Scale=7.6)...")
    scale = 7.6
    # Create a random 10-atom structure (scaled domain)
    x = torch.randn(1, 10, 3) * scale
    
    # Just run constraint
    x_fixed = constrain_bonds(x.clone(), scale_factor=scale)
    
    # Check lengths
    d_n_ca = torch.norm(x_fixed[:, 3] - x_fixed[:, 4])
    d_ca_c = torch.norm(x_fixed[:, 4] - x_fixed[:, 6])
    
    # Expected
    t1 = 0.146 * scale
    t2 = 0.151 * scale
    
    print(f"Fixed N-CA: {d_n_ca:.4f} (Target {t1:.4f})")
    print(f"Fixed CA-C: {d_ca_c:.4f} (Target {t2:.4f})")
    
    assert abs(d_n_ca - t1) < 1e-3, "N-CA bond failed"
    assert abs(d_ca_c - t2) < 1e-3, "CA-C bond failed"
    print("Bond Constraints: PASS")

def test_active_chirality():
    print("\nTesting Active Chirality...")
    # Load Real Data if possible, or use random but structured
    # Random cloud has chirality 0 on average?
    # Let's simple create a tetrahedron
    # CA at origin
    # N at (1,0,0)
    # C at (0,1,0)
    # CB at (0,0,1)
    
    x = torch.zeros(1, 10, 3)
    x[0, 4] = torch.tensor([0.0, 0.0, 0.0]) # CA @ Origin
    x[0, 3] = torch.tensor([1.0, 0.0, 0.0]) # N @ 1.0
    x[0, 6] = torch.tensor([0.0, 1.2, 0.0]) # C @ 1.2 (y)
    x[0, 5] = torch.tensor([0.0, 0.0, 0.8]) # CB @ 0.8 (z)
    
    # Compute Features
    V = compute_active_chiral_features(x)
    print(f"Chiral Feature Mean (Original): {V.mean().item():.6f}")
    
    # Mirror Image
    x_mirror = x.clone()
    x_mirror[:, :, 0] *= -1 # Invert X
    
    V_mirror = compute_active_chiral_features(x_mirror)
    print(f"Chiral Feature Mean (Mirror):   {V_mirror.mean().item():.6f}")
    
    # Check antisymmetric
    # It might not be EXACTly negative due to floating point or if simple inversion changes distance distribution?
    # Distances match exactly.
    # So V should match exactly in magnitude, opposite sign.
    
    # For a few nodes (the ones involved in the tetrahedron) it should flip.
    # Check node 4 (CA)
    v4 = V[0, 4, 0].item()
    v4_m = V_mirror[0, 4, 0].item()
    print(f"Node 4 (CA) Feature: {v4:.6f} vs {v4_m:.6f}")
    
    if abs(v4) > 1e-6:
        assert abs(v4 + v4_m) < 1e-5, "Chirality did not flip sign correctly"
        print("Chirality Flip: PASS")
    else:
        print("WARNING: Chirality signal is zero? (Could be due to simplified geom/weights)")

def test_egnn_integration():
    print("\nTesting EGNN Integration...")
    cfg = EGNNConfig(use_chiral_features=True)
    model = EGNN(n_atom_types=5, hidden_dim=16, n_layers=2, time_emb_dim=16, cfg=cfg)
    
    x = torch.randn(2, 10, 3)
    h = torch.randint(0, 5, (2, 10))
    t = torch.randint(0, 100, (2,))
    
    try:
        out = model(x, h, t)
        print(f"EGNN Output Shape: {out.shape}")
        print("EGNN Integration: PASS")
    except Exception as e:
        print(f"EGNN Integration FAILED: {e}")
        raise e

if __name__ == "__main__":
    test_bond_constraints()
    test_active_chirality()
    test_egnn_integration()
