import pytest
import torch
import yaml
import sys
import numpy as np
from pathlib import Path
from unittest.mock import patch

# Verify we can import the training script
# We might need to adjust sys.path if scripts/ is not a package, 
# but installing via pip -e . usually handles src/, not scripts/.
# Scripts are usually run via python scripts/foo.py.
# However, for testing, we can import them if we make them importable or use subprocess.
# Let's try attempting import by modifying path temporarily or assuming scripts is in root.
# Since we are in tests/, root is up one level.

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR / "scripts"))

try:
    import train_diffusion
except ImportError:
    # If scripts/ is not a python package, we might fail.
    # But usually pytest handles this if __init__.py exists or simple path resolution.
    pass

@pytest.fixture
def synthetic_data(tmp_path):
    """
    Creates a temporary directory with synthetic .npz and .pdb files.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # NPZ Data
    # 100 frames, 10 atoms, 3 dims
    positions = torch.randn(100, 10, 3).numpy()
    atom_types = torch.tensor([1, 0, 0, 2, 1, 1, 1, 2, 2, 1]).numpy() # 10 atoms
    
    np.savez(data_dir / "test.npz", positions=positions, atom_types=atom_types)
    
    # Dummy PDB
    pdb_content = """CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  
ATOM      2  H   ALA A   1       0.000   0.000   0.000  1.00  0.00           H  
ATOM      3  H   ALA A   1       0.000   0.000   0.000  1.00  0.00           H  
ATOM      4  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  
ATOM      5  C   ALA A   1       0.000   0.000   0.000  1.00  0.00           C  
ATOM      6  O   ALA A   1       0.000   0.000   0.000  1.00  0.00           O  
ATOM      7  CB  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  
ATOM      8  HB1 ALA A   1       0.000   0.000   0.000  1.00  0.00           H  
ATOM      9  HB2 ALA A   1       0.000   0.000   0.000  1.00  0.00           H  
ATOM     10  HB3 ALA A   1       0.000   0.000   0.000  1.00  0.00           H  
END
"""
    with open(data_dir / "test.pdb", "w") as f:
        f.write(pdb_content)

    return data_dir

def test_train_diffusion_smoke(synthetic_data, tmp_path):
    """
    Runs train_diffusion.py with minimal config to verify end-to-end execution.
    """
    if 'train_diffusion' not in sys.modules:
        pytest.skip("Could not import train_diffusion script")

    # Create Config
    config = {
        'train': {
            'seed': 42,
            'lr': 1e-4,
            'epochs': 1,
            'save_every': 1,
            'grad_clip': 1.0,
            'out_dir': str(tmp_path / "out")
        },
        'data': {
            'npz_path': str(synthetic_data / "test.npz"),
            'pdb_path': str(synthetic_data / "test.pdb"),
            'batch_size': 10,
            'scale_factor': 1.0,
            'num_workers': 0
        },
        'model': {
            'n_layers': 1,
            'hidden_dim': 16,
            'time_emb_dim': 16,
            'n_atom_types': 3 # Must match atom_types max+1
        },
        'diffusion': {
            'T': 10,
            'beta_start': 1e-4,
            'beta_end': 0.02,
            'schedule': 'linear',
            'recenter_every_step': True
        }
    }
    
    config_path = tmp_path / "ci_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    # Mock sys.argv
    with patch.object(sys, 'argv', ["train_diffusion.py", "--config", str(config_path)]):
        train_diffusion.main()
        
    # Assert Output
    out_dir = tmp_path / "out" / "checkpoints"
    assert out_dir.exists()
    assert (out_dir / "final.pt").exists()
