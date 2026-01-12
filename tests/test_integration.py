import pytest
import torch
import yaml
import sys
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
    Creates a temporary directory with synthetic .pt shards and a config file.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create 5 shards
    for i in range(5):
        # 100 frames, 10 atoms, 3 dims
        positions = torch.randn(100, 10, 3)
        atom_types = torch.tensor([1, 0, 0, 2, 1, 1, 1, 2, 2, 1]) # 10 atoms
        traj_id = torch.full((100,), i)
        
        shard_data = {
            'positions': positions,
            'atom_types': atom_types,
            'traj_id': traj_id
        }
        torch.save(shard_data, data_dir / f"shard_{i}.pt")
        
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
            'data_dir': str(synthetic_data),
            'train_trajs': None, # Use all
            'batch_size': 10,
            'frame_subsample': 1,
            'num_workers': 0
        },
        'model': {
            'n_layers': 1,
            'hidden_dim': 16,
            'time_emb_dim': 16
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
