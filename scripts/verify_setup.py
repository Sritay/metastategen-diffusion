import torch
from metastategen.models.force import ForceEGNN
from metastategen.models.egnn import EGNNConfig
import yaml
from pathlib import Path
import sys
import os
sys.path.append(os.getcwd())
# Also append scripts dir to find train_force if needed, or rely on root import if scripts is package
# But train_force is a script, not module. 
# Let's import it dynamically or assume the user can run it.
# Actually simpler: just redefine dataset loading logic or use sys.path.append("scripts")
sys.path.append("scripts")
from train_force import ForceDataset

def main():
    print("Verifying setup...")
    
    # 1. Load Config
    with open("configs/ala2_force.yaml") as f:
        cfg = yaml.safe_load(f)
    print("Config loaded.")

    # 2. Dataset
    print("Loading valid dataset (traj 1)...")
    # Note: process_timewarp produced shards locally in data/processed/ala2/shards
    # but the config points to data/processed/ala2/shards?
    # Let's check config paths vs actual paths.
    
    ds = ForceDataset(
        shard_dir="data/processed/ala2/shards",
        forces_path="data/processed/ala2/al_forces_ref.pt",
        trajs=[1] # Test set
    )
    print(f"Dataset loaded. Size: {len(ds)}")
    if len(ds) == 0:
        raise ValueError("Dataset is empty!")
        
    item = ds[0]
    print(f"Item 0: x={item['x'].shape}, f={item['f'].shape}, a={len(item['a'])}")
    
    # 3. Model
    print("Initializing model...")
    n_atom_types = 5
    model = ForceEGNN(
         n_atom_types=n_atom_types,
         hidden_dim=32,
         n_layers=2
    )
    
    # 4. Forward
    print("Running forward pass...")
    x = item['x'].unsqueeze(0) # [1, N, 3]
    f_tgt = item['f'].unsqueeze(0)
    a = item['a'].unsqueeze(0) # [1, N]
    
    f_pred = model(x, a)
    print(f"Prediction shape: {f_pred.shape}")
    
    assert f_pred.shape == f_tgt.shape
    print("Verification SUCCESS.")

if __name__ == "__main__":
    main()
