
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from metastategen.models.egnn import EGNN, EGNNConfig
from metastategen.models.diffusion import GaussianDiffusion, DiffusionConfig
from metastategen.models.pairwise import PairwiseEnergyModel
from metastategen.reconstruct import align_and_reconstruct

def test_local_refinement():
    print("--- Testing Local Refinement Pipeline ---")
    
    # 1. Configs
    diff_config_path = "configs/ala2_al_16_hpc.yaml"
    diff_ckpt_path = "runs/day10_al_16_hpc/members/m000/checkpoints/iter_20.pt"
    force_ckpt_path = "runs/energy_pairwise/best_model.pt"
    
    device = torch.device("cpu")
    
    # 2. Load Diffusion (Simulating sample_refined.py logic)
    print("Loading config...")
    with open(diff_config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    print(f"Config loaded. use_chiral_features: {cfg['model'].get('use_chiral_features')}")

    model_cfg = EGNNConfig(
        n_layers=cfg['model']['n_layers'],
        hidden_dim=cfg['model']['hidden_dim'],
        use_chiral_features=cfg['model'].get('use_chiral_features', False),
        use_rbf=cfg['model'].get('use_rbf', False),
        rbf_dim=cfg['model'].get('rbf_dim', 64),
        rbf_cutoff=cfg['model'].get('rbf_cutoff', 1.0)
    )
    
    print("Initializing EGNN...")
    model = EGNN(
        n_atom_types=3,
        hidden_dim=cfg['model']['hidden_dim'],
        n_layers=cfg['model']['n_layers'],
        time_emb_dim=cfg['model']['time_emb_dim'],
        cfg=model_cfg
    ).to(device)
    
    scale_factor = float(cfg['data'].get('scale_factor', 1.0))
    diff_cfg = DiffusionConfig(
        T=10, # Reduced for speed
        beta_start=cfg['diffusion']['beta_start'],
        beta_end=cfg['diffusion']['beta_end'],
        schedule=cfg['diffusion']['schedule'],
        recenter_every_step=cfg['diffusion']['recenter_every_step'],
        scale_factor=scale_factor
    )
    diffusion = GaussianDiffusion(diff_cfg).to(device)

    print(f"Loading Diffusion Checkpoint: {diff_ckpt_path}")
    if Path(diff_ckpt_path).exists():
        d = torch.load(diff_ckpt_path, map_location=device)
        # Handle 'model' key or raw state dict
        state_dict = d['model'] if 'model' in d else d
        try:
            model.load_state_dict(state_dict)
            print("Diffusion Checkpoint Loaded Successfully!")
        except Exception as e:
            print(f"FAILED to load diffusion checkpoint: {e}")
            return
    else:
        print("Checkpoint not found (skipping load, testing init only)")
        
    # 3. Load Force
    print(f"Loading Force Model: {force_ckpt_path}")
    force_model = PairwiseEnergyModel(n_atoms=22).to(device)
    if Path(force_ckpt_path).exists():
        d = torch.load(force_ckpt_path, map_location=device)
        force_model.load_state_dict(d['model'])
        e_std = d['e_std'].to(device)
        print("Force Model Loaded Successfully!")
    else:
        print("Force checkpoint not found, mocking params.")
        e_std = torch.tensor(1.0)
        
    # 4. Dry Run Sampling
    print("Running Dry Run Sampling (1 Batch, 2 Steps)...")
    
    # Fake atom types
    diff_types = torch.tensor([1, 1, 2, 0, 1, 1, 1, 2, 0, 1]) # 10 atoms
    B = 2
    a_batch = diff_types.unsqueeze(0).expand(B, -1)
    shape = (B, 10, 3)
    
    # Diffusion
    with torch.no_grad():
        x_10 = diffusion.p_sample_loop(model, shape, a_batch)
        x_10 = x_10 / diff_cfg.scale_factor
    print(f"Diffusion Output Shape: {x_10.shape}")
    
    # Reconstruct (Mocking template)
    print("Reconstructing...")
    # Need template
    templ_path = Path("data/timewarp/train/positions.pt")
    if templ_path.exists():
        templ_all = torch.load(templ_path)[0].to(device)
        heavy_indices = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
        x_22 = align_and_reconstruct(x_10, templ_all, heavy_indices)
        print(f"Reconstructed Shape: {x_22.shape}")
        
        # Refine
        print("Refining...")
        x_curr = x_22.clone().requires_grad_(True)
        e_norm = force_model(x_curr)
        grad = torch.autograd.grad(e_norm.sum(), x_curr)[0]
        print(f"Gradient Shape: {grad.shape}")
        print("Refinement Step Complete.")
    else:
        print("Template not found, skipping reconstruct/refine.")

    print("\n--- LOCAL TEST PASSED ---")

if __name__ == "__main__":
    test_local_refinement()
