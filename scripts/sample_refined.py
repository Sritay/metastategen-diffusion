import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from metastategen.utils import get_logger, set_deterministic
from metastategen.models.egnn import EGNN, EGNNConfig
from metastategen.models.diffusion import GaussianDiffusion, DiffusionConfig
from metastategen.models.pairwise import PairwiseEnergyModel
from metastategen.reconstruct import align_and_reconstruct

log = get_logger("sample_refined")

def load_diffusion_model(config_path, ckpt_path, device):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    model_cfg = EGNNConfig(
        n_layers=cfg['model']['n_layers'],
        hidden_dim=cfg['model']['hidden_dim']
    )
    # Hardcoded for 10-atom backbone models
    n_atom_types = 5 
    
    model = EGNN(
        n_atom_types=n_atom_types,
        hidden_dim=cfg['model']['hidden_dim'],
        n_layers=cfg['model']['n_layers'],
        time_emb_dim=cfg['model']['time_emb_dim'],
        cfg=model_cfg
    ).to(device)
    
    diff_cfg = DiffusionConfig(
        T=cfg['diffusion']['T'],
        beta_start=cfg['diffusion']['beta_start'],
        beta_end=cfg['diffusion']['beta_end'],
        schedule=cfg['diffusion']['schedule'],
        recenter_every_step=cfg['diffusion']['recenter_every_step']
    )
    diffusion = GaussianDiffusion(diff_cfg).to(device)
    
    if ckpt_path.exists():
        log.info(f"Loading diffusion checkpoint from {ckpt_path}")
        d = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(d['model'] if 'model' in d else d)
    else:
        log.warning(f"Diffusion checkpoint not found at {ckpt_path}!")
        
    return model, diffusion, cfg

def load_pairwise_model(ckpt_path, device):
    # Fixed Pairwise Config
    model = PairwiseEnergyModel(n_atoms=22).to(device)
    
    stats = {}
    if ckpt_path.exists():
        log.info(f"Loading pairwise checkpoint from {ckpt_path}")
        d = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(d['model'])
        stats['e_mean'] = d['e_mean'].to(device)
        stats['e_std'] = d['e_std'].to(device)
        stats['f_std'] = d['f_std'].to(device)
    else:
        log.warning(f"Pairwise checkpoint not found at {ckpt_path}!")
        
    return model, stats

def main():
    parser = argparse.ArgumentParser()
    
    # Diffusion args (Using Loop A / Loop 2 / Loop 3 checkpoint?)
    # User said "Refinement is Loop B".
    # We should use the BEST diffusion model so far.
    # Loop 3 (Scientific Fix) was successful. 'runs/day8_9_al_3/iter_03/checkpoints/best.pt'?
    # Or 'final.pt'. Active Learning loop might not verify/save 'best' in standard way, but 'iter_03' has 'eval_samples.pt'.
    # The models are in 'runs/day8_9_al_3/iter_03/model.pt' usually.
    parser.add_argument("--diff-config", type=str, default="configs/ala2_al_3.yaml")
    parser.add_argument("--diff-ckpt", type=str, default="runs/day8_9_al_3/members/m000/checkpoints/iter_03.pt") 
    
    # Force args
    parser.add_argument("--force-ckpt", type=str, default="runs/energy_pairwise/best_model.pt")
    
    # Sampling args
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--refinement-steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=1e-4) # eta
    parser.add_argument("--temperature", type=float, default=298.0) # Kelvin?
    # Note: Training was on whatever units. If forces are ~1000, energies ~100.
    # We normalized targets.
    # f_pred = (normalized_grad * e_std).
    # Step size 1e-4 depends on units.
    # 1e-4 is defensive.
    
    parser.add_argument("--out-dir", type=str, default="runs/loop_b_refinement")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # 1. Load Diffusion
    # Note: al_3 yaml might have different structure than train_diffusion yaml.
    # But loading relies on 'model' keys which are standard.
    diff_model, diffusion, diff_cfg = load_diffusion_model(Path(args.diff_config), Path(args.diff_ckpt), device)
    diff_model.eval()
    
    # 2. Load Pairwise Force
    force_model, f_stats = load_pairwise_model(Path(args.force_ckpt), device)
    force_model.eval()
    
    # 3. Setup Template for Reconstruction
    # Load first frame of 22-atom data
    templ_path = Path("data/timewarp/train/positions.pt")
    if not templ_path.exists():
        raise FileNotFoundError(f"Template not found at {templ_path}")
    
    templ_all = torch.load(templ_path)[0].to(device) # [22, 3]
    heavy_indices = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
    
    # Atom types for diffusion
    # Generate batch of atom types (10 atoms)
    # We need the 10 atom types for diffusion conditioning.
    # Usually [C, C, O, N, C, C, C, O, N, C]
    # Indices: 0=C, 1=N, 2=O, 3=?, 4=?
    # Let's peek at a shard to preserve exact mapping.
    shard_path = next(Path("data/processed/ala2/shards").glob("*.pt"))
    diff_types = torch.load(shard_path)['atom_types'].to(device) # [10]

    all_samples = []
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    
    log.info(f"Generating {args.n_samples} samples...")
    
    for i in range(n_batches):
        B = min(args.batch_size, args.n_samples - len(all_samples))
        
        # A. Diffusion
        a_batch = diff_types.unsqueeze(0).expand(B, -1)
        shape = (B, 10, 3)
        
        with torch.no_grad():
            x_10 = diffusion.p_sample_loop(diff_model, shape, a_batch)
            
        # B. Reconstruction (10 -> 22)
        x_22 = align_and_reconstruct(x_10, templ_all, heavy_indices)
        
        # C. Refinement
        x_curr = x_22.clone().requires_grad_(True)
        
        # Optimizer typically better than manual update, but manual Langevin is fine
        # x_new = x_old - step * grad + sigma * noise
        # This is Overdamped Langevin.
        # Temp: If model trained on PE, we need consistent units.
        # If we just want minimization, set Temp=0.
        # Let's assume Refinement = Minimization (push to basin bottom) is safer initially.
        # args.temperature default to 0 effectively?
        
        log.info(f"Batch {i+1}: Refining...")
        
        for k in range(args.refinement_steps):
            # Energy & Force
            # Model returns normalized energy
            e_norm = force_model(x_curr)
            # We want Unnormalized Energy gradient?
            # F = -grad(E_unnorm)
            # E_unnorm = E_norm * e_std + e_mean
            # grad(E_unnorm) = grad(E_norm) * e_std
            
            # e_norm.sum() for batch grad
            grad = torch.autograd.grad(e_norm.sum(), x_curr)[0]
            f_pred = -grad * f_stats['e_std']
            
            # Langevin Update
            # x += s * F + noise
            # Careful with step size.
            
            with torch.no_grad():
                # Clean update
                x_curr.data += args.step_size * f_pred
                if args.temperature > 0:
                   # This requires kB in compatible units.
                   # If we don't know units, Temperature is risky.
                   # Let's stick to Gradient Descent (Minimization) for now.
                   pass
                   
        all_samples.append(x_curr.detach().cpu())
        
    final_samples = torch.cat(all_samples, dim=0)
    out_path = Path(args.out_dir) / "refined_samples.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_samples, out_path)
    log.info(f"Saved refined samples to {out_path}")

if __name__ == "__main__":
    main()
