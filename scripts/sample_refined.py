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
        hidden_dim=cfg['model']['hidden_dim'],
        use_chiral_features=cfg['model'].get('use_chiral_features', False),
        use_rbf=cfg['model'].get('use_rbf', False),
        rbf_dim=cfg['model'].get('rbf_dim', 64),
        rbf_cutoff=cfg['model'].get('rbf_cutoff', 1.0)
    )
    # Hardcoded for 10-atom backbone models (Active Learning Loop 5 used 3 types: C, O, N)
    n_atom_types = 3 
    
    model = EGNN(
        n_atom_types=n_atom_types,
        hidden_dim=cfg['model']['hidden_dim'],
        n_layers=cfg['model']['n_layers'],
        time_emb_dim=cfg['model']['time_emb_dim'],
        cfg=model_cfg
    ).to(device)
    
    scale_factor = float(cfg['data'].get('scale_factor', 1.0))
    
    diff_cfg = DiffusionConfig(
        T=cfg['diffusion']['T'],
        beta_start=cfg['diffusion']['beta_start'],
        beta_end=cfg['diffusion']['beta_end'],
        schedule=cfg['diffusion']['schedule'],
        recenter_every_step=cfg['diffusion']['recenter_every_step'],
        scale_factor=scale_factor
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

def constrain_bonds_22(x):
    """
    Projects backbone bonds (N-CA, CA-C) to target lengths for 22-atom Timewarp structure.
    Indices: N=6, CA=8, C=14.
    """
    t1 = 0.146 # N-CA
    t2 = 0.151 # CA-C
    
    constraints = [
        (6, 8, t1),
        (8, 14, t2)
    ]
    
    # Iterative projection
    for _ in range(5):
        for i1, i2, dist_target in constraints:
            p1 = x[:, i1]
            p2 = x[:, i2]
            diff = p2 - p1
            dist = torch.norm(diff, dim=1, keepdim=True) + 1e-8
            delta = diff * (dist_target / dist - 1.0)
            x[:, i1] -= 0.5 * delta
            x[:, i2] += 0.5 * delta
    return x

def main():
    parser = argparse.ArgumentParser()
    
    # Diffusion args
    # Using the designated best checkpoint from Loop 3 for refinement.
    parser.add_argument("--diff-config", type=str, default="configs/ala2_al_3.yaml")
    parser.add_argument("--diff-ckpt", type=str, default="runs/day8_9_al_3/members/m000/checkpoints/iter_03.pt") 
    
    # Force args
    parser.add_argument("--force-ckpt", type=str, default="runs/energy_pairwise/best_model.pt")
    
    # Sampling args
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--refinement-steps", type=int, default=2000)
    parser.add_argument("--step-size", type=float, default=1e-5) # Reduced for stability
    parser.add_argument("--temperature", type=float, default=298.0) 
    
    # Note: Step size 1e-5 is conservative (based on normalized units).
    
    parser.add_argument("--out-dir", type=str, default="runs/loop_b_refinement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Initial steps before energy filtering")
    parser.add_argument("--keep-percent", type=float, default=1.0, help="Fraction of samples to keep (0.0 < p <= 1.0)")
    
    args = parser.parse_args()
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    
    # Ensure output directory exists
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Diffusion
    # Load model using standard config keys.
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
    all_samples = []
    # Adjust n_batches: If filtering, we process input batch size fully, then reduce.
    # We still loop based on 'args.n_samples' which we treat as INPUT samples count for generation.
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    
    log.info(f"Generating {args.n_samples} samples...")
    
    initial_samples = []
    refined_samples = []
    
    for i in range(n_batches):
        # Determine current batch size (might be smaller for last batch)
        
        B = min(args.batch_size, args.n_samples - (i * args.batch_size))
        if B <= 0: break
        
        # A. Diffusion
        a_batch = diff_types.unsqueeze(0).expand(B, -1)
        shape = (B, 10, 3)
        
        with torch.no_grad():
            x_10 = diffusion.p_sample_loop(diff_model, shape, a_batch)
            x_10 = x_10 / diffusion.cfg.scale_factor
            
        # B. Reconstruction (10 -> 22)
        x_22 = align_and_reconstruct(x_10, templ_all, heavy_indices)
        
        # Store initial *before* any refinement
        initial_samples.append(x_22.clone().cpu()) 
        
        # C. Warmup Phase
        x_curr = x_22.clone().to(device).requires_grad_(True)
        
        warmup_k = args.warmup_steps
        main_k = args.refinement_steps - warmup_k
        
        if warmup_k > 0:
            log.info(f"Batch {i+1}: Warmup ({warmup_k} steps)...")
            for k in range(warmup_k):
                e_norm = force_model(x_curr)
                grad = torch.autograd.grad(e_norm.sum(), x_curr)[0]
                f_pred = -grad * f_stats['e_std']
                
                f_norm = f_pred.norm(dim=-1, keepdim=True)
                clip_coef = torch.clamp(10.0 / (f_norm + 1e-6), max=1.0)
                f_pred = f_pred * clip_coef

                with torch.no_grad():
                    x_curr.data += args.step_size * f_pred
                    x_curr.data = constrain_bonds_22(x_curr.data)

        # D. Filtering
        if args.keep_percent < 1.0:
            with torch.no_grad():
                # Calc energy for filtering
                e_vals = force_model(x_curr) * f_stats['e_std'] + f_stats['e_mean'] # Denormalized for logging? No, model returns norm.
                # Just use e_norm for sorting
                e_norm = force_model(x_curr)
                
                k_keep = int(B * args.keep_percent)
                k_keep = max(1, k_keep) # Keep at least 1
                
                # Sort
                vals, indices = torch.sort(e_norm)
                keep_idx = indices[:k_keep]
                
                log.info(f"Batch {i+1}: Filtering top {args.keep_percent*100}% ({B} -> {k_keep}). Best E={vals[0].item():.2f}, Worst kept={vals[k_keep-1].item():.2f}")
                
                # Filter x_curr
                x_curr = x_curr[keep_idx].detach().clone().requires_grad_(True)
                
        # E. Main Refinement Phase
        log.info(f"Batch {i+1}: Main Refinement ({main_k} steps)...")
        for k in range(main_k):
            e_norm = force_model(x_curr)
            grad = torch.autograd.grad(e_norm.sum(), x_curr)[0]
            f_pred = -grad * f_stats['e_std']
            
            f_norm = f_pred.norm(dim=-1, keepdim=True)
            clip_coef = torch.clamp(10.0 / (f_norm + 1e-6), max=1.0)
            f_pred = f_pred * clip_coef

            with torch.no_grad():
                if i == 0 and (k < 5 or k % 5000 == 0 or k == main_k - 1):
                     log.info(f"Main Step {k}: E_norm={e_norm.mean().item():.2f} | Force_norm_pre={f_norm.mean().item():.2f}")
                
                x_curr.data += args.step_size * f_pred
                x_curr.data = constrain_bonds_22(x_curr.data)
                    
        refined_samples.append(x_curr.detach().cpu())
        
        # Checkpoint every batch (since batches are large)
        if (i+1) % 1 == 0:
            ckpt_data = {
                "initial_positions": torch.cat(initial_samples, dim=0),
                "refined_positions": torch.cat(refined_samples, dim=0),
                "atom_types": diff_types.cpu()
            }
            ckpt_path = Path(args.out_dir) / f"checkpoint_batch_{i+1:03d}.pt"
            torch.save(ckpt_data, ckpt_path)
            log.info(f"Saved checkpoint to {ckpt_path}")
        
    results = {
        "initial_positions": torch.cat(initial_samples, dim=0),
        "refined_positions": torch.cat(refined_samples, dim=0),
        "atom_types": diff_types.cpu()
    }
    
    out_path = Path(args.out_dir) / "refined_results.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_path)
    log.info(f"Saved refined results to {out_path}")

if __name__ == "__main__":
    main()
