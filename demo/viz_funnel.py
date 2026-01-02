import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from metastategen.utils import get_logger, set_deterministic
from metastategen.models.egnn import EGNN, EGNNConfig
from metastategen.models.diffusion import GaussianDiffusion, DiffusionConfig
from metastategen.models.pairwise import PairwiseEnergyModel
from metastategen.reconstruct import align_and_reconstruct
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

log = get_logger("viz_funnel")

def load_diffusion_model(config_path, ckpt_path, device):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    model_cfg = EGNNConfig(
        n_layers=cfg['model']['n_layers'],
        hidden_dim=cfg['model']['hidden_dim']
    )
    n_atom_types = 3 
    
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
        d = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(d['model'] if 'model' in d else d)
    else:
        log.warning(f"Diffusion checkpoint not found at {ckpt_path}!")
        
    return model, diffusion, cfg

def load_pairwise_model(ckpt_path, device):
    model = PairwiseEnergyModel(n_atoms=22).to(device)
    stats = {}
    if ckpt_path.exists():
        d = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(d['model'])
        stats['e_mean'] = d['e_mean'].to(device)
        stats['e_std'] = d['e_std'].to(device)
        stats['f_std'] = d['f_std'].to(device)
    return model, stats

def compute_phi_psi(samples: torch.Tensor, pdb_path: Path):
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long, device=samples.device)
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    # Using Loop 3 Final or Best checkpoint
    parser.add_argument("--diff-config", type=str, default="configs/ala2_al_3.yaml")
    # Ensuring we find a valid checkpoint
    parser.add_argument("--diff-ckpt", type=str, default="runs/day8_9_al_3/members/m000/checkpoints/final.pt") 
    parser.add_argument("--force-ckpt", type=str, default="runs/energy_pairwise/best_model.pt")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=1e-7) # Careful step size
    parser.add_argument("--refinement-steps", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="demo")
    parser.add_argument("--pdb-path", type=str, default="/Users/sritaymistry/projects/metastategen-diffusion/data/raw/alanine-dipeptide-nowater.pdb")
    
    args = parser.parse_args()
    set_deterministic(42)
    # Force CPU for local demo if no GPU found or for stability
    device = torch.device("cpu")
    
    # Check paths
    if not Path(args.diff_ckpt).exists():
        # Fallback to iter_03
        args.diff_ckpt = "runs/day8_9_al_3/members/m000/checkpoints/iter_03.pt"
    
    log.info("Loading models...")
    diff_model, diffusion, _ = load_diffusion_model(Path(args.diff_config), Path(args.diff_ckpt), device)
    diff_model.eval()
    
    force_model, f_stats = load_pairwise_model(Path(args.force_ckpt), device)
    force_model.eval()
    
    templ_path = Path("data/timewarp/train/positions.pt")
    templ_all = torch.load(templ_path)[0].to(device)
    heavy_indices = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
    
    # Get atom types
    shard_path = next(Path("data/processed/ala2/shards").glob("*.pt"))
    diff_types = torch.load(shard_path)['atom_types'].to(device)

    pre_samples = []
    post_samples = []
    
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    log.info(f"Generating {args.n_samples} samples...")
    
    for i in range(n_batches):
        B = min(args.batch_size, args.n_samples - (i * args.batch_size))
        
        # Generator
        a_batch = diff_types.unsqueeze(0).expand(B, -1)
        shape = (B, 10, 3)
        with torch.no_grad():
            x_10 = diffusion.p_sample_loop(diff_model, shape, a_batch)
        
        # Reconstruct
        x_22 = align_and_reconstruct(x_10, templ_all, heavy_indices)
        pre_samples.append(x_22.cpu())
        
        # Refine (Minimization)
        x_curr = x_22.clone().to(device).requires_grad_(True)
        for _ in range(args.refinement_steps):
            e_norm = force_model(x_curr)
            grad = torch.autograd.grad(e_norm.sum(), x_curr)[0]
            f_pred = -grad * f_stats['e_std']
            with torch.no_grad():
                x_curr.data += args.step_size * f_pred
        
        post_samples.append(x_curr.detach().cpu())
        log.info(f"Batch {i+1} done")
        
    pre_samples = torch.cat(pre_samples, dim=0)
    post_samples = torch.cat(post_samples, dim=0)
    
    # Compute Phi/Psi
    log.info("Computing Dihedrals...")
    pre_phi_psi = compute_phi_psi(pre_samples, Path(args.pdb_path))
    post_phi_psi = compute_phi_psi(post_samples, Path(args.pdb_path))
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel("Phi")
    plt.ylabel("Psi")
    plt.title("Refinement Funnel: Generator (Blue) -> Refined (Red)")
    
    # Pre: Blue, Transparent
    plt.scatter(pre_phi_psi[:, 0], pre_phi_psi[:, 1], c='blue', alpha=0.3, label='Pre-Refinement', s=10)
    
    # Post: Red, Opaque
    plt.scatter(post_phi_psi[:, 0], post_phi_psi[:, 1], c='red', alpha=1.0, label='Post-Refinement', s=10, edgecolors='black', linewidth=0.5)
    
    # Draw arrows for a subset
    for j in range(min(50, args.n_samples)):
        plt.arrow(pre_phi_psi[j, 0], pre_phi_psi[j, 1], 
                  post_phi_psi[j, 0] - pre_phi_psi[j, 0], 
                  post_phi_psi[j, 1] - pre_phi_psi[j, 1], 
                  color='gray', alpha=0.2, width=0.5)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_file = Path(args.out_dir) / "funnel_plot.png"
    plt.savefig(out_file, dpi=150)
    log.info(f"Saved plot to {out_file}")

    # Save to CSV
    df_funnel = pd.DataFrame({
        'Pre_Phi': pre_phi_psi[:, 0],
        'Pre_Psi': pre_phi_psi[:, 1],
        'Post_Phi': post_phi_psi[:, 0],
        'Post_Psi': post_phi_psi[:, 1]
    })
    csv_file = Path(args.out_dir) / "funnel_data.csv"
    df_funnel.to_csv(csv_file, index=False)
    log.info(f"Saved data to {csv_file}")

if __name__ == "__main__":
    main()
