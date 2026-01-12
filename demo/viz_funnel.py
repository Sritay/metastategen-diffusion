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
try:
    from regions import plot_regions
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from regions import plot_regions

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

def get_atom_info(pdb_path: Path):
    """Parses PDB to extract atom names and residues for writing."""
    atom_info = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_info.append(line.strip())
    return atom_info

def save_to_pdb(samples: torch.Tensor, atom_template: list, out_path: Path):
    """Writes samples to a multi-model PDB file."""
    with open(out_path, 'w') as f:
        for i, sample in enumerate(samples):
            f.write(f"MODEL     {i+1}\n")
            for j, line in enumerate(atom_template):
                # Replace coordinates in the fixed-width format
                # val: 30-38 (x), 38-46 (y), 46-54 (z)
                x = sample[j, 0].item()
                y = sample[j, 1].item()
                z = sample[j, 2].item()
                # PDB format requires specific spacing; simplest is to construct line carefully or overwrite
                # Line structure:
                # 0-30: Identity info (unchanged)
                # 30-54: Coords
                # 54+: Remainder
                
                # Careful with line length handling
                # Standard PDB: 
                # ATOM      1  N   ALA A   1      -0.525   1.363   0.000  1.00  0.00           N
                
                # We can just format a new line if we parsed components, but replacing substring is safer for preserving other fields
                # assuming standard columns.
                # However, Python string slicing is easy.
                
                prefix = line[:30]
                suffix = line[54:]
                coords = f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f.write(f"{prefix}{coords}{suffix}\n")
            f.write("ENDMDL\n")
    log.info(f"Saved PDB to {out_path}")

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
    parser.add_argument("--data-path", type=str, help="Path to refined_results.pt")
    parser.add_argument("--pdb-path", type=str, default="/Users/sritaymistry/projects/metastategen-diffusion/data/raw/alanine-dipeptide-nowater.pdb")
    
    args = parser.parse_args()
    set_deterministic(42)
    # Force CPU for local demo if no GPU found or for stability
    device = torch.device("cpu")
    
    # Check paths
    if not Path(args.diff_ckpt).exists():
        # Fallback to iter_03
        args.diff_ckpt = "runs/day8_9_al_3/members/m000/checkpoints/iter_03.pt"
    
    # Models are not needed for visualization of pre-computed results
    # log.info("Loading models...")
    # diff_model, diffusion, _ = load_diffusion_model(Path(args.diff_config), Path(args.diff_ckpt), device)
    # diff_model.eval()
    # force_model, f_stats = load_pairwise_model(Path(args.force_ckpt), device)
    # force_model.eval()
    # templ_path = Path("data/timewarp/train/positions.pt")
    # templ_all = torch.load(templ_path)[0].to(device)
    # heavy_indices = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
    # shard_path = next(Path("data/processed/ala2/shards").glob("*.pt"))
    # diff_types = torch.load(shard_path)['atom_types'].to(device)

    # Load pre-computed results
    if args.data_path and Path(args.data_path).exists():
        log.info(f"Loading results from {args.data_path}")
        results = torch.load(args.data_path, map_location=device)
        pre_samples = results['initial_positions']
        post_samples = results['refined_positions']
        args.n_samples = pre_samples.shape[0]
    else: 
        log.error("Data path not provided or not found. Please provide --data-path to refined_results.pt")
        return
    
    # Compute Phi/Psi
    log.info("Computing Dihedrals...")
    pre_phi_psi = compute_phi_psi(pre_samples, Path(args.pdb_path))
    post_phi_psi = compute_phi_psi(post_samples, Path(args.pdb_path))

    # Save PDBs (First 100 samples)
    atom_info = get_atom_info(Path(args.pdb_path))
    n_save = min(100, args.n_samples)
    save_to_pdb(pre_samples[:n_save], atom_info, Path(args.out_dir) / "initial_ensemble.pdb")
    save_to_pdb(post_samples[:n_save], atom_info, Path(args.out_dir) / "refined_ensemble.pdb")
    
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
    
    # Overlay Ground Truth Regions
    ax = plt.gca()
    try:
        plot_regions(ax)
    except Exception as e:
        log.warning(f"Could not plot regions: {e}")

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
