import argparse
import yaml
import torch
from pathlib import Path
import numpy as np

from metastategen.utils import get_logger, set_deterministic
from metastategen.models.egnn import EGNN, EGNNConfig
from metastategen.models.diffusion import GaussianDiffusion, DiffusionConfig

log = get_logger("sample")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ala2_day2.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override seed for sampling variability if needed, but keeping deterministic default
    set_deterministic(cfg['train']['seed'] + 100) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Meta (to get atom types and N)
    meta = torch.load(cfg['data']['meta_path'])
    n_atoms = meta['n_atoms']
    
    # Load one shard to get atom_types reference
    import glob
    shard_path = glob.glob(f"{cfg['data']['data_dir']}/*.pt")[0]
    atom_types = torch.load(shard_path)['atom_types'].to(device) # [N]

    # Model Setup
    n_atom_types = int(atom_types.max().item()) + 1
    
    model_cfg = EGNNConfig(
        n_layers=cfg['model']['n_layers'],
        hidden_dim=cfg['model']['hidden_dim']
    )
    
    model = EGNN(
        n_atom_types=n_atom_types,
        hidden_dim=cfg['model']['hidden_dim'],
        n_layers=cfg['model']['n_layers'],
        time_emb_dim=cfg['model']['time_emb_dim'],
        cfg=model_cfg
    ).to(device)

    # Load weights
    log.info(f"Loading weights from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Diffusion Setup
    diff_cfg = DiffusionConfig(
        T=cfg['diffusion']['T'],
        beta_start=cfg['diffusion']['beta_start'],
        beta_end=cfg['diffusion']['beta_end'],
        schedule=cfg['diffusion']['schedule'],
        ddim_eta=cfg['sample']['eta']
    )
    diffusion = GaussianDiffusion(diff_cfg).to(device)

    # Sampling Loop
    n_samples = cfg['sample']['n_samples']
    batch_size = cfg['sample']['batch_size']
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    out_dir = Path(args.outdir) if args.outdir else Path(cfg['train']['out_dir']) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    log.info(f"Generating {n_samples} samples in {n_batches} batches...")
    
    for i in range(n_batches):
        curr_bs = min(batch_size, n_samples - len(all_samples))
        
        # Expand atom types
        batch_types = atom_types.unsqueeze(0).expand(curr_bs, -1)
        
        # CORRECTED: Use tuple for shape, not random noise tensor
        sample_shape = (curr_bs, n_atoms, 3)
        
        # Sample
        if cfg['sample']['eta'] == 0.0 and cfg['sample']['steps'] < cfg['diffusion']['T']:
            # DDIM
            x0 = diffusion.ddim_sample_loop(
                model, 
                sample_shape, 
                batch_types, 
                steps=cfg['sample']['steps'],
                eta=cfg['sample']['eta']
            )
        else:
            # DDPM
            x0 = diffusion.p_sample_loop(
                model, 
                sample_shape, 
                batch_types, 
                steps=None
            )
            
        all_samples.append(x0.cpu())
        log.info(f"Batch {i+1}/{n_batches} done.")

    all_samples = torch.cat(all_samples, dim=0)
    out_path = out_dir / "samples.pt"
    torch.save(all_samples, out_path)
    log.info(f"Saved samples to {out_path}")

if __name__ == "__main__":
    main()
