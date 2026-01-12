from __future__ import annotations

import argparse
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import time

from metastategen.utils import get_logger, set_deterministic
from metastategen.data import Ala2Dataset
from metastategen.models.egnn import EGNN, EGNNConfig
from metastategen.models.diffusion import GaussianDiffusion, DiffusionConfig

log = get_logger("train")

def run_training(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    set_deterministic(cfg['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Data
    ds_train = Ala2Dataset(
        cfg['data']['data_dir'], 
        trajs=cfg['data']['train_trajs'], 
        subsample=cfg['data']['frame_subsample']
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['data'].get('num_workers', 0)
    )

    # Model
    # Get n_atom_types from dataset
    n_atom_types = int(ds_train.atom_types.max().item()) + 1
    
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

    diff_cfg = DiffusionConfig(
        T=cfg['diffusion']['T'],
        beta_start=cfg['diffusion']['beta_start'],
        beta_end=cfg['diffusion']['beta_end'],
        schedule=cfg['diffusion']['schedule'],
        recenter_every_step=cfg['diffusion']['recenter_every_step']
    )
    diffusion = GaussianDiffusion(diff_cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['train']['lr']))

    # Setup output
    out_dir = Path(cfg['train']['out_dir'])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = out_dir / "train_log.csv"
    logs = []

    epochs = cfg['train']['epochs']
    
    log.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        ep_losses = []
        
        for batch in dl_train:
            x = batch['x'].to(device)
            a = batch['a'].to(device) # [B, N] or [N] handled by model
            
            # Sample t
            B = x.shape[0]
            t = torch.randint(1, cfg['diffusion']['T'] + 1, (B,), device=device)
            
            loss, info = diffusion.training_loss(model, x, a, t, rot_aug=True)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            opt.step()
            
            ep_losses.append(loss.item())

        avg_loss = sum(ep_losses) / len(ep_losses)
        log.info(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}")
        
        logs.append({"epoch": epoch, "loss": avg_loss})
        
        if epoch % cfg['train']['save_every'] == 0:
            ckpt_path = ckpt_dir / f"ckpt_{epoch:04d}.pt"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'config': cfg
            }, ckpt_path)
            log.info(f"Saved checkpoint: {ckpt_path}")

    pd.DataFrame(logs).to_csv(log_path, index=False)
    
    # Save final
    final_path = ckpt_dir / "final.pt"
    torch.save(model.state_dict(), final_path) # Just weights for easier loading
    log.info("Done.")
    return 0
