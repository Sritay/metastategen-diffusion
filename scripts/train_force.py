import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import pandas as pd

from metastategen.utils import get_logger, set_deterministic
from metastategen.models.force import ForceEGNN
from metastategen.models.egnn import EGNNConfig

log = get_logger("train_force")

class ForceDataset(Dataset):
    def __init__(self, shard_dir: str, forces_path: str, trajs: list = None, subsample: int = 1):
        super().__init__()
        self.shard_paths = sorted(glob.glob(f"{shard_dir}/*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {shard_dir}")
        
        if not Path(forces_path).exists():
            raise FileNotFoundError(f"Forces file not found at {forces_path}")
            
        # Load all forces [M, N, 3]
        log.info(f"Loading forces from {forces_path}...")
        self.all_forces = torch.load(forces_path)
        
        self.data_pos = []
        self.data_force = []
        self.data_traj = []
        
        current_idx = 0
        total_frames = 0
        
        for p in self.shard_paths:
            d = torch.load(p)
            pos = d['positions'] # [B, N, 3]
            n_frames = pos.shape[0]
            
            # Get corresponding forces
            # check bounds
            if current_idx + n_frames > len(self.all_forces):
                raise ValueError(f"Forces file is smaller than sum of shards! {len(self.all_forces)} < {current_idx + n_frames}")
                
            frc = self.all_forces[current_idx : current_idx + n_frames]
            current_idx += n_frames
            
            traj_ids = d['traj_id']
            
            # Filter
            mask = torch.zeros_like(traj_ids, dtype=torch.bool)
            if trajs is None:
                mask[:] = True
            else:
                for t in trajs:
                    mask |= (traj_ids == t)
            
            if not mask.any():
                continue
                
            # Subsample
            indices = torch.where(mask)[0]
            if subsample > 1:
                indices = indices[::subsample]
            
            self.data_pos.append(pos[indices])
            self.data_force.append(frc[indices])
            self.data_traj.append(traj_ids[indices])
            
            # Atom types (assume constant)
            if not hasattr(self, 'atom_types'):
                self.atom_types = d['atom_types']
        
        if len(self.data_pos) > 0:
            self.positions = torch.cat(self.data_pos, dim=0)
            self.forces = torch.cat(self.data_force, dim=0)
            self.traj_ids = torch.cat(self.data_traj, dim=0)
        else:
            self.positions = torch.empty(0)
            self.forces = torch.empty(0)
            self.traj_ids = torch.empty(0)
            
        log.info(f"Loaded {len(self.positions)} frames. Trajs={trajs}")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return {
            "x": self.positions[idx],
            "f": self.forces[idx],
            "a": self.atom_types
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ala2_force.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    set_deterministic(0) # or config seed if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    
    # 1. Dataset
    # Timewarp data has 2 trajectories: 0 (train/ad1) and 1 (test/ad2)
    # Use 0 for train, 1 for val.
    train_trajs = cfg['data'].get('train_trajs', [0])
    val_trajs = cfg['data'].get('val_trajs', [1])
    
    ds_train = ForceDataset(
        shard_dir=cfg['data']['processed_dir'],
        forces_path=cfg['data']['forces_path'],
        trajs=train_trajs,
        subsample=cfg['data'].get('subsample', 1)
    )
    
    ds_val = ForceDataset(
        shard_dir=cfg['data']['processed_dir'],
        forces_path=cfg['data']['forces_path'],
        trajs=val_trajs,
        subsample=cfg['data'].get('subsample', 1)
    )
    
    dl_train = DataLoader(ds_train, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    dl_val = DataLoader(ds_val, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])
    
    # 2. Model
    n_atom_types = cfg['model']['n_atom_types']
    model_cfg = EGNNConfig(
        n_layers=cfg['model']['n_layers'],
        hidden_dim=cfg['model']['hidden_dim'],
        dropout=cfg['model']['dropout']
    )
    model = ForceEGNN(
        n_atom_types=n_atom_types,
        hidden_dim=cfg['model']['hidden_dim'],
        n_layers=cfg['model']['n_layers'],
        cfg=model_cfg
    ).to(device)
    
    # Compute Normalization Stats from Training Set
    log.info("Computing force normalization statistics (train set)...")
    all_train_f = ds_train.forces
    f_mean = all_train_f.mean(dim=(0, 1)) # [3] or [1, 1, 3]? Usually per-component mean or scalar?
    # Forces are vectors. Normalizing per component [x, y, z] is standard.
    # But usually we want isotropic normalization for rot invariance? 
    # Actually, standardizing each component (0 mean, 1 std) is common.
    # Let's simple: mean=0 (usually true for forces in equilibrium?), std = root mean square magnitude / sqrt(3)?
    # Or just per-element mean/std.
    f_mean = all_train_f.mean() # Scalar mean? Forces have sign. Mean should be near 0.
    f_std = all_train_f.std()
    
    # Let's stick to scalar normalization to preserve vector directionality cleanly?
    # If we shift by mean vector, we break equivariance if mean is not 0 vector (it should be 0 vector for isotropic system).
    # Since Ala2 involves rotation, mean force vector should be 0.
    # So we only scale by std.
    
    f_mean = torch.zeros(1, device=device) # Assume 0 mean
    f_std = all_train_f.std().to(device)
    
    log.info(f"Force Normalization: Mean={f_mean.item():.6f}, Std={f_std.item():.6f}")

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=cfg['training']['patience'] // 2, verbose=True)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    # Setup output
    out_dir = Path("runs") / cfg['experiment_name']
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = out_dir / "train_log.csv"
    logs = []
    
    epochs = cfg['training']['max_epochs']
    
    log.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for batch in dl_train:
            x = batch['x'].to(device)
            f = batch['f'].to(device)
            a = batch['a'].to(device)
            
            # Normalize Target
            f_target = (f - f_mean) / f_std
            
            f_pred = model(x, a)
            loss = criterion(f_pred, f_target)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            opt.step()
            
            train_losses.append(loss.item())
            
        train_mse = sum(train_losses) / len(train_losses)
        
        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in dl_val:
                x = batch['x'].to(device)
                f = batch['f'].to(device)
                a = batch['a'].to(device)
                
                f_target = (f - f_mean) / f_std
                
                f_pred = model(x, a)
                loss = criterion(f_pred, f_target)
                val_losses.append(loss.item())
                
        val_mse = sum(val_losses) / len(val_losses)
        
        # Scheduler step
        scheduler.step(val_mse)
        
        log.info(f"Epoch {epoch}/{epochs} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        
        logs.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        
        # Save checkpoint
        # Save stats too
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'config': cfg,
            'f_mean': f_mean,
            'f_std': f_std
        }
        torch.save(state, ckpt_dir / "latest.pt")
        if epoch % 10 == 0:
             torch.save(state, ckpt_dir / f"ckpt_{epoch:03d}.pt")
             
    pd.DataFrame(logs).to_csv(log_path, index=False)
    log.info("Done.")

if __name__ == "__main__":
    main()
