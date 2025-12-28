import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import pandas as pd

from metastategen.utils import get_logger, set_deterministic
from metastategen.models.energy import EnergyEGNN
from metastategen.models.egnn import EGNNConfig

log = get_logger("train_energy")

class EnergyDataset(Dataset):
    def __init__(self, shard_dir: str, forces_path: str, energies_path: str, trajs: list = None, subsample: int = 1):
        super().__init__()
        self.shard_paths = sorted(glob.glob(f"{shard_dir}/*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {shard_dir}")
        
        if not Path(forces_path).exists():
            raise FileNotFoundError(f"Forces file not found at {forces_path}")
        if not Path(energies_path).exists():
            raise FileNotFoundError(f"Energies file not found at {energies_path}")
            
        log.info(f"Loading forces from {forces_path}...")
        self.all_forces = torch.load(forces_path)
        log.info(f"Loading energies from {energies_path}...")
        self.all_energies = torch.load(energies_path) 
        
        self.data_pos = []
        self.data_force = []
        self.data_energy = []
        self.data_traj = []
        
        current_idx = 0
        
        for p in self.shard_paths:
            d = torch.load(p)
            pos = d['positions'] # [B, N, 3]
            n_frames = pos.shape[0]
            
            # Check bounds
            if current_idx + n_frames > len(self.all_forces):
                raise ValueError(f"Forces/Energies file mismatch: {len(self.all_forces)} < {current_idx + n_frames}")
                
            frc = self.all_forces[current_idx : current_idx + n_frames]
            ene = self.all_energies[current_idx : current_idx + n_frames]
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
            self.data_energy.append(ene[indices])
            self.data_traj.append(traj_ids[indices])
            
            if not hasattr(self, 'atom_types'):
                self.atom_types = d['atom_types']
        
        if len(self.data_pos) > 0:
            self.positions = torch.cat(self.data_pos, dim=0)
            self.forces = torch.cat(self.data_force, dim=0)
            self.energies = torch.cat(self.data_energy, dim=0)
            self.traj_ids = torch.cat(self.data_traj, dim=0)
        else:
            self.positions = torch.empty(0)
            self.forces = torch.empty(0)
            self.energies = torch.empty(0)
            
        log.info(f"Loaded {len(self.positions)} frames. Trajs={trajs}")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return {
            "x": self.positions[idx],
            "f": self.forces[idx],
            "e": self.energies[idx],
            "a": self.atom_types
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ala2_energy.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    set_deterministic(0) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    
    # 1. Dataset
    train_trajs = cfg['data'].get('train_trajs', [0])
    val_trajs = cfg['data'].get('val_trajs', [1])
    
    ds_train = EnergyDataset(
        shard_dir=cfg['data']['processed_dir'],
        forces_path=cfg['data']['forces_path'],
        energies_path=cfg['data']['energies_path'],
        trajs=train_trajs,
        subsample=cfg['data'].get('subsample', 1)
    )
    
    ds_val = EnergyDataset(
        shard_dir=cfg['data']['processed_dir'],
        forces_path=cfg['data']['forces_path'],
        energies_path=cfg['data']['energies_path'],
        trajs=val_trajs,
        subsample=cfg['data'].get('subsample', 1)
    )
    
    dl_train = DataLoader(ds_train, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    dl_val = DataLoader(ds_val, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])
    
    # Normalization
    log.info("Computing normalization stats (train set)...")
    
    # Force:
    f_mean = torch.zeros(1, device=device) # Assume 0 mean for forces
    f_std = ds_train.forces.std().to(device)
    
    # Energy:
    e_train = ds_train.energies
    e_mean = e_train.mean().to(device)
    e_std = e_train.std().to(device)
    
    log.info(f"Force: Mean={f_mean.item():.4f}, Std={f_std.item():.4f}")
    log.info(f"Energy: Mean={e_mean.item():.4f}, Std={e_std.item():.4f}")
    
    # 2. Model
    n_atom_types = cfg['model']['n_atom_types']
    model_cfg = EGNNConfig(
        n_layers=cfg['model']['n_layers'],
        hidden_dim=cfg['model']['hidden_dim'],
        dropout=cfg['model']['dropout'],
        use_rbf=cfg['model'].get('use_rbf', True),
        rbf_dim=cfg['model'].get('rbf_dim', 64),
        rbf_cutoff=cfg['model'].get('rbf_cutoff', 1.0)
    )
    model = EnergyEGNN(
        n_atom_types=n_atom_types,
        hidden_dim=cfg['model']['hidden_dim'],
        n_layers=cfg['model']['n_layers'],
        cfg=model_cfg
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=cfg['training']['patience'] // 2, verbose=True)
    
    criterion = nn.MSELoss()
    rho_e = cfg['training'].get('energy_weight', 1.0)
    rho_f = cfg['training'].get('force_weight', 1.0)
    
    # 3. Training Loop
    out_dir = Path("runs") / cfg['experiment_name']
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    logs = []
    
    epochs = cfg['training']['max_epochs']
    
    log.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        losses_f = []
        losses_e = []
        
        for batch in dl_train:
            x = batch['x'].to(device) # [B, N, 3]
            f = batch['f'].to(device) # [B, N, 3]
            e = batch['e'].to(device) # [B]
            a = batch['a'].to(device)
            
            # Forward
            # Enable grad on x is handled inside model, but we need to ensure torch retains graph for x
            x.requires_grad_(True)
            
            E_pred, _ = model(x, a, create_graph=True)
            
            grad_outputs = torch.ones_like(E_pred)
            gradients = torch.autograd.grad(
                outputs=E_pred,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # F_pred in normalized units?
            # F_phys = -grad(E_phys, x)
            # E_phys = E_pred * e_std + e_mean
            # F_phys = -grad(E_pred, x) * e_std
            # F_target_norm = (F_phys - f_mean) / f_std
            # So F_pred_norm should be:
            # (-grad(E_pred, x) * e_std - f_mean) / f_std
            # Assuming f_mean=0:
            # F_pred_norm = -grad(E_pred, x) * (e_std / f_std)
            
            F_pred = -gradients * (e_std / f_std)

            # Normalize Targets
            f_target = (f - f_mean) / f_std
            e_target = (e - e_mean) / e_std 
            
            # Loss
            loss_f = criterion(F_pred, f_target)
            loss_e = criterion(E_pred, e_target)
            
            loss = rho_f * loss_f + rho_e * loss_e
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            opt.step()
            
            losses.append(loss.item())
            losses_f.append(loss_f.item())
            losses_e.append(loss_e.item())
            
        train_loss = sum(losses) / len(losses)
        train_mse_f = sum(losses_f) / len(losses_f)
        train_mse_e = sum(losses_e) / len(losses_e)
        
        # Validation
        model.eval()
        val_losses = []
        val_losses_f = []
        val_losses_e = []
        
        # For val, we still need create_graph=True to perform autograd for F_pred?
        # Yes, F_pred is computed via autograd.grad. 
        # Even in eval mode, autograd works if inputs require grad.
        # But we don't need the backward pass for weights.
        
        for batch in dl_val:
            x = batch['x'].to(device)
            f = batch['f'].to(device)
            e = batch['e'].to(device)
            a = batch['a'].to(device)
            
            x.requires_grad_(True)
            
            E_pred, _ = model(x, a, create_graph=True) 
            
            grad_outputs = torch.ones_like(E_pred)
            gradients = torch.autograd.grad(
                outputs=E_pred,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=False, 
                retain_graph=True,
                only_inputs=True
            )[0]
            F_pred = -gradients * (e_std / f_std)
            
            f_target = (f - f_mean) / f_std
            e_target = (e - e_mean) / e_std
            
            loss_f = criterion(F_pred, f_target)
            loss_e = criterion(E_pred, e_target)
            loss = rho_f * loss_f + rho_e * loss_e
            
            val_losses.append(loss.item())
            val_losses_f.append(loss_f.item())
            val_losses_e.append(loss_e.item())
            
        val_loss = sum(val_losses) / len(val_losses)
        val_mse_f = sum(val_losses_f) / len(val_losses_f)
        val_mse_e = sum(val_losses_e) / len(val_losses_e)
        
        scheduler.step(val_mse_f) # Schedule on Force MSE primarily? Or Total Loss? 
        # Usually Force MSE is the goal.
        
        log.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} (F={train_mse_f:.4f}, E={train_mse_e:.4f}) | Val F-MSE: {val_mse_f:.4f} | Val E-MSE: {val_mse_e:.4f}")
        
        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f": train_mse_f,
            "train_e": train_mse_e,
            "val_loss": val_loss,
            "val_f": val_mse_f,
            "val_e": val_mse_e
        })
        
        if epoch % cfg['training']['save_every'] == 0:
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "f_mean": f_mean,
                "f_std": f_std,
                "e_mean": e_mean,
                "e_std": e_std
            }
            torch.save(state, ckpt_dir / f"ckpt_{epoch:03d}.pt")
            
    pd.DataFrame(logs).to_csv(log_path, index=False)
    log.info("Done.")

if __name__ == "__main__":
    main()
