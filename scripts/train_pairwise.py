import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import pandas as pd
import numpy as np

from metastategen.utils import get_logger, set_deterministic

log = get_logger("train_pairwise")

# --- Dataset (Reused Logic) ---
class EnergyDataset(Dataset):
    def __init__(self, shard_dir: str, forces_path: str, energies_path: str, trajs: list = None, subsample: int = 1):
        super().__init__()
        # We only need forces/energies and POSITIONS (which must be 22-atom all-atom from TimeWarp?)
        # WAIT: The "shards" in data/processed/ala2/shards are 10-atom.
        # BUT train_energy.py was loading from there? No, train_energy.py was failing/confused.
        # Actually: TimeWarp data IS the ground truth.
        # We should load positions DIRECTLY from the TimeWarp arrays parallel to forces/energies?
        # NO, forces/energies files are [N_total, ...].
        # We need the POSITIONS that match them.
        # Let's assume the user has 22-atom positions somewhere.
        # Ah, 'alanine-dipeptide-3x250ns-heavy-atom-positions.npz' is 10-atom.
        # TimeWarp dataset has 'test' and 'train' folders with full state.
        
        # HACK: For this specific script, let's load the raw 22-atom trajectories if possible?
        # OR: Does the processed directory contain 22-atom data?
        # "al_pool_ref.pt" is 10-atom.
        # "al_forces_ref.pt" is 10-atom forces? No, the user said TimeWarp data is 22-atom.
        
        # Checking previous context: User said "timewarp data is all 22 atom".
        # We must align.
        
        # Let's assume we load everything from 'data/timewarp/train' or similar if available?
        # Or simpler: The user presumably provided matched files.
        # Let's trust proper files are passed in config.
        
        self.shard_paths = sorted(glob.glob(f"{shard_dir}/*.pt"))
        if not self.shard_paths:
            # Fallback for when shard_dir is actually a direct file path or pattern
            pass

        # Loading massive arrays to memory directly (Fastest for < 2GB data)
        log.info(f"Loading forces: {forces_path}")
        self.forces = torch.load(forces_path) # [N, 22, 3] or [N, 66]
        log.info(f"Loading energies: {energies_path}")
        self.energies = torch.load(energies_path) # [N]
        
        # We need positions. Assuming they are saved next to forces?
        # Let's try to infer positions path.
        pos_path = forces_path.replace("forces", "positions")
        if not Path(pos_path).exists():
             # Try replacing "forces" with "coords" or look at config
             raise FileNotFoundError(f"Cannot infer positions path from {forces_path}. Expected {pos_path}")
        
        log.info(f"Loading positions: {pos_path}")
        self.positions = torch.load(pos_path) # [N, 22, 3]

        # Basic validation
        assert self.positions.shape[0] == self.forces.shape[0]
        assert self.positions.shape[0] == self.energies.shape[0]
        
        # Splits
        # Simple split: first 80% train, last 20% val (Time-based split)
        # Or trajectory based.
        # Let's accept indices argument.
        
    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return {
            "x": self.positions[idx],
            "f": self.forces[idx],
            "e": self.energies[idx]
        }

from metastategen.models.pairwise import PairwiseEnergyModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forces", type=str, required=True, help="Path to .pt file with forces")
    parser.add_argument("--energies", type=str, required=True, help="Path to .pt file with energies")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="runs/energy_pairwise")
    args = parser.parse_args()
    
    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Target Device: {device}")
    
    # Init Data (Load all)
    # We create a simple ephemeral dataset since we don't have the config boilerplate
    full_ds = EnergyDataset(None, args.forces, args.energies)
    
    # Split 80/20
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    ds_train, ds_val = torch.utils.data.random_split(full_ds, [n_train, n_val])
    
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Stats
    # Assuming ds_train is a Subset, access underlying dataset
    all_energies = full_ds.energies
    all_forces = full_ds.forces
    
    e_mean = all_energies.mean().to(device)
    e_std = all_energies.std().to(device)
    # Force std?
    f_std = all_forces.std().to(device)
    
    log.info(f"Stats - E_mean: {e_mean:.2f}, E_std: {e_std:.2f}, F_std: {f_std:.2f}")
    
    model = PairwiseEnergyModel(n_atoms=22).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sq_err = 0
        train_f_sq_err = 0
        steps = 0
        
        for batch in dl_train:
            x = batch['x'].to(device).requires_grad_(True)
            f_tgt = batch['f'].to(device)
            e_tgt = batch['e'].to(device)
            
            # Forward
            e_pred_norm = model(x)
            
            # Energies: unnormalize
            e_pred = e_pred_norm * e_std + e_mean
            
            # Forces: -grad(E, x)
            # grad(E_norm, x) * e_std
            grad = torch.autograd.grad(e_pred_norm.sum(), x, create_graph=True)[0]
            f_pred = -grad * e_std
            
            loss_e = ((e_pred - e_tgt)**2).mean()
            loss_f = ((f_pred - f_tgt)**2).mean() # MSE on forces
            
            # Combined Loss (Weight forces heavily as they matter for dynamics)
            loss = loss_e + 100.0 * loss_f
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_sq_err += loss_e.item()
            train_f_sq_err += loss_f.item()
            steps += 1
            
        # Val
        model.eval()
        val_sq_err = 0
        val_f_sq_err = 0
        val_steps = 0
        
        for batch in dl_val:
            x = batch['x'].to(device).requires_grad_(True)
            f_tgt = batch['f'].to(device)
            e_tgt = batch['e'].to(device)
            
            e_pred_norm = model(x)
            e_pred = e_pred_norm * e_std + e_mean
            grad = torch.autograd.grad(e_pred_norm.sum(), x, create_graph=False)[0]
            f_pred = -grad * e_std
            
            loss_e = ((e_pred - e_tgt)**2).mean()
            loss_f = ((f_pred - f_tgt)**2).mean()
            
            val_sq_err += loss_e.item()
            val_f_sq_err += loss_f.item()
            val_steps += 1
            
        rmse_e = np.sqrt(val_sq_err / val_steps)
        rmse_f = np.sqrt(val_f_sq_err / val_steps)
        
        log.info(f"Ep {epoch}: Val RMSE_E={rmse_e:.4f} Val RMSE_F={rmse_f:.4f}")
        
        if rmse_f < best_val_loss:
            best_val_loss = rmse_f
            torch.save({
                "model": model.state_dict(),
                "e_mean": e_mean,
                "e_std": e_std,
                "f_std": f_std
            }, out_path / "best_model.pt")

    log.info("Training complete.")

if __name__ == "__main__":
    main()
