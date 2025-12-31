import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from metastategen.models.energy import EnergyEGNN
from metastategen.models.egnn import EGNNConfig
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("debug_suite")

def check_data_stats(pos, forces, energies):
    log.info("--- Check 1: Statistical Analysis ---")
    
    # 1. Magnitudes
    f_mags = forces.norm(dim=-1)
    
    log.info(f"Forces: Min={f_mags.min():.4f}, Max={f_mags.max():.4f}, Mean={f_mags.mean():.4f}, Std={f_mags.std():.4f}")
    log.info(f"Energies: Min={energies.min():.4f}, Max={energies.max():.4f}, Mean={energies.mean():.4f}, Std={energies.std():.4f}")
    
    # 2. Outliers
    q99 = torch.quantile(f_mags, 0.99)
    log.info(f"Force 99th Percentile: {q99:.4f}")
    outliers = (f_mags > 3000).sum()
    log.info(f"frames with Force > 3000: {outliers} / {len(forces)} ({outliers/len(forces)*100:.2f}%)")
    
    # 3. NaNs
    if torch.isnan(forces).any() or torch.isnan(energies).any():
        log.error("CRITICAL: NaNs found in dataset!")
    else:
        log.info("No NaNs found.")

def check_physics_consistency(pos, forces, energies):
    log.info("\n--- Check 2: Physics Consistency (F = - grad E) ---")
    log.info("Testing dE ~ -F * dx on adjacent frames...")
    
    # Compute delta x and delta E between adjacent frames
    # We assume the passed tensors are a sequential chunk from a shard
    
    dx = pos[1:] - pos[:-1] # [N-1, A, 3]
    de = energies[1:] - energies[:-1] # [N-1]
    
    # Approximating Work done: W = F_avg * dx
    # F_avg = (F_t + F_{t+1}) / 2
    f_avg = (forces[1:] + forces[:-1]) / 2 # [N-1, A, 3]
    
    # W = sum(F_avg * dx) over atoms
    w = (f_avg * dx).sum(dim=(1, 2)) # [N-1]
    
    # Ideally: dE = -W  => dE + W approx 0
    # Let's check correlation between dE and -W
    
    tensor_dot = -w
    
    # Filter for small steps only (to avoid PBC jumps or large gaps)
    # Ala2 box is usually small, but let's check dx magnitude
    dx_norm = dx.norm(dim=-1).max(dim=-1)[0] # Max atom displacement
    mask = dx_norm < 0.2 # only consider steps < 0.2 nm (2 Angstrom)
    
    valid_de = de[mask]
    valid_dot = tensor_dot[mask]
    
    if len(valid_de) < 10:
        log.warning("Not enough valid adjacent frames for physics check (maybe shuffled?).")
        return
    
    # Correlation
    stacked = torch.stack([valid_de, valid_dot])
    corr_matrix = torch.corrcoef(stacked)
    correlation = corr_matrix[0, 1].item()
    
    log.info(f"Analyzed {len(valid_de)} pairs (step < 0.2nm).")
    log.info(f"Correlation between dE and -F.dx: {correlation:.4f}")
    
    # Ratio check (for units)
    # dE = k * (-F.dx)
    # slope = <dE * -F.dx> / <(-F.dx)^2>
    slope = (valid_de * valid_dot).mean() / (valid_dot.pow(2).mean())
    log.info(f"Estimated Unit Slope (dE / -F.dx): {slope:.4f}")
    
    if correlation < 0.5:
        log.error("POOR CORRELATION! Energy likely does not match Forces (or is Total Energy).")
    elif slope > 100 or slope < 0.01:
        log.warning("Slope far from 1.0! Likely MAJOR unit mismatch (e.g. kcal vs kJ vs nm vs A).")
    else:
        log.info(f"Physics check PASSED? (Corr={correlation:.2f}, Slope={slope:.2f})")

class TinyDataset(Dataset):
    def __init__(self, pos, forces, energies, atom_types):
        self.pos = pos
        self.f = forces
        self.e = energies
        self.atom_types = atom_types
    def __len__(self): return len(self.pos)
    def __getitem__(self, idx):
        return {"x": self.pos[idx], "f": self.f[idx], "e": self.e[idx], "a": self.atom_types}

def run_training_sweep(pos, forces, energies, atom_types):
    log.info("\n--- Check 3: Energy Model Sweep (Scaled) ---")
    
    # 0. SCALE FORCES based on Physics Check
    SCALE_FACTOR = 0.07
    log.info(f"Applying Scaling Factor: {SCALE_FACTOR} to Forces")
    forces = forces * SCALE_FACTOR
    
    device = torch.device('cpu')
    n_samples = 1024
    ds = TinyDataset(pos[:n_samples], forces[:n_samples], energies[:n_samples], atom_types)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    
    # Stats
    f_std = ds.f.std()
    e_mean = ds.e.mean()
    e_std = ds.e.std()
    
    log.info(f"Stats: F_std={f_std:.4f}, E_mean={e_mean:.4f}, E_std={e_std:.4f}")

    # --- Mode A: Energy Gradient ---
    log.info(">> Model A: Energy Gradient (F = -grad E)")
    
    # Config
    cfg = EGNNConfig(n_layers=3, hidden_dim=64, use_rbf=True)
    model = EnergyEGNN(6, 64, 3, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 50
    for epoch in range(epochs):
        losses = []
        losses_f = []
        losses_e = []
        
        for batch in dl:
            x = batch['x'].to(device).requires_grad_(True)
            f_t = batch['f'].to(device)
            e_t = batch['e'].to(device)
            a = batch['a'].to(device)
            
            # Predict Energy
            E_pred, _ = model(x, a, create_graph=True)
            
            # Calculate Force = -grad E
            # But E_pred matches (E - mean) / std ? Or just E?
            # Let's verify scaling.
            # If we train E_pred to match (E - mean) / std:
            # E_phys = E_pred * e_std + e_mean
            # F_phys = -grad(E_phys) = -grad(E_pred) * e_std
            # We want F_phys to match f_t
            # Loss F = MSE(F_phys, f_t) / f_std^2  (Normalized MSE)
            #        = MSE(-grad(E_pred)*e_std, f_t) / f_std^2
            #        = MSE(-grad(E_pred), f_t / e_std) * (e_std/f_std)^2 ?
            # Simpler: F_pred_norm = -grad(E_pred) * (e_std / f_std)
            # Target: f_t_norm = f_t / f_std
            
            grad_outputs = torch.ones_like(E_pred)
            grads = torch.autograd.grad(E_pred, x, grad_outputs, create_graph=True, retain_graph=True)[0]
            
            F_pred_norm = -grads * (e_std / f_std)
            
            # Targets
            f_t_norm = f_t / f_std # (f * 0.07) / std
            e_t_norm = (e_t - e_mean) / e_std
            
            loss_f = nn.MSELoss()(F_pred_norm, f_t_norm)
            loss_e = nn.MSELoss()(E_pred, e_t_norm)
            
            loss = loss_f + loss_e
            
            opt.zero_grad(); loss.backward(); opt.step()
            
            losses.append(loss.item())
            losses_f.append(loss_f.item())
            losses_e.append(loss_e.item())
            
        if epoch % 5 == 0:
            log.info(f"   Epoch {epoch}: Loss={np.mean(losses):.4f} (F={np.mean(losses_f):.4f}, E={np.mean(losses_e):.4f})")
            
    # Final R2
    mse_f = np.mean(losses_f) # Variance is 1
    r2_f = 1 - mse_f
    mse_e = np.mean(losses_e)
    r2_e = 1 - mse_e
    
    log.info(f"Final R2: Force={r2_f:.4f}, Energy={r2_e:.4f}")
    if r2_f > 0.2:
        log.info("SUCCESS: Energy Gradient Model works with scaling!")
    else:
        log.error("FAILURE: Energy Gradient Model failed (R2 <= 0.2)")

def main():
    try:
        # Load Shard 0 which should have sequential data for physics check
        shard_path = "data/processed/ala2_all_atom/shards/shard_00000.pt"
        log.info(f"Loading {shard_path}...")
        shard = torch.load(shard_path)
        
        pos = shard['positions'] # [N, 22, 3]
        atom_types = shard['atom_types']
        
        # We need Forces and Energies corresponding to this shard
        # Assuming Shard 0 corresponds to indices 0..N of the global arrays
        # (This is how most sharding works, but verifying length is good)
        
        log.info("Loading global force/energy arrays (partial)...")
        # Load all is slow, map_location? No, just load, it's 300MB
        all_forces = torch.load("data/processed/ala2_all_atom/al_forces_ref.pt")
        all_energies = torch.load("data/processed/ala2_all_atom/al_energies_ref.pt")
        
        n_shard = len(pos)
        forces = all_forces[:n_shard]
        energies = all_energies[:n_shard]
        
        log.info(f"Loaded {n_shard} frames.")
        
        check_data_stats(pos, forces, energies)
        check_physics_consistency(pos, forces, energies)
        run_training_sweep(pos, forces, energies, atom_types)
        
    except Exception as e:
        log.exception(f"Debug Suite Failed: {e}")

if __name__ == "__main__":
    main()
