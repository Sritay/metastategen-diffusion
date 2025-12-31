import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from metastategen.models.energy import EnergyEGNN
from metastategen.models.egnn import EGNNConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("debug_local")

class TinyDataset(Dataset):
    def __init__(self, n_samples=256, outlier_threshold=None):
        # Load a subset of real data
        try:
            print("Loading real data for debugging...")
            forces = torch.load("data/processed/ala2_all_atom/al_forces_ref.pt")
            energies = torch.load("data/processed/ala2_all_atom/al_energies_ref.pt")
            
            # Simple metadata mock if needed, or rely on just the tensors
            # Assuming first shard has positions
            shard_path = "data/processed/ala2_all_atom/shards/shard_00000.pt"
            shard = torch.load(shard_path)
            positions = shard['positions']
            atom_types = shard['atom_types']
            
            # Limit to n_samples
            self.pos = positions[:n_samples]
            self.f = forces[:n_samples]
            self.e = energies[:n_samples]
            self.atom_types = atom_types
            
            # Outlier filtering check
            f_mags = self.f.norm(dim=-1)
            print(f"Dataset Force Stats (Subset N={n_samples}):")
            print(f"  Max Mag: {f_mags.max().item():.4f}")
            print(f"  Mean Mag: {f_mags.mean().item():.4f}")
            print(f"  Std Mag: {f_mags.std().item():.4f}")
            
            if outlier_threshold is not None:
                mask = f_mags.max(dim=-1)[0] < outlier_threshold
                print(f"  Filtering outliers > {outlier_threshold}. Kept {mask.sum()}/{len(mask)}")
                self.pos = self.pos[mask]
                self.f = self.f[mask]
                self.e = self.e[mask]

        except Exception as e:
            print(f"Failed to load real data: {e}")
            print("Generating SYNTHETIC data for purely code verification...")
            self.pos = torch.randn(n_samples, 22, 3)
            self.f = torch.randn(n_samples, 22, 3)
            self.e = torch.randn(n_samples)
            self.atom_types = torch.randint(0, 5, (22,))

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        return {
            "x": self.pos[idx],
            "f": self.f[idx],
            "e": self.e[idx],
            "a": self.atom_types
        }

def train_debug():
    # settings
    lr = 1e-4
    epochs = 20
    batch_size = 32
    hidden_dim = 32 # Small for debug
    n_layers = 2
    
    # device
    device = torch.device('cpu') 
    
    # Dataset
    # Filter forces > 3000 (approx 98-99th percentile based on observation)
    ds = TinyDataset(n_samples=2048, outlier_threshold=3000.0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # Robust Normalization (IQR)
    f_mags = ds.f.norm(dim=-1)
    f_q25 = torch.quantile(f_mags, 0.25)
    f_q75 = torch.quantile(f_mags, 0.75)
    f_iqr = f_q75 - f_q25
    f_scale = f_iqr / 1.349 # Normalize so normal dist has sigma=1
    
    # Fallback if IQR is too small?
    if f_scale < 1e-6: f_scale = ds.f.std()
    
    e_q25 = torch.quantile(ds.e, 0.25)
    e_q75 = torch.quantile(ds.e, 0.75)
    e_iqr = e_q75 - e_q25
    e_scale = e_iqr / 1.349
    # Use Mean for centering to be safer for MSE
    e_center = ds.e.mean() 
    
    # Direct Force Regression Experiment
    print("Switching to DIRECT Force Regression (No Energy Gradient)...")
    
    # Model: Standard EGNN outputting vector per node
    # EGNN outputs [B, N, H] then we project to [B, N, 3]??
    # Actually, standard EGNN updates coordinates x. 
    # For direct force prediction: F_pred = x_new - x_old.
    # Let's use the EGNN coordinate update as the force vector.
    
    cfg = EGNNConfig(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        dropout=0.0,
        use_rbf=True,
        rbf_dim=16,
        rbf_cutoff=1.0
    )
    # We need a wrapper to extract "force" from coordinate update
    class DirectForceEGNN(nn.Module):
        def __init__(self, n_atom_types, hidden_dim, n_layers, cfg):
            super().__init__()
            self.atom_emb = nn.Embedding(n_atom_types, hidden_dim)
            from metastategen.models.egnn import EGNNLayer
            self.layers = nn.ModuleList([EGNNLayer(hidden_dim, cfg=cfg) for _ in range(n_layers)])
            self.final_proj = nn.Linear(hidden_dim, 1) # Not used for force really?
            
        def forward(self, x, a, create_graph=False):
            # We don't need create_graph for direct regression!
            h = self.atom_emb(a)
            x_in = x.clone()
            curr_x = x.clone()
            
            for layer in self.layers:
                h, curr_x = layer(h, curr_x)
            
            # Prediction is the "displacement" learned
            F_pred = curr_x - x_in
            return None, F_pred

    model = DirectForceEGNN(6, hidden_dim, n_layers, cfg).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print("\nStarting Training Loop (Direct Force)...")
    for epoch in range(epochs):
        losses = []
        losses_f = []
        grad_norms = []
        
        for batch in dl:
            x = batch['x'].to(device) # No requires_grad needed
            f_target = batch['f'].to(device)
            a = batch['a'].to(device)
            
            # Predict
            _, F_pred_raw = model(x, a)
            
            # F_pred is in coordinate units.
            # Target F is in force units.
            # We learn F_pred to match F_target_norm
            
            f_t_norm = f_target / f_scale
            
            loss_f = criterion(F_pred_raw, f_t_norm)
            loss = loss_f
            
            opt.zero_grad()
            loss.backward()
            
            # Monitor gradients
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()**2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            losses.append(loss.item())
            losses_f.append(loss_f.item())
            
        avg_loss = sum(losses)/len(losses)
        pred_f_std = 0.0 # TODO capture batch stats if needed
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f} | GradNorm={sum(grad_norms)/len(grad_norms):.4f}")
        # Check prediction std to see if model is collapsing
        pred_e_std = E_pred.std().item()
        pred_f_std = F_pred_norm.std().item()
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f} (F={avg_f:.4f}, E={avg_e:.4f}) | GradNorm={avg_grad:.4f} | PredStd: E={pred_e_std:.4f}, F={pred_f_std:.4f}")

if __name__ == "__main__":
    train_debug()
