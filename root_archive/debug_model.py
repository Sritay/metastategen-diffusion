
import torch
import yaml
from metastategen.models.energy import EnergyEGNN
from metastategen.models.egnn import EGNNConfig
from pathlib import Path

def main():
    print("Initializing Debug Session (WITH SCALING FIX)...")
    
    # Load Config
    config_path = "configs/ala2_energy.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cpu") 
    
    # Init Model
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
    
    # Fake stats from data
    f_std = 1309.0
    e_std = 14.0
    scale_factor = f_std / e_std
    print(f"Using scale factor: {scale_factor:.4f}")
    
    # Fake Data (One batch)
    B = 4
    N = 22 # Updated for 22 atoms (with H)
    x = torch.randn(B, N, 3, requires_grad=True).to(device) * 0.1 
    a = torch.randint(0, n_atom_types, (B, N)).to(device)
    
    print("Running Forward Pass...")
    model.train()
    
    # Simulate training loop logic
    E_pred, _ = model(x, a, create_graph=True)
    
    print(f"E_pred stats: Mean={E_pred.mean().item():.4f}, Std={E_pred.std().item():.4f}")
    
    # Calculate Force
    grad_outputs = torch.ones_like(E_pred)
    gradients = torch.autograd.grad(
        outputs=E_pred,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    F_pred = -gradients * (e_std / f_std)
    
    # Check F_pred magnitude
    print(f"F_pred stats: Mean={F_pred.mean().item():.4f}, Std={F_pred.std().item():.4f}")
    
    if F_pred.std().item() > 0.01:
        print("SUCCESS: F_pred Std is significant (O(1) expected).")
    else:
        print("WARNING: F_pred Std is still very small.")

    # Check Grads w.r.t parameters
    target_f = torch.randn_like(F_pred)
    
    # Check Energy Loss Scale
    target_e = torch.randn(B).to(device) # Normalized energy target (O(1))
    # E_pred_scaled is O(100) if model not trained.
    
    loss_f = torch.nn.functional.mse_loss(F_pred, target_f)
    loss_e = torch.nn.functional.mse_loss(E_pred_scaled, target_e)
    
    rho_e = 1e-4
    loss = loss_f + rho_e * loss_e
    
    print(f"Loss F: {loss_f.item():.4f}")
    print(f"Loss E: {loss_e.item():.4f}")
    print(f"Total Loss: {loss.item():.4f}")
    
    loss.backward()
    
    grads_ok = False
    param_grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
             norm = param.grad.abs().sum().item()
             param_grad_norms.append(norm)
             if norm > 0:
                 grads_ok = True
    
    if grads_ok:
        avg_grad = sum(param_grad_norms)/len(param_grad_norms)
        print(f"Gradients are flowing. Avg Grad Norm: {avg_grad:.4f}")
    else:
        print("CRITICAL: No gradients on parameters!")

if __name__ == "__main__":
    main()
