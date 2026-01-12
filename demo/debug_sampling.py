
import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from metastategen.models.diffusion import GaussianDiffusion, DiffusionConfig
from metastategen.models.egnn import EGNN, EGNNConfig

def debug_sampling():
    print("Initializing Model (Loop 9 Config)...")
    
    # Recreate Loop 9 Model
    egnn_cfg = EGNNConfig(
        n_layers=4, hidden_dim=128, 
        use_rbf=True, rbf_dim=128, rbf_cutoff=20.0
    )
    model = EGNN(n_atom_types=3, hidden_dim=128, n_layers=4, time_emb_dim=128, cfg=egnn_cfg)
    
    diff_cfg = DiffusionConfig(T=1000, schedule="cosine", recenter_every_step=True)
    diffusion = GaussianDiffusion(diff_cfg)
    
    # Load Checkpoint (Iter 10 from Loop 10)
    ckpt_path = "runs/day10_al_10_hpc/members/m000/checkpoints/final.pt"
    if not os.path.exists(ckpt_path):
        print(f"Ckpt not found: {ckpt_path}")
        return

    print(f"Loading checkpoint {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False) # strict=False to be safe with RBF buffers
    model.eval()
    
    # Try Sampling
    print("Starting Sampling (1 sample)...")
    shape = (1, 10, 3)
    
    # We want to track the SIZE (Norm) of the molecule during sampling
    # Use DDIM as in Loop 9 (eta=0.0, steps=100)
    
    device = torch.device("cpu")
    B = shape[0]
    # x_T ~ N(0, 1)
    xt = torch.randn(shape, device=device)
    h = torch.zeros((B, 10), dtype=torch.long) # Dummy atom types
    
    initial_norm = torch.norm(xt, dim=-1).mean().item()
    print(f"Step 100 (Start): Mean Atom Dist from Origin: {initial_norm:.4f}")
    
    # Custom Sampling Loop to print stats
    times = torch.linspace(1000, 1, 100).long()
    
    with torch.no_grad():
        for i, t_val in enumerate(times):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            
            # Predict
            eps_hat = model(xt, h, t)
            
            # Stats
            xt_norm = torch.norm(xt, dim=-1).mean().item()
            eps_norm = torch.norm(eps_hat, dim=-1).mean().item()
            
            if i % 10 == 0:
                print(f"Step {1000-t_val.item():3d}/1000: x_norm={xt_norm:.4f}, eps_norm={eps_norm:.4f}")
                
            # Step (Simplified DDIM Euler)
            # This logic should match diffusion.py exactly, but for debug I just want to see trend
            # Actually better to call the real function if I can hook into it? 
            # I will just run the loop manually using logic from diffusion.py
            
            # DDIM update variables
            idx = t_val - 1
            prev_t_val = times[i+1] if i < len(times)-1 else torch.tensor(0)
            prev_idx = prev_t_val - 1
            
            alpha_bar = diffusion.alphas_cumprod[idx]
            alpha_bar_prev = diffusion.alphas_cumprod[prev_idx] if prev_t_val > 0 else torch.tensor(1.0)
            
            pred_x0 = (xt - torch.sqrt(1 - alpha_bar) * eps_hat) / torch.sqrt(alpha_bar)
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps_hat
            xt = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            
    final_norm = torch.norm(xt, dim=-1).mean().item()
    print(f"Final: Mean Atom Dist from Origin: {final_norm:.4f}")
    
    # Unscale
    scale = 7.6
    unscaled_norm = final_norm / scale
    print(f"Unscaled Final Norm: {unscaled_norm:.4f} (Expected ~0.15)")

if __name__ == "__main__":
    debug_sampling()
