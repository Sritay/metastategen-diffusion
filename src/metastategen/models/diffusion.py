from __future__ import annotations
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

def center(x: torch.Tensor) -> torch.Tensor:
    """Centers coordinates to COM=0."""
    return x - x.mean(dim=1, keepdim=True)

def random_rotation_matrices(batch: int, device: torch.device) -> torch.Tensor:
    """Generates random 3x3 rotation matrices."""
    # Random quaternions
    u1 = torch.rand(batch, device=device)
    u2 = torch.rand(batch, device=device)
    u3 = torch.rand(batch, device=device)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)

    # Convert to rotation matrix
    R = torch.zeros((batch, 3, 3), device=device)
    
    R[:, 0, 0] = 1 - 2 * (q3**2 + q4**2)
    R[:, 0, 1] = 2 * (q2*q3 - q1*q4)
    R[:, 0, 2] = 2 * (q2*q4 + q1*q3)
    
    R[:, 1, 0] = 2 * (q2*q3 + q1*q4)
    R[:, 1, 1] = 1 - 2 * (q2**2 + q4**2)
    R[:, 1, 2] = 2 * (q3*q4 - q1*q2)
    
    R[:, 2, 0] = 2 * (q2*q4 - q1*q3)
    R[:, 2, 1] = 2 * (q3*q4 + q1*q2)
    R[:, 2, 2] = 1 - 2 * (q2**2 + q3**2)
    return R

def apply_rotation(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Applies rotation R [B,3,3] to x [B,N,3]."""
    # x: [B, N, 3] -> transpose to [B, 3, N] for matmul, or usage einsum
    # We want x_new = x @ R.T  OR  (R @ x.T).T
    # Let's use einsum for clarity: "bij,bnj->bni" matches R * x^T
    return torch.einsum("bij,bnj->bni", R, x)

@dataclass
class DiffusionConfig:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "linear"  # "linear" | "cosine"
    recenter_every_step: bool = True
    ddim_eta: float = 0.0

class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion for E(3) equivariant data.
    """
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        
        if cfg.schedule == "linear":
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
        elif cfg.schedule == "cosine":
            steps = cfg.T + 1
            s = 0.008
            t = torch.linspace(0, cfg.T, steps) / cfg.T
            f = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {cfg.schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        # Register buffers
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())
        self.register_buffer("posterior_variance", (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Diffuse the data (x0) to timestep t.
        x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * epsilon
        """
        # Gather coefficients
        t_idx = t - 1
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t_idx].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t_idx].view(-1, 1, 1)
        
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt

    def training_loss(self, model: nn.Module, x0: torch.Tensor, h: torch.Tensor, t: torch.Tensor, rot_aug: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes MSE loss between added noise and predicted noise.
        
        Args:
            model: EGNN model
            x0: Clean coordinates [B, N, 3]
            h: Node features/Types [B, N]
            t: Timesteps [B]
            rot_aug: Whether to apply random rotation to x0 (and noise)
            
        Returns:
            (loss, info_dict)
        """
        if self.cfg.recenter_every_step:
            x0 = center(x0)
            
        # 1. Rotation Augmentation
        if rot_aug:
            R = random_rotation_matrices(x0.shape[0], x0.device)
            x0 = apply_rotation(x0, R)
            
        # 2. Add Noise
        noise = torch.randn_like(x0)
        if self.cfg.recenter_every_step:
            noise = center(noise)

        xt = self.q_sample(x0, t, noise)
        
        if self.cfg.recenter_every_step:
            xt = center(xt)
            
        # 3. Predict Noise
        # Note: Model is expected to return epsilon_hat
        eps_hat = model(xt, h, t)
        
        if self.cfg.recenter_every_step:
            eps_hat = center(eps_hat)

        loss = torch.mean((eps_hat - noise) ** 2)
        
        return loss, {"mse": loss.item()}

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, ...], h: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """
        DDPM Sampling (Stochastic).
        """
        device = self.betas.device
        B = shape[0]
        xt = torch.randn(shape, device=device)
        
        if self.cfg.recenter_every_step:
            xt = center(xt)
            
        # Allow fewer steps (basic striding not implemented here, assuming full T for DDPM)
        T_loop = self.cfg.T
        
        for i in reversed(range(1, T_loop + 1)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # Predict noise
            eps_hat = model(xt, h, t)
            if self.cfg.recenter_every_step:
                eps_hat = center(eps_hat)
            
            # Coefficients
            idx = i - 1
            beta = self.betas[idx]
            alpha = self.alphas[idx]
            alpha_bar = self.alphas_cumprod[idx]
            
            # Mean = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_bar) * eps_hat)
            mean = (1 / torch.sqrt(alpha)) * (xt - (beta / torch.sqrt(1 - alpha_bar)) * eps_hat)
            
            if i > 1:
                sigma = torch.sqrt(self.posterior_variance[idx])
                noise = torch.randn_like(xt)
                xt = mean + sigma * noise
            else:
                xt = mean
                
            if self.cfg.recenter_every_step:
                xt = center(xt)
                
        return xt

    @torch.no_grad()
    def ddim_sample_loop(self, model: nn.Module, shape: Tuple[int, ...], h: torch.Tensor, steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """
        DDIM Sampling.
        """
        device = self.betas.device
        B = shape[0]
        xt = torch.randn(shape, device=device)
        
        if self.cfg.recenter_every_step:
            xt = center(xt)
            
        # Create strided timeline
        times = torch.linspace(self.cfg.T, 1, steps).long().to(device)
        
        for i, t_val in enumerate(times):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            # Next timestep (t-1 in paper, but strided here)
            prev_t_val = times[i+1] if i < len(times)-1 else torch.tensor(0, device=device)
            
            eps_hat = model(xt, h, t)
            if self.cfg.recenter_every_step:
                eps_hat = center(eps_hat)

            # DDIM update variables
            idx = t_val - 1
            prev_idx = prev_t_val - 1
            
            alpha_bar = self.alphas_cumprod[idx]
            alpha_bar_prev = self.alphas_cumprod[prev_idx] if prev_t_val > 0 else torch.tensor(1.0, device=device)
            
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
            
            # Predict x0
            pred_x0 = (xt - torch.sqrt(1 - alpha_bar) * eps_hat) / torch.sqrt(alpha_bar)
            
            # Direction to xt
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_hat
            
            # Noise
            noise = torch.randn_like(xt) if eta > 0 else 0.0
            
            xt = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
            
            if self.cfg.recenter_every_step:
                xt = center(xt)
                
        return xt
