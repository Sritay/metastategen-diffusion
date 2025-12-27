from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional


def _safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """Calculates the norm of a tensor with a small epsilon to avoid NaN gradients."""
    return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=keepdim), min=eps))


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Functions.
    """
    def __init__(self, n_rbf: int, cutoff: float = 1.0, start: float = 0.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        # Centers evenly spaced
        self.centers = nn.Parameter(torch.linspace(start, cutoff, n_rbf), requires_grad=False)
        # Widths
        self.gamma = nn.Parameter(torch.tensor((n_rbf / (cutoff - start))**2), requires_grad=False)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: [..., 1]
        Returns:
            [..., n_rbf]
        """
        return torch.exp(-self.gamma * (dist - self.centers)**2)

class MLP(nn.Module):
    """Simple Multilayer Perceptron."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EGNNConfig:
    n_layers: int = 4
    hidden_dim: int = 128
    edge_mlp_layers: int = 2
    node_mlp_layers: int = 2
    coord_mlp_layers: int = 2
    dropout: float = 0.0
    use_rbf: bool = True
    rbf_dim: int = 64
    rbf_cutoff: float = 1.0


class EGNNLayer(nn.Module):
    """
    Equivariant Graph Neural Network Layer.
    """
    def __init__(self, hidden_dim: int, cfg: EGNNConfig, edge_attr_dim: int = 0):
        super().__init__()
        self.distance_eps = 1e-8
        
        self.distance_eps = 1e-8
        self.cfg = cfg
        
        if cfg.use_rbf:
            self.rbf = GaussianRBF(n_rbf=cfg.rbf_dim, cutoff=cfg.rbf_cutoff)
            dist_dim = cfg.rbf_dim
        else:
            self.rbf = None
            dist_dim = 1
        
        # Edge model: Inputs are [h_i, h_j, dist_feats, edge_attr]
        edge_in_dim = 2 * hidden_dim + dist_dim + edge_attr_dim
        self.phi_e = MLP(edge_in_dim, hidden_dim, hidden_dim, 
                         n_layers=cfg.edge_mlp_layers, dropout=cfg.dropout)
        
        # Node model: Inputs are [h_i, m_i_agg]
        self.phi_h = MLP(2 * hidden_dim, hidden_dim, hidden_dim, 
                         n_layers=cfg.node_mlp_layers, dropout=cfg.dropout)
        
        # Coord model: Inputs are [m_ij] -> outputs scalar weight
        self.phi_x = MLP(hidden_dim, 1, hidden_dim, 
                         n_layers=cfg.coord_mlp_layers, dropout=cfg.dropout)
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, H = h.shape
        
        # Pairwise differences and distances
        x_i = x[:, :, None, :]  # [B, N, 1, 3]
        x_j = x[:, None, :, :]  # [B, 1, N, 3]
        diff = x_i - x_j        # [B, N, N, 3]
        
        dist_sq = torch.sum(diff**2, dim=-1, keepdim=True) # [B, N, N, 1]
        dist_sq = torch.clamp(dist_sq, min=self.distance_eps)
        
        # Construct edge inputs
        h_i = h[:, :, None, :].expand(B, N, N, H)
        h_j = h[:, None, :, :].expand(B, N, N, H)
        
        if self.rbf is not None:
             # Compute actual distance
             dist = torch.sqrt(dist_sq)
             dist_feats = self.rbf(dist) # [B, N, N, R]
        else:
             dist_feats = dist_sq # [B, N, N, 1]

        edge_inputs = [h_i, h_j, dist_feats]
        if edge_attr is not None:
            edge_inputs.append(edge_attr)
            
        e_in = torch.cat(edge_inputs, dim=-1)
        
        # Compute messages m_ij
        m_ij = self.phi_e(e_in) # [B, N, N, H]
        
        # Coordinate update (vector * scalar)
        coord_weights = self.phi_x(m_ij) # [B, N, N, 1]
        
        # Eq: x_i = x_i + sum (x_i - x_j) * weights
        delta = torch.sum(diff * coord_weights, dim=2) # Sum over j -> [B, N, 3]
        
        # Clip delta for stability
        delta = torch.clamp(delta, -10.0, 10.0)
        x_new = x + delta
        
        # Node update
        m_i_agg = torch.sum(m_ij, dim=2) # [B, N, H]
        h_in = torch.cat([h, m_i_agg], dim=-1)
        h_new = h + self.phi_h(h_in)
        h_new = self.norm(h_new)
        
        return h_new, x_new


class EGNN(nn.Module):
    """
    Stacked EGNN Model. 
    Predicts noise epsilon_hat [B, N, 3] from noisy coordinates.
    """
    def __init__(self, 
                 n_atom_types: int,
                 hidden_dim: int, 
                 n_layers: int, 
                 time_emb_dim: int,
                 cfg: Optional[EGNNConfig] = None):
        super().__init__()
        
        if cfg is None:
            cfg = EGNNConfig(n_layers=n_layers, hidden_dim=hidden_dim)
        
        # Time embedding (Sinusoidal)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_emb_dim = time_emb_dim
        
        # Atom type embedding
        self.atom_emb = nn.Embedding(n_atom_types, hidden_dim)
        
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, cfg=cfg) 
            for _ in range(n_layers)
        ])
        
        # Output projection (optional, but standard EGNN usually returns coords directly)
        # We don't need a specific output head if we just use the coordinate updates.
        
    def _get_time_embedding(self, t: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Standard sinusoidal embedding."""
        half_dim = self.time_emb_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        if self.time_emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return self.time_mlp(emb)

    def forward(self, x: torch.Tensor, h: torch.Tensor, t: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Noisy Coordinates [B, N, 3]
            h: Atom types (Indices) [B, N] OR [N]
            t: Time step [B]
        Returns:
            eps_hat: Predicted noise [B, N, 3] (calculated as displacement)
        """
        x_in = x.clone()
        
        # Handle atom types
        if h.dim() == 1:
            h = h.unsqueeze(0).expand(x.shape[0], -1) # [B, N]
        
        h_emb = self.atom_emb(h) # [B, N, H]
        
        # Add time embedding to nodes
        t_emb = self._get_time_embedding(t, x.device) # [B, H]
        h_emb = h_emb + t_emb[:, None, :]
        
        for layer in self.layers:
            h_emb, x = layer(h_emb, x, edge_attr)
            
        # Return the displacement as the predicted epsilon
        return x - x_in
