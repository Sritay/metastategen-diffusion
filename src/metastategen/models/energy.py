from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

from metastategen.models.egnn import EGNNLayer, EGNNConfig

class EnergyEGNN(nn.Module):
    """
    EGNN for Energy Prediction.
    Outputs:
        E: Scalar Potential Energy [B] or [B, 1]
        F: Forces [B, N, 3] = -grad(E, x)
    """
    def __init__(self, 
                 n_atom_types: int,
                 hidden_dim: int, 
                 n_layers: int, 
                 cfg: Optional[EGNNConfig] = None):
        super().__init__()
        
        if cfg is None:
            cfg = EGNNConfig(n_layers=n_layers, hidden_dim=hidden_dim)
        
        # Atom type embedding
        self.atom_emb = nn.Embedding(n_atom_types, hidden_dim)
        
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, cfg=cfg) 
            for _ in range(n_layers)
        ])
        
        # Readout: Aggregates node embeddings to a single scalar
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, create_graph=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Coordinates [B, N, 3] (Requires Grad!)
            h: Atom types [B, N]
        Returns:
            E: [B]
            F: [B, N, 3]
        """
        # Ensure x requires grad
        if not x.requires_grad:
            x.requires_grad_(True)
            
        B, N, _ = x.shape
        
        # Embed atoms
        if h.dim() == 1:
            h = h.unsqueeze(0).expand(B, -1)
            
        h_emb = self.atom_emb(h) # [B, N, H]
        
        # Pass through layers (updates h and x, but we only care about h for Energy usually, 
        # or we use the updated x for next layers. Standard EGNN uses updated features)
        # However, for Energy model, we want the system Energy based on state X.
        # The standard EGNN updates coordinates X -> X_new. This is for Dynamics/diffusion.
        # For Energy prediction, we might not want to update coordinates, OR we treat the layers 
        # as message passing that refines representation.
        # Standard approach: Don't update X, just update H based on distances.
        # BUT current EGNNLayer updates X. 
        # For an Energy model, x inputs are fixed (the state). We calculate E(x).
        # Internal coordinate updates are "fictitious" or valid latent updates?
        # Typically for E(x), we just do invariant message passing on h, using distances from x.
        # If we update x, E becomes E(x_final(x)).
        
        # We use the standard EGNN layer which updates both features (h) and coordinates (x).
        # While strictly an Energy model E(x) should be invariant to internal coordinate updates,
        # allowing latent coordinate updates can increase expressivity. 
        # The final energy is read out from the node features h.
        
        curr_x = x
        for layer in self.layers:
            h_emb, curr_x = layer(h_emb, curr_x, edge_attr)
            
        # Global Pooling (Sum)
        # [B, N, H] -> [B, H]
        h_pool = torch.sum(h_emb, dim=1)
        
        # Readout
        E = self.readout_mlp(h_pool).squeeze(-1) # [B]
        
        # Force = -grad(E, x)
        # create_graph=True is needed if we want second derivatives (e.g. Hessian), usually False for just forces training?
        # NO: we are training F_pred vs F_ref. Loss = (F_pred - F_ref)^2.
        # F_pred contains gradients of model weights.
        # So we differentiate E w.r.t x.
        # This requires graph only for x, but we need the backward pass of Loss w.r.t weights to flow THROUGH the grad(E,x) op.
        # So YES, create_graph=True is required for training.
        
        grad_outputs = torch.ones_like(E)
        gradients = torch.autograd.grad(
            outputs=E,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=create_graph, # Crucial for training forces
            retain_graph=True,
            only_inputs=True
        )[0]
        
        F = -gradients
        
        return E, F
