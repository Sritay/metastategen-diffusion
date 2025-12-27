from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

from metastategen.models.egnn import EGNNLayer, EGNNConfig

class ForceEGNN(nn.Module):
    """
    EGNN for Force Prediction (Regression).
    
    Inputs:
        x: Positions [B, N, 3]
        h: Atom types [B, N] (indices)
        
    Outputs:
        f_pred: Predicted Forces [B, N, 3]
        
    Difference from standard EGNN: No time embeddings, direct regression.
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
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Coordinates [B, N, 3]
            h: Atom types (Indices) [B, N] OR [N]
        Returns:
            f_pred: Predicted Force [B, N, 3]
        """
        x_in = x.clone()
        
        # Handle atom types
        if h.dim() == 1:
            h = h.unsqueeze(0).expand(x.shape[0], -1) # [B, N]
        
        h_emb = self.atom_emb(h) # [B, N, H]
        
        # No time embedding
        
        for layer in self.layers:
            h_emb, x = layer(h_emb, x, edge_attr)
            
        # The output of EGNN layers is a coordinate update mechanism.
        # We interpret the total displacement as the predicted force vector.
        # f_pred = x_final - x_initial
        # This aligns with the idea that the network predicts a "shift" or "gradient".
        
        return x - x_in
