import torch
import torch.nn as nn

class GaussianRBF(nn.Module):
    def __init__(self, n_rbf=16, cutoff=10.0, start=0.0):
        super().__init__()
        self.n_rbf = n_rbf
        # Centers uniform from start to cutoff
        self.centers = nn.Parameter(torch.linspace(start, cutoff, n_rbf), requires_grad=False)
        # Gamma implies width. width ~ step size
        step = (cutoff - start) / n_rbf
        self.gamma = nn.Parameter(torch.ones(n_rbf) * (1.0 / step), requires_grad=False)

    def forward(self, dist):
        # dist: [...]
        # return: [..., n_rbf]
        diff = dist.unsqueeze(-1) - self.centers # [..., n_rbf]
        return torch.exp(-self.gamma * (diff ** 2))

class PairwiseEnergyModel(nn.Module):
    def __init__(self, n_atoms=22, hidden_dim=256, n_rbf=32, rbf_cutoff=10.0):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_pairs = n_atoms * (n_atoms - 1) // 2
        
        self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=rbf_cutoff)
        
        # Input dim is now pairs * rbf_dim
        input_dim = self.n_pairs * n_rbf
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, 3] Positions
        Returns:
            e_pred: [B] Energy
        """
        B, N, _ = x.shape
        
        # Compute Pairwise Distances of upper triangle
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        dist = torch.norm(diff, dim=-1) # [B, N, N]
        
        # Extract upper triangle
        rows, cols = torch.triu_indices(N, N, offset=1, device=x.device)
        dist_flat = dist[:, rows, cols] # [B, P]
        
        # RBF Expansion
        # [B, P] -> [B, P, n_rbf]
        rbf_feat = self.rbf(dist_flat)
        
        # Flatten for MLP: [B, P * n_rbf]
        mlp_in = rbf_feat.view(B, -1)
        
        e_pred = self.net(mlp_in).squeeze(-1) # [B]
        
        return e_pred
