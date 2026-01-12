
import torch
import torch.nn as nn
from metastategen.models.egnn import EGNN, EGNNConfig
from metastategen.models.features import compute_active_chiral_features

def debug_feature_magnitude():
    print("--- Debugging Feature Magnitudes (Loop 16 Settings) ---")
    
    # 1. Setup Model
    cfg = EGNNConfig(
        n_layers=1, # Just one layer needed
        hidden_dim=32, 
        use_chiral_features=True,
        rbf_dim=32
    )
    model = EGNN(n_atom_types=5, hidden_dim=32, n_layers=1, time_emb_dim=32, cfg=cfg)
    model.eval()
    
    # 2. Setup Data
    seed_path = "data/processed/ala2/split_12/al_seed.pt"
    data = torch.load(seed_path)
    x = data["positions"][:100] # Batch of 100
    if x.shape[1] > 10: x = x[:, :10, :] # Limit to 10 atoms
    
    scale_factor = 7.6
    x = x * scale_factor
    h = torch.zeros(x.shape[0], x.shape[1], dtype=torch.long) # Dummy atom types
    t = torch.zeros(x.shape[0]) # Dummy time
    
    # 3. Manual Forward Pass snippet (Layer 0)
    # Replicating logic to inspect intermediate values
    
    # Embeddings
    h_emb = model.atom_emb(h)
    t_emb = model._get_time_embedding(t, x.device)
    h_emb = h_emb + t_emb[:, None, :]
    
    print(f"Embedding (h): Mean={h_emb.abs().mean():.4f}, Std={h_emb.std():.4f}")
    
    # Chiral Calc
    chiral_feats = compute_active_chiral_features(x)
    print(f"Chiral Feature: MeanAbs={chiral_feats.abs().mean():.4f}, MaxAbs={chiral_feats.abs().max():.4f}, Clamp=[-5, 5]")
    
    # Inside Layer Logic (Simulated)
    layer = model.layers[0]
    B, N, H = h_emb.shape
    
    # Diff/Dist
    x_i = x[:, :, None, :]
    x_j = x[:, None, :, :]
    diff = x_i - x_j
    dist_sq = torch.sum(diff**2, dim=-1, keepdim=True)
    
    # Neighbors/Messages
    # We need to run the message passing part to get m_i_agg
    # Just run the real layer but hook? Or just run it.
    
    # Let's use a hook on phi_h?
    # phi_h input is [h, m_agg, chiral]
    
    def hook_fn(module, input, output):
        # Input is a tuple roughly (tensor,)
        inp = input[0] # [B, N, 2*H + 1]
        
        # Split it back
        # h: [0:H]
        # m: [H:2H]
        # c: [2H:]
        
        h_part = inp[..., :32]
        m_part = inp[..., 32:64]
        c_part = inp[..., 64:]
        
        print("\n--- Insight inside MLP Input ---")
        print(f"H (Node State) MeanAbs: {h_part.abs().mean():.4f}")
        print(f"M (Message)    MeanAbs: {m_part.abs().mean():.4f}")
        print(f"C (Chiral)     MeanAbs: {c_part.abs().mean():.4f}")
        
    handle = layer.phi_h.register_forward_hook(hook_fn)
    
    # Run Layer
    layer(h_emb, x, chiral_features=chiral_feats)
    
    handle.remove()

if __name__ == "__main__":
    debug_feature_magnitude()
