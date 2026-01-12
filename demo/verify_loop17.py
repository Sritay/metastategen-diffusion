
import os
import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from metastategen.models.ensemble import load_ensemble, build_diffusion_from_cfg
from metastategen.utils import set_deterministic
from metastategen.models.diffusion import center
from metastategen.models.features import compute_chiral_volume_signal
import mdtraj as md

def compute_dihedrals(pos):
    # Same as debug_loop16_phi.py
    # pos: [B, N, 3]
    # Phi: C_prev(1) - N(3) - CA(4) - C(6)
    p0 = pos[:, 1]
    p1 = pos[:, 3]
    p2 = pos[:, 4]
    p3 = pos[:, 6]
    
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    
    b1_norm = torch.norm(b1, dim=1, keepdim=True)
    b1 = b1 / b1_norm
    
    v = b0 - torch.sum(b0 * b1, dim=1, keepdim=True) * b1
    w = torch.cross(b1, b0, dim=1)
    
    x = torch.sum(v * b2, dim=1)
    y = torch.sum(w * b2, dim=1)
    
    return torch.atan2(y, x)

def compute_bond_lengths(pos):
    # N-CA (3-4)
    # CA-C (4-6)
    # Return mean and std of deviations
    n_ca = torch.norm(pos[:, 3] - pos[:, 4], dim=1)
    ca_c = torch.norm(pos[:, 4] - pos[:, 6], dim=1)
    return n_ca, ca_c

def main():
    run_dir = Path("runs/day11_al_17_hpc")
    config_path = run_dir / "config.yaml"
    
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    meta_path = cfg["data"]["meta_path"]
    if not os.path.exists(meta_path):
         print(f"Warning: {meta_path} not found. Using assumption n=10.")
         n_atoms = 10
         atom_types = torch.zeros(n_atoms, dtype=torch.long, device=device)
    else:
        meta = torch.load(meta_path)
        n_atoms = int(meta["n_atoms"])
        # Find a shard for atom types
        import glob
        pool_source = cfg.get('active_learning', {}).get('oracle_pool_source', "data/processed/ala2/shards")
        shards = glob.glob(f"{pool_source}/*.pt")
        if shards:
            batch = torch.load(shards[0])
            atom_types = batch["atom_types"].to(device)
            print("Loaded atom_types from shard.")
        else:
             print("Warning: No shards found. Using zeros.")
             atom_types = torch.zeros(n_atoms, dtype=torch.long, device=device)

    # Load Ensemble
    members_dir = run_dir / "members"
    member_paths = sorted([d for d in members_dir.iterdir() if d.is_dir()])
    print(f"Found members: {[p.name for p in member_paths]}")
    
    real_n_types = int(atom_types.max().item()) + 1
    ensemble = load_ensemble(member_paths, cfg, n_atom_types=real_n_types, device=device)
    diffusion = build_diffusion_from_cfg(cfg).to(device)
    
    set_deterministic(17)
    
    n_samples = 20 # Enough for stats
    steps = 100
    
    results = {}
    
    # Target Condition
    ref_pdb = md.load("data/raw/alanine-dipeptide-nowater.pdb")
    heavy_idx = ref_pdb.topology.select("symbol != H")
    scale_factor = 7.6
    ref_pos_scaled = torch.tensor(ref_pdb.xyz[0, heavy_idx], dtype=torch.float32) * scale_factor
    target_signal = compute_chiral_volume_signal(ref_pos_scaled.unsqueeze(0)).mean(0) # [N, 1]
    c_batch = target_signal.unsqueeze(0).expand(n_samples, -1, -1).to(device)
    
    print("\n--- Verifying Models ---")
    for i, model in enumerate(ensemble.members):
        model.eval()
        print(f"Sampling from m{i}...")
        
        shape = (n_samples, n_atoms, 3)
        batch_types = atom_types.unsqueeze(0).expand(n_samples, -1)
        
        with torch.no_grad():
            xt = torch.randn(shape, device=device)
            if diffusion.cfg.recenter_every_step:
                xt = center(xt)
            
            # Use CONDITIONING
            xt = diffusion.ddim_sample_loop(model, shape, batch_types, steps=steps, eta=0.0, model_kwargs={"condition": c_batch})
            
        # 1. Chirality (Phi)
        phi = compute_dihedrals(xt)
        phi_deg = torch.rad2deg(phi).cpu().numpy()
        
        # 2. Bond Lengths (unscaled)
        xt_unscaled = xt / scale_factor
        n_ca, ca_c = compute_bond_lengths(xt_unscaled)
        
        print(f"  Phi Mean: {phi_deg.mean():.2f} (std: {phi_deg.std():.2f})")
        
        # Check constraints (nm)
        # N-CA Target: 0.146
        # CA-C Target: 0.151
        n_ca_mu, n_ca_std = n_ca.mean().item(), n_ca.std().item()
        ca_c_mu, ca_c_std = ca_c.mean().item(), ca_c.std().item()
        
        print(f"  N-CA Bond: {n_ca_mu:.4f} +/- {n_ca_std:.4f} (Target 0.146)")
        print(f"  CA-C Bond: {ca_c_mu:.4f} +/- {ca_c_std:.4f} (Target 0.151)")
        
        results[f"m{i}"] = phi_deg
        
        # Simple assertions
        if abs(n_ca_mu - 0.146) > 0.005: 
             print("  [FAIL] N-CA bond length deviation > 0.005")
        if abs(ca_c_mu - 0.151) > 0.005:
             print("  [FAIL] CA-C bond length deviation > 0.005")
             
        # Check Phi clustering (should be generally negative for L-Ala)
        # Fraction of positive Phis (should be low)
        pos_phi_frac = (phi_deg > 0).mean()
        print(f"  Fraction Phi > 0: {pos_phi_frac:.2f}")
        if pos_phi_frac > 0.3:
             print("  [WARN] Significant fraction of positive Phi angles (potential Phi=0 or D-Ala mixture?)")


    # Plot
    plt.figure(figsize=(10, 6))
    for name, phis in results.items():
        plt.hist(phis, bins=20, alpha=0.5, label=name, range=(-180, 180))
    plt.axvline(0, color='k', linestyle='--', label="Phi=0")
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Count")
    plt.title("Phi Distribution - Loop 17 (Conditioned)")
    plt.legend()
    plt.savefig("demo/verify_loop17_phi.png")
    print("\nSaved plot to demo/verify_loop17_phi.png")

if __name__ == "__main__":
    main()
