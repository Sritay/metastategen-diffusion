
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

# Minimal Phi calculation (dihedral angle)
def compute_dihedrals(pos):
    # pos: [B, N, 3]
    # Ala2 backbone indices: C(prev)-N-CA-C
    # Standard PDB definitions for Phi: C(prev)-N-CA-C
    # Check find_heavy_indices.py or standard mapping
    # 0: CH3, 1: C=O, 2: N, 3: H, 4: CA, 5: HA, 6: CB, 7: HB1, 8: HB2, 9: HB3, 10: C=O, 11: N, 12: H, 13: CH3
    # Wait, mdshare 10 atoms usually:
    # 0: C, 1: O, 2: N, 3: CA, 4: CB, 5: C, 6: O, 7: N, 8: CA, 9: CB? No.
    # Let's rely on standard indices if typically used.
    # Standard 10 heavy atoms for Ala2 in mdshare:
    # 0: CH3 (ACE)
    # 1: C (ACE)
    # 2: O (ACE)
    # 3: N
    # 4: CA
    # 5: CB
    # 6: C
    # 7: O
    # 8: N (NME)
    # 9: CH3 (NME)
    
    # Phi: C(0)-N(3)-CA(4)-C(6) -> Wait ACE C is index 1.
    # Phi: C_prev(1) - N(3) - CA(4) - C(6)
    
    p0 = pos[:, 1] # C_prev
    p1 = pos[:, 3] # N
    p2 = pos[:, 4] # CA
    p3 = pos[:, 6] # C
    
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    
    # normalize b1
    b1 /= torch.norm(b1, dim=1, keepdim=True)
    
    # v = projection of b0 onto plane perpendicular to b1
    # v = b0 - <b0, b1>*b1
    v = b0 - torch.sum(b0 * b1, dim=1, keepdim=True) * b1
    w = torch.cross(b1, b0, dim=1) # also perpendicular to b1
    
    # x = <v, b2>
    # y = <w, b2>
    x = torch.sum(v * b2, dim=1)
    y = torch.sum(w * b2, dim=1)
    
    return torch.atan2(y, x)

def compute_chiral_volume(pos):
    # pos: [B, N, 3]
    # Simple proxy: (b1 x b2) . b3 for some backbone vectors?
    # Or just use the one from features.py to see what the model sees?
    # The models/features.py one is local per node.
    # The user says "chiral volume (*5000)".
    # Let's just check Phi for now as requested.
    pass

def main():
    run_dir = Path("runs/day10_al_16_hpc")
    config_path = run_dir / "config.yaml"
    
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Atom Types (Stubbing minimal if file access issue, but should be fine)
    # We need meta_path from cfg
    meta_path = cfg["data"]["meta_path"]
    if not os.path.exists(meta_path):
        print(f"Warning: {meta_path} not found. Using dummy atom types for Ala2 (10 atoms).")
        # Just assume 10 atoms, all type 0 is fine for shape, but model uses embeddings.
        # Need correct mapping.
        # Let's try to locate the file or infer.
        # Typically 10 atoms.
        n_atoms = 10
        atom_types = torch.zeros(n_atoms, dtype=torch.long, device=device) # Dummy
    else:
        meta = torch.load(meta_path)
        n_atoms = int(meta["n_atoms"])
        # We also need the atom_types tensor usually found in data shards
        # For this repro we might cheat if we can't find a shard easily.
        # Let's try to find one.
        import glob
        if 'active_learning' in cfg and 'oracle_pool_source' in cfg['active_learning']:
             pool_source = cfg['active_learning']['oracle_pool_source']
        else:
             # Fallback or try to guess
             pool_source = "data/processed/ala2/shards"
             
        shards = glob.glob(f"{pool_source}/*.pt")
        if shards:
            batch = torch.load(shards[0])
            atom_types = batch["atom_types"].to(device)
            print("Loaded atom_types from shard.")
        else:
             print("Warning: No shards found. Using zeros for atom_types.")
             atom_types = torch.zeros(n_atoms, dtype=torch.long, device=device)

    # Load Ensemble
    members_dir = run_dir / "members"
    member_paths = sorted([d for d in members_dir.iterdir() if d.is_dir()])
    print(f"Found members: {[p.name for p in member_paths]}")
    
    # Re-use load_ensemble logic
    real_n_types = int(atom_types.max().item()) + 1
    ensemble = load_ensemble(member_paths, cfg, n_atom_types=real_n_types, device=device)
    diffusion = build_diffusion_from_cfg(cfg).to(device)
    
    set_deterministic(42)
    
    n_samples = 5
    steps = 100 # Fast check
    
    results = {}
    
    for i, model in enumerate(ensemble.members):
        model.eval()
        print(f"Sampling from m{i}...")
        
        # Sampling loop (simplified from sample_ensemble.py)
        shape = (n_samples, n_atoms, 3)
        batch_types = atom_types.unsqueeze(0).expand(n_samples, -1)
        
        with torch.no_grad():
            xt = torch.randn(shape, device=device)
            if diffusion.cfg.recenter_every_step:
                xt = center(xt)
                
            # Compute Target Condition (L-Ala) from reference
            # Use fixed 7.6 scale_factor matching config assumption if not available
            # We will just load reference PDB here locally
            ref_pdb = md.load("data/raw/alanine-dipeptide-nowater.pdb")
            # Subset to heavy atoms (10) to match model
            heavy_idx = ref_pdb.topology.select("symbol != H")
            scale_factor = 7.6
            ref_pos_scaled = torch.tensor(ref_pdb.xyz[0, heavy_idx], dtype=torch.float32) * scale_factor
            target_signal = compute_chiral_volume_signal(ref_pos_scaled.unsqueeze(0)).mean(0)
            c_batch = target_signal.unsqueeze(0).expand(n_samples, -1, -1).to(device)

            # Using DDIM for speed
            xt = diffusion.ddim_sample_loop(model, shape, batch_types, steps=steps, eta=0.0, model_kwargs={"condition": c_batch})
            
        # Compute metrics
        phi = compute_dihedrals(xt)
        phi_deg = torch.rad2deg(phi).cpu().numpy()
        results[f"m{i}"] = phi_deg
        
        print(f"m{i} Phi mean: {phi_deg.mean():.2f}, std: {phi_deg.std():.2f}")
        
    # Validating "Near 0"
    # Plot
    plt.figure(figsize=(10, 6))
    for name, phis in results.items():
        plt.hist(phis, bins=20, alpha=0.5, label=name, range=(-180, 180))
    
    plt.axvline(0, color='k', linestyle='--', label="Phi=0")
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Count")
    plt.title("Phi Distribution for AL Loop 16 Models")
    plt.legend()
    plt.savefig("demo/debug_loop16_phi.png")
    print("Saved plot to demo/debug_loop16_phi.png")
    
    # ASCII Histogram
    print("\n--- Phi Distribution (Counts per bin) ---")
    bins = np.linspace(-180, 180, 10)
    for name, phis in results.items():
        hist, _ = np.histogram(phis, bins=bins)
        print(f"{name}: {hist}")
    print("Bins edges:", bins)


if __name__ == "__main__":
    main()
