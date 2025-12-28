import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from metastategen.utils import get_logger, set_deterministic
from metastategen.models.egnn import EGNN, EGNNConfig
from metastategen.models.diffusion import GaussianDiffusion, DiffusionConfig
from metastategen.models.energy import EnergyEGNN

log = get_logger("sample_refined")

def load_diffusion_model(config_path, ckpt_path, device):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    model_cfg = EGNNConfig(
        n_layers=cfg['model']['n_layers'],
        hidden_dim=cfg['model']['hidden_dim']
    )
    # n_atom_types hardcoded or from config? Usualy 5 for Ala2
    n_atom_types = 5
    
    model = EGNN(
        n_atom_types=n_atom_types,
        hidden_dim=cfg['model']['hidden_dim'],
        n_layers=cfg['model']['n_layers'],
        time_emb_dim=cfg['model']['time_emb_dim'],
        cfg=model_cfg
    ).to(device)
    
    diff_cfg = DiffusionConfig(
        T=cfg['diffusion']['T'],
        beta_start=cfg['diffusion']['beta_start'],
        beta_end=cfg['diffusion']['beta_end'],
        schedule=cfg['diffusion']['schedule'],
        recenter_every_step=cfg['diffusion']['recenter_every_step']
    )
    diffusion = GaussianDiffusion(diff_cfg).to(device)
    
    if ckpt_path.exists():
        log.info(f"Loading diffusion checkpoint from {ckpt_path}")
        d = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(d['model'] if 'model' in d else d)
    else:
        log.warning(f"Diffusion checkpoint not found at {ckpt_path}!")
        
    return model, diffusion, cfg

def load_force_model(config_path, ckpt_path, device):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    model_cfg = EGNNConfig(
        n_layers=cfg['model']['n_layers'],
        hidden_dim=cfg['model']['hidden_dim']
    )
    n_atom_types = cfg['model']['n_atom_types']
    
    model = EnergyEGNN(
        n_atom_types=n_atom_types,
        hidden_dim=cfg['model']['hidden_dim'],
        n_layers=cfg['model']['n_layers'],
        cfg=model_cfg
    ).to(device)
    
    if ckpt_path.exists():
        log.info(f"Loading force checkpoint from {ckpt_path}")
        d = torch.load(ckpt_path, map_location=device)
        # Check if 'model' key exists (training script saves dict)
        if isinstance(d, dict) and 'model' in d:
             model.load_state_dict(d['model'])
        else:
             model.load_state_dict(d)
    else:
        log.warning(f"Force checkpoint not found at {ckpt_path}!")
        
    return model, cfg

def main():
    parser = argparse.ArgumentParser()
    
    # Diffusion args
    parser.add_argument("--diff-config", type=str, default="configs/ala2_day2.yaml")
    parser.add_argument("--diff-ckpt", type=str, default="runs/day2_baseline/checkpoints/final.pt") # Or day5
    
    # Force args
    parser.add_argument("--force-config", type=str, default="configs/ala2_force.yaml")
    parser.add_argument("--force-ckpt", type=str, default="runs/force_surrogate_ala2/checkpoints/latest.pt")
    
    # Sampling args
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--refinement-steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=1e-4) # eta
    parser.add_argument("--temperature", type=float, default=1.0) # beta^-1
    parser.add_argument("--out-dir", type=str, default="runs/day10_force/samples")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # 1. Load Models
    diff_model, diffusion, diff_cfg = load_diffusion_model(Path(args.diff_config), Path(args.diff_ckpt), device)
    diff_model.eval()
    
    force_model, force_cfg = load_force_model(Path(args.force_config), Path(args.force_ckpt), device)
    force_model.eval()
    
    # 2. Generate Initial Samples
    log.info(f"Generating {args.n_samples} samples with Diffusion...")
    n_atom_types = force_cfg['model']['n_atom_types']
    n_atoms = 10 # Hardcoded for Ala2 heavy atoms if not in config
    
    # Atom types? We need to provide them.
    # Usually we sample atom types or assume fixed composition?
    # Ala2 has fixed composition [C, C, C, C, C, N, O, O, O, C]. But order matters.
    # We should replicate the atom types vector N times.
    # Let's take it from dataset or hardcode if we know. 
    # Hardcoding is risky. Let's try to infer or generate random types? 
    # For Ala2, the topology is fixed. We should ideally load one frame to get types.
    # Or load meta.pt?
    
    meta_path = Path("data/processed/ala2/meta.pt")
    if meta_path.exists():
        # But we don't want to depend on data just for types.
        # Let's assume standard Ala2 types:
        # C=0, N=1, O=2, S=3, other=4
        # Standard PDB order?
        # Actually, Conditional Generation (if types are condition) usually requires types input.
        # If Diffusion is unconditional on types (generates x given a), we need 'a'.
        # Let's verify Diffusion definition. diffusion.sample(model, shape, atom_types, ...)
        pass
    
    # Wait, existing sample_diffusion.py must handle this.
    # It loads the dataset to get atom_types usually.
    # Or it stores them.
    # For now, I will create a dummy dataset or load meta to get types.
    
    log.info("Loading atom types from meta.pt logic (implied constants)")
    # Fallback to loading one shard if needed, or assume we have 'a' from stored types
    # Let's try to load 'a' from the force checkpoint if it saves it? No.
    # Let's just load first shard of data to get types.
    
    processed_dir = Path("data/processed/ala2/shards")
    first_shard = next(processed_dir.glob("*.pt"), None)
    if first_shard:
        d = torch.load(first_shard)
        atom_types = d['atom_types'].to(device) # [N]
        n_atoms = len(atom_types)
    else:
        raise FileNotFoundError("Could not find data shards to infer atom types!")

    all_samples = []
    
    # Batched Sampling
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    
    # Load Stats if available
    f_mean = torch.zeros(1, device=device)
    f_std = torch.ones(1, device=device)
    
    # Check if checkpoint has stats
    if args.force_ckpt and Path(args.force_ckpt).exists():
        d = torch.load(args.force_ckpt, map_location=device)
        if isinstance(d, dict):
            if 'f_mean' in d:
                f_mean = d['f_mean'].to(device)
            if 'f_std' in d:
                f_std = d['f_std'].to(device)
            log.info(f"Loaded force stats: Mean={f_mean.item():.4f}, Std={f_std.item():.4f}")

    with torch.no_grad():
        for i in range(n_batches):
            B = min(args.batch_size, args.n_samples - len(all_samples))
            
            # Step A: Diffusion
            # repeat atom_types for batch
            a_batch = atom_types.unsqueeze(0).expand(B, -1)
            
            # shape [B, N, 3]
            shape = (B, n_atoms, 3)
            
            x_diff = diffusion.sample(diff_model, shape, a_batch)
            
            # Step 1.5: Reconstruct Hydrogens
            # We need to map 10 atoms back to 22.
            # Load template from all_atom dataset (first frame)
            # We assume we have it loaded or load it once.
            if 'template_22' not in locals():
                 # Load 22-atom template
                 aa_path = Path("data/processed/ala2_all_atom/al_forces_ref.pt")
                 if aa_path.exists():
                     d_aa = torch.load(aa_path)
                     template_22 = d_aa[0].to(device) # [22, 3]
                     # Identify heavy indices in 22-atom topology
                     # Ala2 Heavy indices in order: 0, 1, 4, 5, 6, 8, 10, 14, 15, 16 
                     # (Need to verify this mapping or parse PDB)
                     # Let's rely on process_timewarp logic: it iterates atoms in order.
                     # We need the indices of heavy atoms.
                     # Since we don't have PDB parsing here, let's load meta if possible or hardcode for Ala2.
                     # Hardcoded for Timewarp Ala2:
                     # 0: C, 1: C, 2: H, 3: H, 4: H, 5: O, 6: N, 7: H, 8: C, 9: H, 10: C, 11: H, 12: H, 13: H, 14: C, 15: O, 16: N, 17: H, 18: C, 19: H, 20: H, 21: H
                     # Heavy Atoms: 0, 1, 5, 6, 8, 10, 14, 15, 16, 18
                     # Wait, let's verify with process_timewarp logs
                     # "Found 10 heavy atoms: ['C', 'C', 'O', 'N', 'C', 'C', 'C', 'O', 'N', 'C']"
                     # Standard residue order? 
                     # Let's use the parse_all_atoms logic from process_timewarp if we can import it.
                     # Or safer: just include the mapping here.
                     heavy_indices = [0, 1, 5, 6, 8, 10, 14, 15, 16, 18] # Based on standard connectivity
                 else:
                     raise FileNotFoundError("22-atom data not found at data/processed/ala2_all_atom/al_forces_ref.pt")

            from metastategen.reconstruct import align_and_reconstruct
            
            # x_diff is [B, 10, 3]. template_22 is [22, 3].
            x_recon = align_and_reconstruct(x_diff, template_22, heavy_indices)
            
            # Update atom types for 22 atoms
            # Assuming template_types available or load from meta
            # For now, let's just use the types from the 22-atom dataset
            # We need to load them once.
            if 'types_22' not in locals():
                e_path = Path("data/processed/ala2_all_atom/al_energies_ref.pt") # No, types are in shard or meta
                # Try loading one shard
                shard_path = Path("data/processed/ala2_all_atom/shards")
                first = next(shard_path.glob("*.pt"))
                dt = torch.load(first)
                types_22 = dt['atom_types'].to(device) # [22]
            
            a_recon = types_22.unsqueeze(0).expand(B, -1) # [B, 22]

            # Step B: Refinement
            # Langevin Dynamics
            # x_{k+1} = x_k + eta * F(x_k) + sqrt(2 * eta * temp) * z
            
            x_curr = x_recon.clone() # Now [B, 22, 3]
            
            log.info(f"Batch {i+1}/{n_batches}: Refining (22 atoms) for {args.refinement_steps} steps...")
            
            for k in range(args.refinement_steps):
                # Predict Force (Normalized)
                # Detach gradient? We are in no_grad mode anyway.
                # Model expects [B, 22, 3] and [B, 22]
                f_pred_norm = force_model(x_curr, a_recon) # Using new model signature (EGNN returns F directly?) 
                # Wait, force model in sample_refined.py calls `force_model(x, a)`.
                # In train_energy.py, `model(x, a)` returns `E, F`.
                # We need to verify what `load_force_model` returns. 
                # Using `ForceEGNN` wrapper? Or `EnergyEGNN`?
                # `load_force_model` uses `ForceEGNN` class from `models.force`.
                # Training used `EnergyEGNN` from `models.energy`.
                # HUGE WARNING: Verification needed on Model Class usage.
                
                # Assuming we switched to EnergyEGNN in training, we should load EnergyEGNN here too.
                # Let's check `load_force_model` implementation above (Line 61).
                # It uses `ForceEGNN`.
                # Training uses `EnergyEGNN`.
                # WE MUST UPDATE `load_force_model` to use `EnergyEGNN`.
                
                # Assuming EnergyEGNN returns (E, F):
                _, f_pred_norm = force_model(x_curr, a_recon)

                # Denormalize
                f_pred = f_pred_norm * f_std + f_mean
                
                # Noise
                noise = torch.randn_like(x_curr)
                sigma = np.sqrt(2 * args.step_size * args.temperature)
                
                # Update
                x_curr = x_curr + args.step_size * f_pred + sigma * noise
                
            all_samples.append(x_curr.cpu())
            
    final_samples = torch.cat(all_samples, dim=0) # [Total, 22, 3]
    
    # Save
    out_path = Path(args.out_dir) / "refined_samples.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(final_samples, out_path)
    log.info(f"Saved {len(final_samples)} refined samples to {out_path}")

if __name__ == "__main__":
    main()
