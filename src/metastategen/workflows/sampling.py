from __future__ import annotations

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from metastategen.utils import get_logger, set_deterministic
from metastategen.models.egnn import EGNN, EGNNConfig
from metastategen.models.diffusion import GaussianDiffusion, DiffusionConfig, constrain_bonds
from metastategen.models.pairwise import PairwiseEnergyModel
from metastategen.reconstruct import align_and_reconstruct
from metastategen.utils import io_formats
from metastategen.data.topology import MoleculeTopology

log = get_logger("sample_refined")

def load_diffusion_model(config_path, ckpt_path, device):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    model_cfg = EGNNConfig(
        n_layers=cfg['model']['n_layers'],
        hidden_dim=cfg['model']['hidden_dim'],
        use_chiral_features=cfg['model'].get('use_chiral_features', False),
        use_rbf=cfg['model'].get('use_rbf', False),
        rbf_dim=cfg['model'].get('rbf_dim', 64),
        rbf_cutoff=cfg['model'].get('rbf_cutoff', 1.0)
    )
    
    if ckpt_path.exists():
        log.info(f"Loading diffusion checkpoint from {ckpt_path}")
        d = torch.load(ckpt_path, map_location=device)
        state_dict = d['model'] if 'model' in d else d
    else:
        raise FileNotFoundError(f"Diffusion checkpoint not found at {ckpt_path}!")

    # Infer n_atom_types from checkpoint if possible
    # EGNN embedding weight: atom_emb.weight [n_types, hidden_dim]
    if 'atom_emb.weight' in state_dict:
        n_ckpt_types = state_dict['atom_emb.weight'].shape[0]
        log.info(f"Inferred {n_ckpt_types} atom types from checkpoint.")
        n_atom_types = n_ckpt_types
    else:
        # Fallback to config or default
        n_atom_types = cfg.get('model', {}).get('n_atom_types', 3)
        log.warning(f"Could not infer atom types from checkpoint. Using config/default: {n_atom_types}")

    model = EGNN(
        n_atom_types=n_atom_types,
        hidden_dim=cfg['model']['hidden_dim'],
        n_layers=cfg['model']['n_layers'],
        time_emb_dim=cfg['model']['time_emb_dim'],
        cfg=model_cfg
    ).to(device)
    
    scale_factor = float(cfg['data'].get('scale_factor', 1.0))
    
    diff_cfg = DiffusionConfig(
        T=cfg['diffusion']['T'],
        beta_start=cfg['diffusion']['beta_start'],
        beta_end=cfg['diffusion']['beta_end'],
        schedule=cfg['diffusion']['schedule'],
        recenter_every_step=cfg['diffusion']['recenter_every_step'],
        scale_factor=scale_factor
    )
    diffusion = GaussianDiffusion(diff_cfg).to(device)
    
    model.load_state_dict(state_dict)
        
    return model, diffusion, cfg

def load_pairwise_model(ckpt_path, device, n_atoms):
    # Fixed Pairwise Config
    model = PairwiseEnergyModel(n_atoms=n_atoms).to(device)
    
    stats = {}
    if ckpt_path.exists():
        log.info(f"Loading pairwise checkpoint from {ckpt_path}")
        d = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(d['model'])
        stats['e_mean'] = d['e_mean'].to(device)
        stats['e_std'] = d['e_std'].to(device)
        stats['f_std'] = d['f_std'].to(device)
    else:
        log.warning(f"Pairwise checkpoint not found at {ckpt_path}!")
        # Default stats for random initialization testing
        stats['e_mean'] = torch.tensor(0.0).to(device)
        stats['e_std'] = torch.tensor(1.0).to(device)
        stats['f_std'] = torch.tensor(1.0).to(device)
        
    return model, stats

# constrain_bonds_22 removed in favor of generalized constrain_bonds from diffusion.py

def run_sampling(
    diff_config: str,
    diff_ckpt: str,
    force_ckpt: str,
    out_dir: str,
    n_samples: int = 100,
    batch_size: int = 100,
    refinement_steps: int = 2000,
    step_size: float = 1e-5,
    temperature: float = 298.0,
    seed: int = 42,
    warmup_steps: int = 1000,
    keep_percent: float = 1.0,
    output_formats: list[str] = None,
    topology_path: str = None,
):
    if output_formats is None:
        output_formats = []
    set_deterministic(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if topology_path is None:
        # Fallback
        topology_path = "data/raw/alanine-dipeptide-nowater.pdb"
        log.warning(f"No topology_path provided. Defaulting to {topology_path}")

    # Initialize Topology
    topo = MoleculeTopology(topology_path)
    log.info(f"Loaded topology: {topo.n_atoms} atoms, {len(topo.heavy_indices)} heavy atoms.")

    # 1. Load Diffusion
    diff_model, diffusion, diff_cfg = load_diffusion_model(Path(diff_config), Path(diff_ckpt), device)
    diff_model.eval()
    
    # 2. Load Pairwise Force (Dynamic n_atoms)
    force_model, f_stats = load_pairwise_model(Path(force_ckpt), device, n_atoms=topo.n_atoms)
    force_model.eval()
    
    # 3. Setup Template for Reconstruction (Dynamic)
    # Load template from file used for topology
    import mdtraj as md
    traj_templ = md.load(topology_path)
    templ_all = torch.tensor(traj_templ.xyz[0], dtype=torch.float32).to(device) * 1.0 
    
    heavy_indices = topo.heavy_indices
    
    # Derive global constraints for refinement
    # topo.infer_constraints() returns indices relative to HEAVY atoms by default?
    # No, let's assume valid remapping.
    # We need constraints for the WHOLE molecule.
    # Workaround: map inferred heavy constraints to global indices
    cons_local = topo.infer_constraints() 
    constraints_global = []
    for row in cons_local:
        i_sub, j_sub, d = int(row[0]), int(row[1]), float(row[2])
        if i_sub < len(heavy_indices) and j_sub < len(heavy_indices):
             i_global = heavy_indices[i_sub]
             j_global = heavy_indices[j_sub]
             constraints_global.append([i_global, j_global, d])
        
    constraints_tensor = torch.tensor(constraints_global, device=device)
    log.info(f"Refinement Constraints: {len(constraints_tensor)} bonds")

    # Atom types for diffusion
    diff_types = topo.get_atom_types().to(device)

    all_samples = []
    # Adjust n_batches: If filtering, we process input batch size fully, then reduce.
    # We still loop based on 'n_samples' which we treat as INPUT samples count for generation.
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    log.info(f"Generating {n_samples} samples...")
    
    initial_samples = []
    refined_samples = []
    
    for i in range(n_batches):
        # Determine current batch size (might be smaller for last batch)
        
        B = min(batch_size, n_samples - (i * batch_size))
        if B <= 0: break
        
        # A. Diffusion
        a_batch = diff_types.unsqueeze(0).expand(B, -1)
        shape = (B, len(heavy_indices), 3)
        
        with torch.no_grad():
            x_gen = diffusion.p_sample_loop(diff_model, shape, a_batch)
            x_gen = x_gen / diffusion.cfg.scale_factor
            
        # B. Reconstruction (Backbone -> All)
        # RENAMED: x_10 -> x_gen, x_22 -> x_recon
        x_recon = align_and_reconstruct(x_gen, templ_all, heavy_indices)
        
        # Store initial *before* any refinement
        initial_samples.append(x_recon.clone().cpu()) 
        
        # C. Warmup Phase
        x_curr = x_recon.clone().to(device).requires_grad_(True)
        
        warmup_k = warmup_steps
        main_k = refinement_steps - warmup_k
        
        if warmup_k > 0:
            log.info(f"Batch {i+1}: Warmup ({warmup_k} steps)...")
            for k in range(warmup_k):
                e_norm = force_model(x_curr)
                grad = torch.autograd.grad(e_norm.sum(), x_curr)[0]
                f_pred = -grad * f_stats['e_std']
                
                f_norm = f_pred.norm(dim=-1, keepdim=True)
                clip_coef = torch.clamp(10.0 / (f_norm + 1e-6), max=1.0)
                f_pred = f_pred * clip_coef

                with torch.no_grad():
                    x_curr.data += step_size * f_pred
                    x_curr.data = constrain_bonds(x_curr.data, constraints_tensor)

        # D. Filtering
        if keep_percent < 1.0:
            with torch.no_grad():
                # Calc energy for filtering
                e_vals = force_model(x_curr) * f_stats['e_std'] + f_stats['e_mean'] # Denormalized for logging? No, model returns norm.
                # Just use e_norm for sorting
                e_norm = force_model(x_curr)
                
                k_keep = int(B * keep_percent)
                k_keep = max(1, k_keep) # Keep at least 1
                
                # Sort
                vals, indices = torch.sort(e_norm)
                keep_idx = indices[:k_keep]
                
                log.info(f"Batch {i+1}: Filtering top {keep_percent*100}% ({B} -> {k_keep}). Best E={vals[0].item():.2f}, Worst kept={vals[k_keep-1].item():.2f}")
                
                # Filter x_curr
                x_curr = x_curr[keep_idx].detach().clone().requires_grad_(True)
                
        # E. Main Refinement Phase
        log.info(f"Batch {i+1}: Main Refinement ({main_k} steps)...")
        for k in range(main_k):
            e_norm = force_model(x_curr)
            grad = torch.autograd.grad(e_norm.sum(), x_curr)[0]
            f_pred = -grad * f_stats['e_std']
            
            f_norm = f_pred.norm(dim=-1, keepdim=True)
            clip_coef = torch.clamp(10.0 / (f_norm + 1e-6), max=1.0)
            f_pred = f_pred * clip_coef

            with torch.no_grad():
                if i == 0 and (k < 5 or k % 5000 == 0 or k == main_k - 1):
                     log.info(f"Main Step {k}: E_norm={e_norm.mean().item():.2f} | Force_norm_pre={f_norm.mean().item():.2f}")
                
                x_curr.data += step_size * f_pred
                x_curr.data = constrain_bonds(x_curr.data, constraints_tensor)
                    
        refined_samples.append(x_curr.detach().cpu())
        
        # Checkpoint every batch (since batches are large)
        if (i+1) % 1 == 0:
            ckpt_data = {
                "initial_positions": torch.cat(initial_samples, dim=0),
                "refined_positions": torch.cat(refined_samples, dim=0),
                "atom_types": diff_types.cpu()
            }
            ckpt_path = Path(out_dir) / f"checkpoint_batch_{i+1:03d}.pt"
            torch.save(ckpt_data, ckpt_path)
            log.info(f"Saved checkpoint to {ckpt_path}")
        
    results = {
        "initial_positions": torch.cat(initial_samples, dim=0),
        "refined_positions": torch.cat(refined_samples, dim=0),
        "atom_types": diff_types.cpu()
    }
    
    out_path = Path(out_dir) / "refined_results.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_path)
    log.info(f"Saved refined results to {out_path}")
    
    # Save requested formats
    if output_formats:
        # Save refined
        io_formats.save_outputs(
             torch.cat(refined_samples, dim=0),
             out_dir,
             output_formats,
             prefix="refined",
             topology=traj_templ.topology
        )
        # Save initial
        io_formats.save_outputs(
             torch.cat(initial_samples, dim=0),
             out_dir,
             output_formats,
             prefix="initial",
             topology=traj_templ.topology
        )

    return 0
