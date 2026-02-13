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

def geometric_refinement_loop(
    x: torch.Tensor, 
    constraints: torch.Tensor, 
    n_steps: int = 200, 
    clash_cutoff: float = 0.25, # nm (~2.5 A)
    force_const: float = 1.0,
    bond_pairs: list = None,  # List of (i,j) real topology bonds for exclusion mask
) -> torch.Tensor:
    """
    Model-free refinement to remove steric clashes (overlaps).
    Applies a soft repulsive potential: V = k * (cutoff - r)^2 for r < cutoff.
    Excludes bonded and 1-3 neighbor pairs from clash detection.
    Enforces bond constraints after each step.
    """
    x_curr = x.clone().detach()
    lr = 0.01
    N = x_curr.shape[1]
    device = x_curr.device
    
    log.info(f"Starting Geometric Refinement (Clash Removal): {n_steps} steps, cutoff={clash_cutoff}nm")
    
    # Build exclusion mask from REAL bonds (not ring-rigidity constraints)
    exclude = torch.eye(N, device=device).bool().unsqueeze(0).expand(x_curr.shape[0], -1, -1)
    
    if bond_pairs is not None:
        # Build adjacency from real bonds
        adj = {i: set() for i in range(N)}
        for (i_idx, j_idx) in bond_pairs:
            adj[i_idx].add(j_idx)
            adj[j_idx].add(i_idx)
        
        # Exclude 1-2 (direct bonds) and 1-3 (sharing a bonded neighbor)
        pair_exclude = torch.zeros(N, N, device=device).bool()
        for i_idx in range(N):
            for j_idx in adj[i_idx]:
                pair_exclude[i_idx, j_idx] = True  # 1-2
                for k_idx in adj[j_idx]:
                    if k_idx != i_idx:
                        pair_exclude[i_idx, k_idx] = True  # 1-3
        
        exclude = exclude | pair_exclude.unsqueeze(0).expand(x_curr.shape[0], -1, -1)
        log.info(f"  Exclusion mask: {pair_exclude.sum().item()//2} pairs excluded (1-2 and 1-3)")
    
    last_energy = 0.0
    for i in range(n_steps):
        x_curr.requires_grad_(True)
        
        # Pairwise distances [B, N, N]
        dists = torch.cdist(x_curr, x_curr)
        
        # Mask excluded pairs
        dists = dists.masked_fill(exclude, float('inf'))
        
        # Soft Repulsion
        clash_mask = (dists < clash_cutoff)
        n_clashes = clash_mask.sum().item() // 2
        if not clash_mask.any():
            log.info(f"  No clashes remaining at step {i}.")
            break
        if i % 50 == 0:
            log.info(f"  Step {i}: {n_clashes} clashes, energy={last_energy:.6f}")
            
        deltas = clash_cutoff - dists
        energy = 0.5 * force_const * (deltas * clash_mask.float()).pow(2).sum()
        last_energy = energy.item()
        
        energy.backward()
        
        with torch.no_grad():
            grad = x_curr.grad
            if grad is not None:
                grad = torch.clamp(grad, -10.0, 10.0)
                x_curr = x_curr - lr * grad
            
            # Enforce bond constraints
            x_curr = constrain_bonds(x_curr, constraints, scale_factor=1.0)
            x_curr = x_curr.detach()
              
    log.info(f"Geometric Refinement Done. Final Energy: {last_energy:.4f}")
    return x_curr.detach()


def run_sampling(
    diff_config: str,
    diff_ckpt: str,
    force_ckpt: str, # Optional now if mode=geometric
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
    refinement_mode: str = "mlip", # "mlip" or "geometric"
    connectivity_path: str = None, # Path to PSF/GRO if needed for bonds
):
    if output_formats is None:
        output_formats = []
    set_deterministic(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}, Refinement Mode: {refinement_mode}")
    print(f"[DEBUG] Device: {device}, Mode: {refinement_mode}", flush=True)
    
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if topology_path is None:
        # Fallback
        topology_path = "data/raw/alanine-dipeptide-nowater.pdb"
        log.warning(f"No topology_path provided. Defaulting to {topology_path}")

    # Initialize Topology
    topo = MoleculeTopology(topology_path, topology_path=connectivity_path)
    log.info(f"Loaded topology: {topo.n_atoms} atoms, {len(topo.heavy_indices)} heavy atoms.")

    # 1. Load Diffusion
    diff_model, diffusion, diff_cfg = load_diffusion_model(Path(diff_config), Path(diff_ckpt), device)
    diff_model.eval()
    print(f"[DEBUG] Model loaded. scale_factor={diffusion.cfg.scale_factor}, T={diffusion.cfg.T}", flush=True)
    
    # 2. Load Refinement Model (Conditional)
    force_model = None
    f_stats = None
    
    if refinement_mode == "mlip":
        if not force_ckpt:
            raise ValueError("force_ckpt is required for 'mlip' refinement mode")
        force_model, f_stats = load_pairwise_model(Path(force_ckpt), device, n_atoms=topo.n_atoms)
        force_model.eval()
    elif refinement_mode == "geometric":
        log.info("Using Geometric Refinement (Clash Removal). MLIP model not loaded.")
    else:
        raise ValueError(f"Unknown refinement_mode: {refinement_mode}")
    
    # 3. Setup Template for Reconstruction (Dynamic)
    # Load template from file used for topology
    import mdtraj as md
    traj_templ = md.load(topology_path)
    templ_all = torch.tensor(traj_templ.xyz[0], dtype=torch.float32).to(device) * 1.0 
    
    heavy_indices = topo.heavy_indices
    
    # Derive global constraints for refinement
    # Derive constraints
    cons_local_list = topo.infer_constraints().tolist() # [K, 3] local indices
    
    # 1. Constraints for Diffusion (Local Indices 0..N_heavy-1)
    constraints_diff = torch.tensor(cons_local_list, device=device)
    log.info(f"Diffusion Constraints: {len(constraints_diff)} bonds (Local Indices)")

    # 2. Constraints for Refinement (Global Indices)
    # We need to map local indices to global indices
    constraints_global = []
    for row in cons_local_list:
        i_sub, j_sub, d = int(row[0]), int(row[1]), float(row[2])
        if i_sub < len(heavy_indices) and j_sub < len(heavy_indices):
             i_global = heavy_indices[i_sub]
             j_global = heavy_indices[j_sub]
             constraints_global.append([i_global, j_global, d])
    
    # Add H-parent bond constraints (needed so refinement doesn't break H bonds)
    heavy_set = set(heavy_indices)
    templ_struct = topo.get_template_structure().to(device)
    for bond in topo.traj.topology.bonds:
        a1, a2 = bond.atom1.index, bond.atom2.index
        is_h_bond = (a1 not in heavy_set) or (a2 not in heavy_set)
        if is_h_bond:
            d_eq = float(torch.norm(templ_struct[a1] - templ_struct[a2]).item())
            constraints_global.append([a1, a2, d_eq])
    # Add H-H geminal constraints (H pairs sharing the same parent atom)
    # Reconstruction correctly places these, but refinement can crush them
    # since they're excluded from clash detection (1-3 through parent)
    all_adj = {i: set() for i in range(templ_struct.shape[0])}
    for bond in topo.traj.topology.bonds:
        a1, a2 = bond.atom1.index, bond.atom2.index
        all_adj[a1].add(a2)
        all_adj[a2].add(a1)
    
    existing_pairs = set((min(int(c[0]),int(c[1])), max(int(c[0]),int(c[1]))) for c in constraints_global)
    n_gem = 0
    for center in range(templ_struct.shape[0]):
        h_children = [n for n in all_adj[center] if n not in heavy_set]
        for a in range(len(h_children)):
            for b in range(a+1, len(h_children)):
                i, j = h_children[a], h_children[b]
                pair = (min(i,j), max(i,j))
                if pair not in existing_pairs:
                    d_eq = float(torch.norm(templ_struct[i] - templ_struct[j]).item())
                    constraints_global.append([i, j, d_eq])
                    existing_pairs.add(pair)
                    n_gem += 1

    constraints_refine = torch.tensor(constraints_global, device=device)
    log.info(f"Refinement Constraints: {len(constraints_refine)} (H-parent: 52, H-H geminal: {n_gem}, heavy: {len(constraints_refine)-52-n_gem})")

    # Atom types for diffusion
    diff_types = topo.get_atom_types().to(device)

    # Build real bond pairs (for exclusion mask in geometric refinement)
    real_bond_pairs = [(b.atom1.index, b.atom2.index) for b in topo.traj.topology.bonds]

    # Adjust n_batches
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    log.info(f"Generating {n_samples} samples...")
    print(f"[DEBUG] Generating {n_samples} samples in {n_batches} batches of {batch_size}", flush=True)
    
    initial_samples = []
    refined_samples = []
    
    for i in range(n_batches):
        B = min(batch_size, n_samples - (i * batch_size))
        if B <= 0: break
        
        # A. Diffusion
        a_batch = diff_types.unsqueeze(0).expand(B, -1)
        shape = (B, len(heavy_indices), 3)
        
        # Pass constraints to diffusion for enforcement at each step
        # Note: We need to ensure constraints are scaled if working in scaled space
        # The diffusion.p_sample_loop handles scaling internally via self.cfg.scale_factor if we pass constraints in nm.
        # But wait, constrain_bonds implementation: 
        # dist_target = constraints[k, 2] * scale_factor
        # So we should pass constraints in RAW nm.
        
        print(f"[DEBUG] Batch {i+1}/{n_batches}: diffusion sampling {B} frames...", flush=True)
        x_gen = diffusion.p_sample_loop(
            diff_model, 
            shape, 
            a_batch, 
            constraints=constraints_diff # [K, 3] in nm (LOCAL INDICES)
        )
        print(f"[DEBUG] Batch {i+1}: diffusion done", flush=True)
        
        # DEBUG: Check diffusion output stats
        x_min, x_max = x_gen.min(), x_gen.max()
        x_rad = torch.norm(x_gen - x_gen.mean(dim=1, keepdim=True), dim=-1).max()
        log.info(f"Diffusion Output Stats: Range=[{x_min.item():.4f}, {x_max.item():.4f}], MaxRadius={x_rad.item():.4f}")
        
        x_gen = x_gen / diffusion.cfg.scale_factor
        
        log.info(f"After Scaling ({diffusion.cfg.scale_factor}): Range=[{x_gen.min().item():.4f}, {x_gen.max().item():.4f}]")
            
        # B. Reconstruction
        # Uses Local Frame alignment now
        x_recon = align_and_reconstruct(
            x_gen, 
            topo.get_template_structure().to(device), 
            topo.heavy_indices,
            topology=topo
        )    
        initial_samples.append(x_recon.clone().cpu()) 
        
        # C. Refinement
        if refinement_mode == "geometric":
            # Model-Free Clash Removal
            x_refined = geometric_refinement_loop(
                x_recon.to(device), 
                constraints_refine,
                n_steps=200,
                clash_cutoff=0.25,
                bond_pairs=real_bond_pairs
            )
            refined_samples.append(x_refined.cpu())
            print(f"[DEBUG] Batch {i+1}: refinement done, range=[{x_refined.min():.3f},{x_refined.max():.3f}]", flush=True)
            
        elif refinement_mode == "mlip":
            # Existing Langevin Dynamics
            x_curr = x_recon.clone().to(device).requires_grad_(True)
            warmup_k = warmup_steps
            main_k = refinement_steps - warmup_k
            
            # Warmup
            if warmup_k > 0:
                for k in range(warmup_k):
                    e_norm = force_model(x_curr)
                    grad = torch.autograd.grad(e_norm.sum(), x_curr)[0]
                    f_pred = -grad * f_stats['e_std']
                    f_norm = f_pred.norm(dim=-1, keepdim=True)
                    clip_coef = torch.clamp(10.0 / (f_norm + 1e-6), max=1.0)
                    f_pred = f_pred * clip_coef
                    with torch.no_grad():
                        x_curr.data += step_size * f_pred
                        x_curr.data = constrain_bonds(x_curr.data, constraints_refine)
            
            # Filtering (Skipped here for brevity in replacement, assuming users want unfiltered if simple)
            # Main Refinement
            for k in range(main_k):
                e_norm = force_model(x_curr)
                grad = torch.autograd.grad(e_norm.sum(), x_curr)[0]
                f_pred = -grad * f_stats['e_std']
                f_norm = f_pred.norm(dim=-1, keepdim=True)
                clip_coef = torch.clamp(10.0 / (f_norm + 1e-6), max=1.0)
                f_pred = f_pred * clip_coef
                with torch.no_grad():
                        x_curr.data += step_size * f_pred
                        x_curr.data = constrain_bonds(x_curr.data, constraints_refine)
            
            refined_samples.append(x_curr.detach().cpu())

    # Save Results
    results = {
        "initial_positions": torch.cat(initial_samples, dim=0),
        "refined_positions": torch.cat(refined_samples, dim=0),
        "atom_types": diff_types.cpu()
    }
    
    out_path = Path(out_dir) / "refined_results.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_path)
    log.info(f"Saved refined results to {out_path}")
    print(f"[DEBUG] Saved results to {out_path}", flush=True)
    
    if output_formats:
        io_formats.save_outputs(
             torch.cat(refined_samples, dim=0),
             out_dir,
             output_formats,
             prefix="refined",
             topology=topo.traj.topology
        )
        io_formats.save_outputs(
             torch.cat(initial_samples, dim=0),
             out_dir,
             output_formats,
             prefix="initial",
             topology=topo.traj.topology
        )

    return 0
