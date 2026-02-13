#!/usr/bin/env python
"""
Self-contained L0 pipeline: diffusion sampling + geometric refinement.
Explicit progress printing to bypass logging buffering issues.
"""
import sys, time, torch
from pathlib import Path

def p(msg):
    print(msg, flush=True)

def main():
    p("=== L0 Pipeline: 100 frames, geometric refinement ===")
    
    from metastategen.workflows.sampling import load_diffusion_model, geometric_refinement_loop
    from metastategen.data.topology import MoleculeTopology
    from metastategen.reconstruct import align_and_reconstruct
    from metastategen.models.diffusion import constrain_bonds
    from metastategen.utils import io_formats
    import mdtraj as md

    device = torch.device("cpu")
    out_dir = Path("runs/verify_l0_geo/l0_100_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Load ---
    p("Loading model...")
    model, diffusion, diff_cfg = load_diffusion_model(
        Path("configs/verify_l0_local.yaml"),
        Path("runs/verify_l0_geo/checkpoints/final.pt"),
        device
    )
    model.eval()
    p(f"  scale_factor={diffusion.cfg.scale_factor}, T={diffusion.cfg.T}")
    
    p("Loading topology...")
    topo = MoleculeTopology("data/miscanthus/L0.pdb", topology_path="data/miscanthus/L0.psf")
    constraints_diff = topo.infer_constraints().to(device)
    diff_types = topo.get_atom_types().to(device)
    heavy_indices = topo.heavy_indices
    p(f"  {topo.n_atoms} atoms, {topo.n_heavy_atoms} heavy, {constraints_diff.shape[0]} constraints")

    # Global constraints for refinement
    cons_local_list = constraints_diff.tolist()
    constraints_global = []
    for row in cons_local_list:
        i_sub, j_sub, d = int(row[0]), int(row[1]), float(row[2])
        if i_sub < len(heavy_indices) and j_sub < len(heavy_indices):
            constraints_global.append([heavy_indices[i_sub], heavy_indices[j_sub], d])
    constraints_refine = torch.tensor(constraints_global, device=device)

    # Template
    traj_templ = md.load("data/miscanthus/L0.pdb", top="data/miscanthus/L0.psf")
    templ_all = torch.tensor(traj_templ.xyz[0], dtype=torch.float32).to(device)
    
    # --- Generate ---
    n_samples = 100
    batch_size = 10
    n_batches = n_samples // batch_size
    
    initial_samples = []
    refined_samples = []
    
    for i in range(n_batches):
        B = batch_size
        a_batch = diff_types.unsqueeze(0).expand(B, -1)
        shape = (B, topo.n_heavy_atoms, 3)
        
        t0 = time.time()
        p(f"Batch {i+1}/{n_batches}: diffusion sampling ({B} frames)...")
        x_gen = diffusion.p_sample_loop(model, shape, a_batch, constraints=constraints_diff)
        
        # Unscale
        x_gen = x_gen / diffusion.cfg.scale_factor
        dt = time.time() - t0
        p(f"  Diffusion done in {dt:.1f}s. Range=[{x_gen.min():.3f}, {x_gen.max():.3f}]")
        
        # Reconstruct
        p(f"  Reconstructing all-atom...")
        x_recon = align_and_reconstruct(x_gen, templ_all, heavy_indices, topology=topo)
        initial_samples.append(x_recon.clone().cpu())
        
        # Geometric Refinement
        p(f"  Geometric refinement...")
        x_refined = geometric_refinement_loop(
            x_recon.to(device),
            constraints_refine,
            n_steps=100,
            clash_cutoff=0.25
        )
        refined_samples.append(x_refined.cpu())
        p(f"  Batch {i+1} complete. Refined range=[{x_refined.min():.3f}, {x_refined.max():.3f}]")
    
    # --- Save ---
    results = {
        "initial_positions": torch.cat(initial_samples, dim=0),
        "refined_positions": torch.cat(refined_samples, dim=0),
        "atom_types": diff_types.cpu()
    }
    
    out_path = out_dir / "refined_results.pt"
    torch.save(results, out_path)
    p(f"Saved {out_path}")
    
    # Save PDB
    try:
        io_formats.save_outputs(
            torch.cat(refined_samples, dim=0),
            str(out_dir),
            ["pdb"],
            prefix="refined",
            topology=traj_templ.topology
        )
        p("Saved PDB output.")
    except Exception as e:
        p(f"PDB save failed (non-critical): {e}")
    
    # --- Sanity Check ---
    pos = results["refined_positions"]
    p(f"\n=== RESULTS ===")
    p(f"Shape: {pos.shape}")
    p(f"Range: [{pos.min():.4f}, {pos.max():.4f}]")
    
    dists = torch.cdist(pos[0:1], pos[0:1])[0]
    mask = torch.eye(pos.shape[1]).bool()
    dists = dists.masked_fill(mask, float('inf'))
    nn_dist = dists.min(dim=1).values.mean()
    p(f"Avg NN dist: {nn_dist:.4f} nm (expect ~0.1-0.2)")
    
    max_r = torch.norm(pos - pos.mean(dim=1, keepdim=True), dim=-1).max()
    p(f"Max radius: {max_r:.4f} nm (expect < 2.0)")
    
    p("DONE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
