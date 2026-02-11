#!/usr/bin/env python
import torch
import numpy as np
import mdtraj as md
from metastategen.reconstruct import align_and_reconstruct
from metastategen.data.topology import MoleculeTopology
from metastategen.utils import get_logger

log = get_logger("verify_l0")

def test_l0_reconstruction():
    pdb_path = "data/miscanthus/L0.pdb"
    topo_path = "data/miscanthus/L0.psf"
    
    # Load Topology
    try:
        topo = MoleculeTopology(pdb_path, topology_path=topo_path)
    except Exception as e:
        log.error(f"Failed to load L0 topology: {e}")
        return

    heavy_indices = topo.heavy_indices
    log.info(f"L0 Heavy Indices: {len(heavy_indices)} / {topo.n_atoms} total")
    
    # Load Template (Full Structure)
    traj = md.load(pdb_path, top=topo_path)
    xyz = torch.tensor(traj.xyz[0], dtype=torch.float32) # [N_all, 3]
    
    # Create a "Generated" Backbone
    # Let's take the actual heavy atoms and rotate/translate them to simulate generation
    backbone_true = xyz[heavy_indices] # [N_heavy, 3]
    
    # Apply random rotation/translation
    theta = np.pi / 4
    R = torch.tensor([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    T = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    
    backbone_gen = (backbone_true @ R.T) + T
    backbone_gen = backbone_gen.unsqueeze(0) # [1, N_heavy, 3]
    
    log.info("Running Reconstruct/Kabsch...")
    
    # Reconstruct
    # x_gen: [B, N_heavy, 3]
    # x_template: [N_all, 3]
    # heavy_indices: List[int]
    
    x_recon = align_and_reconstruct(backbone_gen, xyz, heavy_indices) # [1, N_all, 3]
    
    # Check 1: Heavy Atoms should match EXACTLY (since we just rotated them)
    # Extract heavy from recon
    heavy_recon = x_recon[0, heavy_indices]
    
    # RMSD check
    diff = heavy_recon - backbone_gen[0]
    rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean())
    
    log.info(f"Heavy Atom RMSD (Gen vs Recon): {rmsd.item():.6f}")
    
    if rmsd < 1e-4:
        log.info("SUCCESS: Heavy atoms reconstructed perfectly.")
    else:
        log.error("FAILURE: High RMSD for heavy atoms!")
        
    # Check 2: Hydrogens
    # The reconstruction aligns the template to the backbone. 
    # Since our "backbone_gen" is just a rigid transformation of the template backbone,
    # the reconstructed hydrogens should also match the transformed template hydrogens perfectly (rigid body).
    
    full_true_transformed = (xyz @ R.T) + T
    diff_full = x_recon[0] - full_true_transformed
    rmsd_full = torch.sqrt((diff_full ** 2).sum(dim=-1).mean())
    
    log.info(f"Full Atom RMSD (Rigid Transform): {rmsd_full.item():.6f}")
    
    if rmsd_full < 1e-4:
        log.info("SUCCESS: Full structure reconstructed perfectly (Rigid Body).")
    else:
        log.error("FAILURE: Hydrogens misaligned!")

if __name__ == "__main__":
    test_l0_reconstruction()
