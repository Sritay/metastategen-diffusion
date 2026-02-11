#!/usr/bin/env python
import torch
import numpy as np
import logging
from metastategen.utils import get_logger
from metastategen.data.manager import _check_units
from metastategen.workflows.sampling import geometric_refinement_loop, constrain_bonds

log = get_logger("verify_v023")
logging.basicConfig(level=logging.INFO)

def test_unit_check():
    log.info("--- Testing Unit Safety Check ---")
    
    # CASE 1: Nanometers (Good)
    # C-C bond approx 0.15 nm
    pos_nm = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0]
    ], dtype=torch.float32).unsqueeze(0) # [1, 2, 3]
    atom_types = torch.tensor([0, 0])
    
    log.info("Testing valid nm data (0.15 bond)...")
    _check_units(pos_nm, atom_types) # Should be silent/info
    
    # CASE 2: Angstroms (Bad)
    # C-C bond approx 1.5 A
    pos_ang = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0]
    ], dtype=torch.float32).unsqueeze(0)
    
    log.info("Testing suspicious Angstrom data (1.5 bond)...")
    _check_units(pos_ang, atom_types) # Should log warning

def test_geometric_refinement():
    log.info("--- Testing Geometric Refinement ---")
    
    # Create two atoms with a clash (dist = 0.1 nm, cutoff = 0.25 nm)
    x = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0]
    ], dtype=torch.float32).unsqueeze(0) # [1, 2, 3]
    
    # No constraints for basic clash test
    constraints = torch.empty((0, 3))
    
    log.info(f"Initial Dist: {torch.norm(x[0,0] - x[0,1])}")
    
    x_refined = geometric_refinement_loop(x, constraints, n_steps=100, clash_cutoff=0.25)
    
    final_dist = torch.norm(x_refined[0,0] - x_refined[0,1])
    log.info(f"Final Dist: {final_dist}")
    
    if final_dist >= 0.24: # Allow small epsilon
        log.info("SUCCESS: Clash removed.")
    else:
        log.error("FAILURE: Clash persists.")

def test_constrained_refinement():
    log.info("--- Testing Constrained Refinement ---")
    
    # 3 Atoms: 0-1 bonded (fixed 0.15), 0-2 non-bonded (clashing)
    # 0 at origin
    # 1 at 0.15 x
    # 2 at 0.05 y (Clash with 0)
    
    x = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.0, 0.1, 0.0] # 0.1 dist from 0, clash (cutoff 0.25)
    ], dtype=torch.float32).unsqueeze(0)
    
    # Constraint: 0-1 must be 0.15
    constraints = torch.tensor([[0, 1, 0.15]])
    
    log.info("Running refinement with constraint 0-1=0.15...")
    x_refined = geometric_refinement_loop(x, constraints, n_steps=200, clash_cutoff=0.25)
    
    dist_01 = torch.norm(x_refined[0,0] - x_refined[0,1])
    dist_02 = torch.norm(x_refined[0,0] - x_refined[0,2])
    
    log.info(f"Final 0-1 (Constraint): {dist_01:.4f} (Target 0.15)")
    log.info(f"Final 0-2 (Clash): {dist_02:.4f} (Target > 0.25)")
    
    if abs(dist_01 - 0.15) < 1e-3 and dist_02 > 0.24:
        log.info("SUCCESS: Constraints maintained and clash resolved.")
    else:
         log.error("FAILURE: Constraint broken or clash persists.")

if __name__ == "__main__":
    test_unit_check()
    test_geometric_refinement()
    test_constrained_refinement()
