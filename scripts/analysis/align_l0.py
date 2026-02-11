#!/usr/bin/env python
import mdtraj as md
import numpy as np
import argparse
from pathlib import Path
from metastategen.utils import get_logger

log = get_logger("align_l0")

def main():
    parser = argparse.ArgumentParser(description="Align L0 structures to a reference ring.")
    parser.add_argument("--input", type=str, required=True, help="Input PDB file (e.g. refined.pdb)")
    parser.add_argument("--output", type=str, default="aligned_l0.pdb", help="Output aligned PDB file")
    parser.add_argument("--topology", type=str, required=True, help="Topology file (PSF/PDB) to identify ring atoms")
    
    args = parser.parse_args()
    
    log.info(f"Loading trajectory from {args.input} with topology {args.topology}...")
    try:
        traj = md.load(args.input, top=args.topology)
    except Exception as e:
        # Fallback: try loading topology from input PDB if specific topology fails or not provided
        log.warning(f"Failed to load with explicit topology: {e}. Trying input PDB as topology.")
        traj = md.load(args.input)

    log.info(f"Loaded {traj.n_frames} frames.")
    
    # Identify a Benzene Ring
    # Lignin L0 typically has aromatic rings. We can find 6-membered carbon rings.
    # mdtraj topology graph
    from metastategen.data.topology import MoleculeTopology
    
    # Use our robust topology class
    meta_topo = MoleculeTopology(args.input, topology_path=args.topology)
    
    rings = meta_topo.rings # List of lists of atom indices
    six_mem_rings = [r for r in rings if len(r) == 6]
    
    if not six_mem_rings:
        log.error("No 6-membered rings found for alignment!")
        log.warning("Aligning to all heavy atoms instead.")
        atom_indices = meta_topo.heavy_indices
    else:
        # Select first 6-ring
        target_ring = six_mem_rings[0]
        # Check if it's all carbon (optional, but good)
        is_carbon_ring = all(traj.topology.atom(i).element.symbol == 'C' for i in target_ring)
        
        if is_carbon_ring:
             log.info(f"Found Carbon Benzene Ring: {target_ring}")
             atom_indices = np.array(target_ring)
        else:
             log.warning(f"Using 6-ring (not all Carbon?): {target_ring}")
             atom_indices = np.array(target_ring)

    # Reference: First frame
    log.info("Aligning all frames to frame 0...")
    traj.superpose(traj, frame=0, atom_indices=atom_indices)
    
    # Save
    log.info(f"Saving aligned trajectory to {args.output}...")
    traj.save_pdb(args.output)
    log.info("Done.")

if __name__ == "__main__":
    main()
