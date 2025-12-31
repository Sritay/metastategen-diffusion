import argparse
import torch
import numpy as np
from pathlib import Path

from metastategen.utils import get_logger
from metastategen.utils.pdb import get_ala2_heavy_atom_indices
from metastategen.utils.geometry import compute_dihedrals, rad2deg

log = get_logger("compute_dihedrals")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True, help="Path to samples.pt")
    parser.add_argument("--pdb", type=str, default="data/raw/alanine-dipeptide-nowater.pdb")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    samples_path = Path(args.samples)
    pdb_path = Path(args.pdb)
    
    if args.outdir:
        out_dir = Path(args.outdir)
    else:
        out_dir = samples_path.parent
    
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading samples from {samples_path}")
    pos = torch.load(samples_path) # [N, Atoms, 3]
    
    log.info(f"Getting indices from PDB: {pdb_path}")
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long) # [2, 4]
    
    log.info("Computing dihedrals...")
    # pos: [B, Atoms, 3]
    rads = compute_dihedrals(pos, indices) # [B, 2]
    degs = rad2deg(rads)
    
    # Wrap to [-180, 180]
    degs = (degs + 180.0) % 360.0 - 180.0
    
    out_npz = out_dir / "generated_dihedrals.npz"
    np.savez(out_npz, phi_psi=degs.numpy())
    log.info(f"Saved dihedrals to {out_npz}")

if __name__ == "__main__":
    main()
