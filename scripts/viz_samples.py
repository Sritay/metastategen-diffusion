
import argparse
import torch
import numpy as np
from pathlib import Path
from metastategen.utils import get_logger

log = get_logger("viz_samples")

def write_pdb_trajectory(samples: torch.Tensor, template_pdb: Path, out_path: Path):
    """
    Writes a multi-model PDB file using a template PDB for topology.
    Assumes samples are (N, n_atoms, 3) in Angstroms or Nanometers.
    If template is Alanine Dipeptide, it expects appropriate atom count.
    """
    
    with open(template_pdb, 'r') as f:
        lines = f.readlines()
        
    atom_lines = [l for l in lines if l.startswith("ATOM") or l.startswith("HETATM")]
    
    n_atoms_template = len(atom_lines)
    n_samples, n_atoms_sample, _ = samples.shape
    
    if n_atoms_template != n_atoms_sample:
        log.warning(f"Template has {n_atoms_template} atoms but samples have {n_atoms_sample} atoms.")
        # Proceed if it's the 10 vs 22 atom case (just warn)
        if n_atoms_sample == 10 and n_atoms_template == 22:
             log.info("Visualizing Heavy Atoms Only.")
    
    # Check units roughly
    # C-N bond is ~1.3 Angstrom (0.13 nm)
    # If mean distance is < 1, likely nm. PDB needs Angstroms.
    sample_mean_dist = torch.mean(torch.norm(samples[0] - samples[0].mean(0), dim=1))
    scale = 1.0
    if sample_mean_dist < 5.0:
        log.info("Detected Nanometers. Converting to Angstroms for PDB (x10).")
        scale = 10.0
        
    samples = samples * scale
    
    with open(out_path, 'w') as f:
        for i in range(n_samples):
            f.write(f"MODEL     {i+1}\n")
            
            for j, line in enumerate(atom_lines):
                if j >= n_atoms_sample:
                    break
                    
                # PDB Format specific columns
                # x: 30-38, y: 38-46, z: 46-54
                x = samples[i, j, 0].item()
                y = samples[i, j, 1].item()
                z = samples[i, j, 2].item()
                
                # Careful string replacement to preserve PDB columns
                # We reconstruct the line with new coords
                pre = line[:30]
                post = line[54:]
                
                f.write(f"{pre}{x:8.3f}{y:8.3f}{z:8.3f}{post}")
                
            f.write("ENDMDL\n")
            
    log.info(f"Wrote {n_samples} frames to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert .pt samples to .pdb trajectory")
    parser.add_argument("input", type=str, help="Path to .pt samples")
    parser.add_argument("--pdb", type=str, default="data/raw/alanine-dipeptide-nowater.pdb", help="Template PDB")
    parser.add_argument("--out", type=str, default=None, help="Output PDB path")
    parser.add_argument("--num", type=int, default=100, help="Number of samples to write")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        return
        
    samples = torch.load(input_path, map_location='cpu')
    if isinstance(samples, list):
        samples = torch.stack(samples)
    if hasattr(samples, 'numpy'):
        samples = samples  # it's a tensor
    else:
        # maybe it's a dict?
        pass

    # Handle shape
    if len(samples.shape) == 2: # (n_atoms, 3) -> (1, n, 3)
        samples = samples.unsqueeze(0)
        
    log.info(f"Loaded samples shape: {samples.shape}")
    
    # Subsample
    if args.num < samples.shape[0]:
        samples = samples[:args.num]
        
    pdb_template = Path(args.pdb)
    if not pdb_template.exists():
        # Try finding in timewarp if default fails
        alt = Path("data/timewarp/train/ad1-traj-state0.pdb")
        if alt.exists():
            pdb_template = alt
        else:
            log.error(f"Template PDB not found at {pdb_template} or {alt}")
            return
            
    if args.out is None:
        out_path = input_path.with_suffix(".pdb")
    else:
        out_path = Path(args.out)
        
    write_pdb_trajectory(samples, pdb_template, out_path)

if __name__ == "__main__":
    main()
