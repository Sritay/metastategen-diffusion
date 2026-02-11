#!/usr/bin/env python
import argparse
import sys
from metastategen.workflows.sampling import run_sampling

def main():
    parser = argparse.ArgumentParser(description="Proxy script for sampling and refinement")
    
    parser.add_argument("--diff-config", type=str, default="configs/ala2_al_3.yaml")
    parser.add_argument("--diff-ckpt", type=str, default="runs/day8_9_al_3/members/m000/checkpoints/iter_03.pt") 
    parser.add_argument("--force-ckpt", type=str, default="runs/energy_pairwise/best_model.pt")
    
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--refinement-steps", type=int, default=2000)
    parser.add_argument("--step-size", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=298.0) 
    
    parser.add_argument("--out-dir", type=str, default="runs/loop_b_refinement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--keep-percent", type=float, default=1.0)
    parser.add_argument("--output-formats", nargs='+', default=['pdb', 'lammps_data'], 
                        help="List of output formats: pdb, gro, xyz, lammps_dump, lammps_data")
    parser.add_argument("--topology", type=str, default=None, help="Path to topology for generalized sampling")
    
    parser.add_argument("--refinement-mode", type=str, default="mlip", choices=["mlip", "geometric"], help="Refinement type: 'mlip' (Force Field) or 'geometric' (Clash Removal)")
    
    args = parser.parse_args()
    
    return run_sampling(
        diff_config=args.diff_config,
        diff_ckpt=args.diff_ckpt,
        force_ckpt=args.force_ckpt,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        refinement_steps=args.refinement_steps,
        step_size=args.step_size,
        temperature=args.temperature,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        keep_percent=args.keep_percent,
        output_formats=args.output_formats,
        topology_path=args.topology,
        refinement_mode=args.refinement_mode
    )

if __name__ == "__main__":
    sys.exit(main())
