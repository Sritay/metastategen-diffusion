import argparse
from pathlib import Path
import numpy as np

from metastategen.utils import get_logger
from metastategen.eval.ramachandran import plot_ramachandran_density, compute_ramachandran_density
from metastategen.eval.free_energy import prob_from_phi_psi, free_energy_from_prob, plot_free_energy

log = get_logger("report_day2")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-npz", type=str, required=True)
    parser.add_argument("--ref-npz", type=str, default="data/raw/alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load Gen
    log.info(f"Loading generated: {args.gen_npz}")
    with np.load(args.gen_npz) as f:
        gen_phi_psi = f['phi_psi'] # [N, 2]

    # Load Ref (handling mdshare multi-traj format)
    log.info(f"Loading reference: {args.ref_npz}")
    from metastategen.eval.ramachandran import load_phi_psi_npz
    ref_phi_psi = load_phi_psi_npz(args.ref_npz)

    # 1. Ramachandran Density Comparison
    log.info("Plotting Density...")
    plot_ramachandran_density(
        gen_phi_psi, 
        outdir / "generated_ramachandran_density.png", 
        title="Generated Density"
    )
    
    # 2. Free Energy Comparison
    log.info("Plotting Free Energy...")
    P, ext = prob_from_phi_psi(gen_phi_psi)
    try:
        F = free_energy_from_prob(P)
        plot_free_energy(
            F, ext, 
            outdir / "generated_free_energy.png", 
            title="Generated Free Energy"
        )
    except ValueError as e:
        log.warning(f"Could not plot free energy (likely empty bins): {e}")

    log.info("Done.")

if __name__ == "__main__":
    main()
