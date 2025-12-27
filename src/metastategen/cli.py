import argparse
import sys
from pathlib import Path

from metastategen.utils import get_logger, set_deterministic
from metastategen.eval.ramachandran import load_phi_psi_npz, plot_ramachandran_density
from metastategen.eval.free_energy import (
    prob_from_phi_psi,
    free_energy_from_prob,
    plot_free_energy,
)

log = get_logger("msgen")

def _cmd_report(args: argparse.Namespace) -> int:
    set_deterministic(args.seed)
    dihedrals_path = Path(args.dihedrals)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    phi_psi = load_phi_psi_npz(dihedrals_path)
    log.info(f"Loaded phi/psi: shape={phi_psi.shape}, dtype={phi_psi.dtype}")

    density_png = outdir / "reference_ramachandran_density.png"
    fe_png = outdir / "reference_free_energy.png"

    plot_ramachandran_density(
        phi_psi=phi_psi,
        outpath=density_png,
        bins=args.bins,
        title=args.title or "Reference Ramachandran density (mdshare dihedrals)",
    )

    P, extent = prob_from_phi_psi(phi_psi, bins=args.bins)
    F = free_energy_from_prob(P, kT=args.kT, eps=args.eps)
    plot_free_energy(
        F=F,
        extent=extent,
        outpath=fe_png,
        title=args.title_fe or f"Reference free energy F=-kT log p (kT={args.kT:g}, offset removed)",
    )

    log.info(f"Wrote: {density_png}")
    log.info(f"Wrote: {fe_png}")
    return 0

def _not_impl(_: argparse.Namespace, name: str) -> int:
    log.error(f"Subcommand '{name}' is not implemented in Day-1.")
    return 2

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="msgen", description="metastategen-diffusion CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # report
    pr = sub.add_parser("report", help="Generate evaluation plots from mdshare dihedrals")
    pr.add_argument("--dihedrals", type=str, required=True, help="Path to dihedrals NPZ")
    pr.add_argument("--outdir", type=str, default="reports/reference", help="Output directory")
    pr.add_argument("--bins", type=int, default=180, help="Bins per dimension (phi, psi)")
    pr.add_argument("--kT", type=float, default=1.0, help="kT used for F=-kT log p")
    pr.add_argument("--eps", type=float, default=1e-12, help="Epsilon to avoid log(0)")
    pr.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    pr.add_argument("--title", type=str, default=None, help="Title for density plot")
    pr.add_argument("--title-fe", dest="title_fe", type=str, default=None, help="Title for free-energy plot")
    pr.set_defaults(func=_cmd_report)

    # placeholders
    pt = sub.add_parser("train", help="(placeholder) Train diffusion model")
    pt.set_defaults(func=lambda a: _not_impl(a, "train"))
    ps = sub.add_parser("sample", help="(placeholder) Sample from diffusion model")
    ps.set_defaults(func=lambda a: _not_impl(a, "sample"))
    pa = sub.add_parser("al", help="(placeholder) Active learning loop")
    pa.set_defaults(func=lambda a: _not_impl(a, "al"))

    return p

def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())

