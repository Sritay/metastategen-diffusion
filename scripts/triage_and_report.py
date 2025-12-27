import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from metastategen.utils import get_logger
from metastategen.eval.coverage import kl_from_phi_psi
from metastategen.eval.ramachandran import _detect_and_convert_to_degrees, load_phi_psi_npz
from metastategen.eval.rmsd import greedy_cluster
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

log = get_logger("triage_and_report")


def _load_uncertainty(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    u = torch.load(path)
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    return np.asarray(u)


def _compute_phi_psi(samples: torch.Tensor, pdb_path: Path) -> np.ndarray:
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long, device=samples.device)
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return _detect_and_convert_to_degrees(degs.cpu().numpy())


def _parse_rejection_fracs(spec: str) -> list[float]:
    fracs = []
    for item in spec.split(","):
        val = float(item.strip())
        if val < 0 or val >= 1:
            raise ValueError(f"rejection fraction must be in [0,1): {val}")
        fracs.append(val)
    return fracs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncertainty", type=str, required=True)
    ap.add_argument("--samples", type=str, required=True)
    ap.add_argument("--pdb", type=str, default="data/raw/alanine-dipeptide-nowater.pdb")
    ap.add_argument("--ref-dihedrals", type=str, default="data/raw/alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
    ap.add_argument("--bins", type=int, default=180)
    ap.add_argument("--rmsd-thresh", type=float, default=0.75)
    ap.add_argument("--rejection-fracs", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--tau", type=float, default=None, help="Uncertainty threshold (overrides rejection-fracs)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    uncertainty = _load_uncertainty(Path(args.uncertainty)).reshape(-1)
    samples = torch.load(args.samples)

    if samples.shape[0] != uncertainty.shape[0]:
        raise ValueError(f"samples N={samples.shape[0]} != uncertainty N={uncertainty.shape[0]}")

    ref_phi_psi = load_phi_psi_npz(Path(args.ref_dihedrals))
    gen_phi_psi = _compute_phi_psi(samples, Path(args.pdb))

    n_total = uncertainty.shape[0]
    sort_idx = np.argsort(uncertainty)

    rows = []
    if args.tau is not None:
        keep_mask = uncertainty <= float(args.tau)
        keep_idx = np.where(keep_mask)[0]
        n_kept = int(keep_idx.shape[0])
        rejection_rate = 1.0 - (n_kept / n_total)
        if n_kept == 0:
            kl = float("nan")
            basin_count = 0
        else:
            kl = kl_from_phi_psi(gen_phi_psi[keep_idx], ref_phi_psi, bins=args.bins)
            kept_samples = samples[torch.tensor(keep_idx, dtype=torch.long)]
            _, medoids, _ = greedy_cluster(kept_samples, rmsd_thresh=args.rmsd_thresh)
            basin_count = len(medoids)
        rows.append(
            {
                "rejection_rate": rejection_rate,
                "n_kept": n_kept,
                "kl_to_ref": kl,
                "basin_count": basin_count,
            }
        )
    else:
        for r in _parse_rejection_fracs(args.rejection_fracs):
            n_kept = max(1, int(round((1.0 - r) * n_total)))
            keep_idx = sort_idx[:n_kept]
            rejection_rate = 1.0 - (n_kept / n_total)
            kl = kl_from_phi_psi(gen_phi_psi[keep_idx], ref_phi_psi, bins=args.bins)
            kept_samples = samples[torch.tensor(keep_idx, dtype=torch.long)]
            _, medoids, _ = greedy_cluster(kept_samples, rmsd_thresh=args.rmsd_thresh)
            rows.append(
                {
                    "rejection_rate": rejection_rate,
                    "n_kept": n_kept,
                    "kl_to_ref": kl,
                    "basin_count": len(medoids),
                }
            )

    out_path = Path(args.out) if args.out else Path(args.uncertainty).parent / "triage_curve.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rejection_rate", "n_kept", "kl_to_ref", "basin_count"])
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
