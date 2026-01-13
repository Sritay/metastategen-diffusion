import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from metastategen.utils import get_logger
from metastategen.eval.coverage import kl_from_phi_psi
from metastategen.eval.ramachandran import _detect_and_convert_to_degrees, load_phi_psi_npz
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices

log = get_logger("eval_uncertainty")


def _load_uncertainty(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    u = torch.load(path)
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    return np.asarray(u)


from typing import Optional

def _load_gen_phi_psi(gen_dihedrals: Optional[Path], samples: Optional[Path], pdb_path: Path) -> np.ndarray:
    if gen_dihedrals is not None:
        with np.load(gen_dihedrals) as f:
            if "phi_psi" not in f:
                raise KeyError(f"Expected key 'phi_psi' in {gen_dihedrals}")
            phi_psi = f["phi_psi"]
        return _detect_and_convert_to_degrees(phi_psi)

    if samples is None:
        raise ValueError("Provide either --gen-dihedrals or --samples")

    pos = torch.load(samples)
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long, device=pos.device)
    rads = compute_dihedrals(pos, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return _detect_and_convert_to_degrees(degs.cpu().numpy())


def _build_periodic_tree(ref_phi_psi: np.ndarray):
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError("scipy is required for nearest-neighbor distance") from exc

    offsets = (-360.0, 0.0, 360.0)
    tiled = []
    for dx in offsets:
        for dy in offsets:
            tiled.append(ref_phi_psi + np.array([dx, dy], dtype=np.float32))
    tiled = np.concatenate(tiled, axis=0)
    return cKDTree(tiled)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncertainty", type=str, required=True)
    ap.add_argument("--gen-dihedrals", type=str, default=None, help="NPZ with phi_psi for generated samples")
    ap.add_argument("--samples", type=str, default=None, help="samples.pt (used if gen-dihedrals not provided)")
    ap.add_argument("--pdb", type=str, default="data/raw/alanine-dipeptide-nowater.pdb")
    ap.add_argument("--ref-dihedrals", type=str, default="data/raw/alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
    ap.add_argument("--bins", type=int, default=180)
    ap.add_argument("--n-bins", type=int, default=10, help="Number of uncertainty quantile bins")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    uncertainty_path = Path(args.uncertainty)
    uncertainty = _load_uncertainty(uncertainty_path).reshape(-1)

    gen_phi_psi = _load_gen_phi_psi(
        Path(args.gen_dihedrals) if args.gen_dihedrals else None,
        Path(args.samples) if args.samples else None,
        Path(args.pdb),
    )
    ref_phi_psi = load_phi_psi_npz(Path(args.ref_dihedrals))

    if gen_phi_psi.shape[0] != uncertainty.shape[0]:
        raise ValueError(
            f"Uncertainty length {uncertainty.shape[0]} != generated phi_psi length {gen_phi_psi.shape[0]}"
        )

    tree = _build_periodic_tree(ref_phi_psi)
    nn_dist, _ = tree.query(gen_phi_psi, k=1)

    edges = np.quantile(uncertainty, np.linspace(0.0, 1.0, args.n_bins + 1))
    out_path = Path(args.out) if args.out else uncertainty_path.parent / "uncertainty_bins.csv"

    rows = []
    for i in range(args.n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == args.n_bins - 1:
            mask = (uncertainty >= lo) & (uncertainty <= hi)
        else:
            mask = (uncertainty >= lo) & (uncertainty < hi)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin": i,
                    "uncertainty_min": float(lo),
                    "uncertainty_max": float(hi),
                    "uncertainty_mean": float("nan"),
                    "count": 0,
                    "kl_to_ref": float("nan"),
                    "nn_phi_psi_dist_mean": float("nan"),
                }
            )
            continue

        phi_psi_bin = gen_phi_psi[mask]
        kl = kl_from_phi_psi(phi_psi_bin, ref_phi_psi, bins=args.bins)
        nn_mean = float(np.mean(nn_dist[mask]))
        unc_mean = float(np.mean(uncertainty[mask]))
        rows.append(
            {
                "bin": i,
                "uncertainty_min": float(lo),
                "uncertainty_max": float(hi),
                "uncertainty_mean": unc_mean,
                "count": count,
                "kl_to_ref": kl,
                "nn_phi_psi_dist_mean": nn_mean,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin",
                "uncertainty_min",
                "uncertainty_max",
                "uncertainty_mean",
                "count",
                "kl_to_ref",
                "nn_phi_psi_dist_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
