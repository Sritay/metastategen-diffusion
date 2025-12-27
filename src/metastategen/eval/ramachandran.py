from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from metastategen.utils import get_logger

log = get_logger("ramachandran")

def _load_npz_arrays(npz_path: Path) -> list[np.ndarray]:
    with np.load(npz_path) as fh:
        keys = sorted(fh.keys())
        arrs = [fh[k] for k in keys]
    if len(arrs) == 0:
        raise ValueError(f"No arrays found in NPZ: {npz_path}")
    return arrs

def _detect_and_convert_to_degrees(phi_psi: np.ndarray) -> np.ndarray:
    """Return phi/psi in degrees in [-180, 180] best-effort."""
    if phi_psi.ndim != 2 or phi_psi.shape[1] != 2:
        raise ValueError(f"Expected phi_psi shape [N,2], got {phi_psi.shape}")

    mx = float(np.nanmax(np.abs(phi_psi)))
    x = phi_psi.astype(np.float64, copy=False)

    # Heuristic:
    # - If values are within ~[-pi, pi], treat as radians.
    # - Else if within ~[-180,180] treat as degrees.
    if mx <= (np.pi * 1.2):
        deg = np.rad2deg(x)
        unit = "radians"
    elif mx <= 180.0 * 1.2:
        deg = x
        unit = "degrees"
    else:
        # Fall back: assume radians if extremely large is unlikely for degrees.
        deg = np.rad2deg(x)
        unit = "unknown->assumed radians"

    # Wrap to [-180, 180]
    deg = (deg + 180.0) % 360.0 - 180.0
    log.info(f"Dihedral unit inference: max|x|={mx:.3g} interpreted as {unit}")
    return deg.astype(np.float32)

def load_phi_psi_npz(dihedrals_npz: Path) -> np.ndarray:
    """Load mdshare dihedral NPZ and return concatenated phi/psi as [N,2] float32 degrees."""
    dihedrals_npz = Path(dihedrals_npz)
    if not dihedrals_npz.exists():
        raise FileNotFoundError(dihedrals_npz)

    arrs = _load_npz_arrays(dihedrals_npz)
    # mdshare ALA2: 3 trajectories stored as arr_0, arr_1, arr_2
    for i, a in enumerate(arrs):
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f"Array {i} expected [T,2], got {a.shape}")
    cat = np.concatenate([a for a in arrs], axis=0)
    return _detect_and_convert_to_degrees(cat)

def compute_ramachandran_density(phi_psi_deg: np.ndarray, bins: int = 180) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D histogram density over phi/psi in degrees."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    phi = phi_psi_deg[:, 0].astype(np.float64, copy=False)
    psi = phi_psi_deg[:, 1].astype(np.float64, copy=False)

    H, xedges, yedges = np.histogram2d(
        phi, psi,
        bins=bins,
        range=[[-180.0, 180.0], [-180.0, 180.0]],
        density=True,
    )
    # H is density in (deg^-2). For plotting, density=True is fine.
    return H.astype(np.float32), xedges.astype(np.float32), yedges.astype(np.float32)

def plot_ramachandran_density(phi_psi: np.ndarray, outpath: Path, bins: int = 180, title: str = "") -> None:
    """Save Ramachandran density plot as PNG."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    H, xedges, yedges = compute_ramachandran_density(phi_psi, bins=bins)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Density")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

