from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from metastategen.utils import get_logger
from metastategen.eval.ramachandran import _detect_and_convert_to_degrees

log = get_logger("free_energy")

def prob_from_phi_psi(phi_psi: np.ndarray, bins: int = 180) -> Tuple[np.ndarray, list[float]]:
    """
    Return probability mass P over bins (sum P = 1) and plot extent in degrees.

    We compute a 2D histogram with density=False (counts), then normalize.
    """
    if phi_psi.ndim != 2 or phi_psi.shape[1] != 2:
        raise ValueError(f"Expected phi_psi [N,2], got {phi_psi.shape}")

    # Ensure degrees in [-180,180]
    phi_psi_deg = _detect_and_convert_to_degrees(phi_psi)

    phi = phi_psi_deg[:, 0].astype(np.float64, copy=False)
    psi = phi_psi_deg[:, 1].astype(np.float64, copy=False)

    counts, xedges, yedges = np.histogram2d(
        phi, psi,
        bins=bins,
        range=[[-180.0, 180.0], [-180.0, 180.0]],
        density=False,
    )
    total = float(np.sum(counts))
    if total <= 0:
        raise ValueError("Histogram counts sum to zero; cannot form probability.")
    P = (counts / total).astype(np.float64)
    extent = [float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])]
    return P.astype(np.float32), extent

def free_energy_from_prob(P: np.ndarray, kT: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """
    Compute free energy surface: F = -kT log(P + eps), with constant offset removed (min=0).
    """
    if kT <= 0:
        raise ValueError("kT must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    P64 = P.astype(np.float64, copy=False)
    F = -kT * np.log(P64 + eps)
    F = F - np.nanmin(F)
    return F.astype(np.float32)

def plot_free_energy(F: np.ndarray, extent: list[float], outpath: Path, title: str = "") -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        F.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Free energy (arb., offset removed)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

