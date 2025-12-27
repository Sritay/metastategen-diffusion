from __future__ import annotations

from typing import Tuple

import numpy as np

from metastategen.eval.free_energy import prob_from_phi_psi


def kl_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P || Q) for discrete distributions."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    P = P.astype(np.float64, copy=False)
    Q = Q.astype(np.float64, copy=False)
    P = P / (np.sum(P) + eps)
    Q = Q / (np.sum(Q) + eps)
    return float(np.sum(P * np.log((P + eps) / (Q + eps))))


def kl_from_phi_psi(
    gen_phi_psi: np.ndarray, ref_phi_psi: np.ndarray, bins: int = 180, eps: float = 1e-12
) -> float:
    P, _ = prob_from_phi_psi(gen_phi_psi, bins=bins)
    Q, _ = prob_from_phi_psi(ref_phi_psi, bins=bins)
    return kl_divergence(P, Q, eps=eps)


def wasserstein_phi_psi(gen_phi_psi: np.ndarray, ref_phi_psi: np.ndarray) -> float:
    """1D Wasserstein distance averaged over phi and psi (degrees)."""
    try:
        from scipy.stats import wasserstein_distance
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError("scipy is required for Wasserstein distance") from exc
    w_phi = wasserstein_distance(gen_phi_psi[:, 0], ref_phi_psi[:, 0])
    w_psi = wasserstein_distance(gen_phi_psi[:, 1], ref_phi_psi[:, 1])
    return float(0.5 * (w_phi + w_psi))
