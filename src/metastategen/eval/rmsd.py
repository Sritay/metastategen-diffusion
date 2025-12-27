from __future__ import annotations

from typing import List, Tuple

import torch


def _safe_svd(cov: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robust SVD with small diagonal jitter to handle degenerate configurations."""
    try:
        return torch.linalg.svd(cov)
    except RuntimeError:
        eye = torch.eye(3, device=cov.device, dtype=cov.dtype)
        if cov.dim() == 3:
            eye = eye.unsqueeze(0).expand(cov.shape[0], -1, -1)
        cov = cov + eps * eye
        return torch.linalg.svd(cov)


def rmsd_kabsch(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute RMSD after Kabsch alignment.

    Args:
        P: [B,N,3] or [N,3]
        Q: [B,N,3] or [N,3]
    Returns:
        rmsd: [B] or scalar tensor
    """
    if P.dim() == 2:
        P = P.unsqueeze(0)
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P {P.shape} vs Q {Q.shape}")

    P = P.to(dtype=torch.float64)
    Q = Q.to(dtype=torch.float64)

    # Center
    P_cent = P - P.mean(dim=1, keepdim=True)
    Q_cent = Q - Q.mean(dim=1, keepdim=True)

    # Covariance
    cov = torch.matmul(P_cent.transpose(1, 2), Q_cent)  # [B,3,3]

    U, _, Vh = _safe_svd(cov, eps=eps)
    # Reflection handling
    det = torch.det(torch.matmul(U, Vh))
    sign = torch.where(det < 0.0, -torch.ones_like(det), torch.ones_like(det))
    D = torch.eye(3, device=U.device, dtype=U.dtype).unsqueeze(0).repeat(U.shape[0], 1, 1)
    D[:, 2, 2] = sign
    R = U @ D @ Vh

    P_rot = torch.matmul(P_cent, R)
    diff = P_rot - Q_cent
    rmsd = torch.sqrt(torch.mean(torch.sum(diff * diff, dim=-1), dim=-1))
    return rmsd.to(dtype=P.dtype)


def greedy_cluster(samples: torch.Tensor, rmsd_thresh: float) -> tuple[List[int], List[int], List[int]]:
    """
    Greedy clustering by RMSD threshold.

    Args:
        samples: [M,N,3] tensor
        rmsd_thresh: threshold for assigning to existing cluster
    Returns:
        assignments: list of cluster id per sample
        medoid_indices: list of medoid indices (first sample in cluster)
        cluster_sizes: list of cluster sizes
    """
    if samples.dim() != 3 or samples.shape[-1] != 3:
        raise ValueError(f"Expected samples [M,N,3], got {samples.shape}")
    if rmsd_thresh <= 0:
        raise ValueError("rmsd_thresh must be positive")

    samples = samples.to(dtype=torch.float64, device="cpu")
    n_samples = samples.shape[0]
    assignments: List[int] = [-1] * n_samples
    medoid_indices: List[int] = []

    for i in range(n_samples):
        if not medoid_indices:
            medoid_indices.append(i)
            assignments[i] = 0
            continue

        medoids = samples[medoid_indices]  # [K,N,3]
        candidate = samples[i].unsqueeze(0).expand(medoids.shape[0], -1, -1)
        rmsds = rmsd_kabsch(candidate, medoids)
        best = int(torch.argmin(rmsds).item())
        if float(rmsds[best].item()) < rmsd_thresh:
            assignments[i] = best
        else:
            assignments[i] = len(medoid_indices)
            medoid_indices.append(i)

    cluster_sizes = [0] * len(medoid_indices)
    for cid in assignments:
        cluster_sizes[cid] += 1

    return assignments, medoid_indices, cluster_sizes
