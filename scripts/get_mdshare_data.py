#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import mdshare

from metastategen.utils import get_logger, set_deterministic

log = get_logger("get_mdshare_data")

ALA2_DIHEDRALS = "alanine-dipeptide-3x250ns-backbone-dihedrals.npz"
ALA2_POSITIONS = "alanine-dipeptide-3x250ns-heavy-atom-positions.npz"

def _load_npz_arrays(npz_path: Path):
    with np.load(npz_path) as fh:
        keys = sorted(fh.keys())
        arrs = [fh[k] for k in keys]
    return keys, arrs

def verify_dihedrals(npz_path: Path) -> None:
    keys, arrs = _load_npz_arrays(npz_path)
    if len(arrs) != 3:
        raise ValueError(f"Expected 3 trajectories in {npz_path}, found {len(arrs)} keys={keys}")
    for i, a in enumerate(arrs):
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f"Dihedrals arr_{i} expected shape [T,2], got {a.shape}")
        if a.dtype not in (np.float32, np.float64):
            raise ValueError(f"Dihedrals arr_{i} expected float dtype, got {a.dtype}")
    log.info(f"Verified dihedrals: {[a.shape for a in arrs]} (keys={keys})")

def verify_positions(npz_path: Path) -> None:
    keys, arrs = _load_npz_arrays(npz_path)
    if len(arrs) != 3:
        raise ValueError(f"Expected 3 trajectories in {npz_path}, found {len(arrs)} keys={keys}")
    for i, a in enumerate(arrs):
        if a.ndim != 2:
            raise ValueError(f"Positions arr_{i} expected shape [T,n_features], got {a.shape}")
        if a.shape[1] % 3 != 0:
            raise ValueError(f"Positions arr_{i} n_features must be divisible by 3, got {a.shape[1]}")
        if a.dtype not in (np.float32, np.float64):
            raise ValueError(f"Positions arr_{i} expected float dtype, got {a.dtype}")
    nfeat = arrs[0].shape[1]
    n_atoms = nfeat // 3
    logit = get_logger("get_mdshare_data")
    if logit:
        logit.info(f"Verified positions: {[a.shape for a in arrs]} (keys={keys}), inferred N={n_atoms} heavy atoms")



from typing import Optional

def select_indices(T: int, stride: int, max_frames: Optional[int], seed: int, random_subset: bool) -> np.ndarray:
    """Duplicate of scripts/preprocess_positions.py logic to ensure alignment."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    base = np.arange(0, T, stride, dtype=np.int64)
    if max_frames is None or max_frames >= base.shape[0]:
        return base

    if random_subset:
        rng = np.random.default_rng(seed)
        pick = rng.choice(base.shape[0], size=max_frames, replace=False)
        pick = np.sort(pick)
        return base[pick]
    else:
        return base[:max_frames]



def fetch_one(filename: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    # mdshare.fetch downloads to working_directory and returns local path
    local = mdshare.fetch(filename, working_directory=str(outdir))
    local_path = Path(local)
    if not local_path.exists():
        raise FileNotFoundError(f"mdshare.fetch returned {local_path} but file does not exist")
    return local_path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="data/raw", help="Download directory")
    ap.add_argument("--seed", type=int, default=0, help="Deterministic seed (for any randomized ops)")
    args = ap.parse_args()

    set_deterministic(args.seed)
    outdir = Path(args.outdir)

    try:
        log.info(f"Downloading mdshare artifacts into: {outdir.resolve()}")
        dih = fetch_one(ALA2_DIHEDRALS, outdir)
        pos = fetch_one(ALA2_POSITIONS, outdir)

        log.info(f"Downloaded: {dih.name} ({dih.stat().st_size/1e6:.1f} MB)")
        log.info(f"Downloaded: {pos.name} ({pos.stat().st_size/1e6:.1f} MB)")

        verify_dihedrals(dih)
        verify_positions(pos)


        log.info("All downloads and verifications succeeded.")
        return 0
    except Exception as e:
        log.error(f"FAILED: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

