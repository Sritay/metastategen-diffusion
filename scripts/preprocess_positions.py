#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import mdshare

from metastategen.utils import get_logger, set_deterministic
from metastategen.eval.ramachandran import _detect_and_convert_to_degrees

log = get_logger("preprocess_positions")

ALA2_PDB = "alanine-dipeptide-nowater.pdb"

def load_mdshare_npz(npz_path: Path) -> List[np.ndarray]:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as fh:
        keys = sorted(fh.keys())
        arrs = [fh[k] for k in keys]
    if len(arrs) != 3:
        raise ValueError(f"Expected 3 arrays (arr_0..arr_2) in {npz_path}, got keys={keys}")
    return arrs

def parse_heavy_atom_types_from_pdb(pdb_path: Path, expected_n: int) -> np.ndarray:
    """
    Parse element symbols from PDB ATOM/HETATM records and keep heavy atoms only (exclude H).
    Map elements -> integer types:
      C=0, N=1, O=2, S=3, else=4
    """
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    elems: List[str] = []
    with pdb_path.open("r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            # PDB element is typically columns 77-78 (1-indexed), i.e., line[76:78]
            elem = line[76:78].strip()
            if not elem:
                # fallback: atom name field (cols 13-16) may start with element
                aname = line[12:16].strip()
                elem = "".join([c for c in aname if c.isalpha()])[:2].strip().title()
            elem = elem.title()
            if elem.startswith("H"):
                continue
            elems.append(elem)

    if len(elems) != expected_n:
        raise ValueError(
            f"Heavy atom count from PDB ({len(elems)}) does not match expected N ({expected_n}). "
            f"PDB={pdb_path}"
        )

    mapping = {"C": 0, "N": 1, "O": 2, "S": 3}
    types = np.array([mapping.get(e, 4) for e in elems], dtype=np.int64)
    return types

def recenter_positions(x: np.ndarray) -> np.ndarray:
    """Subtract per-frame centroid. x: [T,N,3]."""
    centroid = np.mean(x, axis=1, keepdims=True)
    return (x - centroid).astype(np.float32)



def select_indices(T: int, stride: int, max_frames: Optional[int], seed: int, random_subset: bool) -> np.ndarray:
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

def save_shards(
    outdir: Path,
    positions: torch.Tensor,     # [M,N,3]
    atom_types: torch.Tensor,    # [N]
    phi_psi: torch.Tensor,       # [M,2]
    traj_id: torch.Tensor,       # [M]
    frame_id: torch.Tensor,      # [M]
    shard_size: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    nshards = (positions.shape[0] + shard_size - 1) // shard_size
    for s in range(nshards):
        lo = s * shard_size
        hi = min((s + 1) * shard_size, positions.shape[0])
        shard = {
            "positions": positions[lo:hi].contiguous(),
            "atom_types": atom_types.contiguous(),
            "phi_psi": phi_psi[lo:hi].contiguous(),
            "traj_id": traj_id[lo:hi].contiguous(),
            "frame_id": frame_id[lo:hi].contiguous(),
        }
        path = outdir / f"shard_{s:05d}.pt"
        torch.save(shard, path)
    log.info(f"Wrote {nshards} shard(s) to {outdir} (shard_size={shard_size})")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions-npz", type=str, default="data/raw/alanine-dipeptide-3x250ns-heavy-atom-positions.npz")
    ap.add_argument("--dihedrals-npz", type=str, default="data/raw/alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
    ap.add_argument("--raw-dir", type=str, default="data/raw", help="Where to look for (and optionally fetch) the PDB")
    ap.add_argument("--outdir", type=str, default="data/processed/ala2", help="Output base directory")
    ap.add_argument("--stride", type=int, default=10, help="Stride for subsampling within each trajectory")
    ap.add_argument("--max-frames-per-traj", type=int, default=20000, help="Cap frames per trajectory after striding")
    ap.add_argument("--random-subset", action="store_true", help="If set, choose max-frames-per-traj randomly (deterministic via seed)")
    ap.add_argument("--shard-size", type=int, default=5000, help="Frames per .pt shard")
    ap.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    args = ap.parse_args()

    set_deterministic(args.seed)

    positions_npz = Path(args.positions_npz)
    dihedrals_npz = Path(args.dihedrals_npz)
    raw_dir = Path(args.raw_dir)
    outdir = Path(args.outdir)
    shard_dir = outdir / "shards"
    meta_path = outdir / "meta.pt"

    try:
        # Load arrays (3 trajectories each)
        pos_trajs = load_mdshare_npz(positions_npz)
        dih_trajs = load_mdshare_npz(dihedrals_npz)

        # Validate alignment and infer N
        for i in range(3):
            if pos_trajs[i].shape[0] != dih_trajs[i].shape[0]:
                raise ValueError(
                    f"Length mismatch traj {i}: positions T={pos_trajs[i].shape[0]} vs dihedrals T={dih_trajs[i].shape[0]}"
                )
            if pos_trajs[i].shape[1] % 3 != 0:
                raise ValueError(f"positions traj {i} n_features not divisible by 3: {pos_trajs[i].shape[1]}")

        n_atoms = pos_trajs[0].shape[1] // 3
        log.info(f"Inferred heavy atoms N={n_atoms}")

        # Fetch / parse PDB for atom types
        pdb_path = raw_dir / ALA2_PDB
        if not pdb_path.exists():
            raw_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"PDB not found at {pdb_path}; fetching via mdshare...")
            mdshare.fetch(ALA2_PDB, working_directory=str(raw_dir))
        atom_types_np = parse_heavy_atom_types_from_pdb(pdb_path, expected_n=n_atoms)
        atom_types = torch.from_numpy(atom_types_np)

        # Collect selected frames across the 3 trajectories
        all_pos: List[np.ndarray] = []
        all_phi_psi: List[np.ndarray] = []
        all_traj: List[np.ndarray] = []
        all_frame: List[np.ndarray] = []

        for tid in range(3):
            X = pos_trajs[tid].astype(np.float32, copy=False)   # [T, 3N]
            T = X.shape[0]
            X = X.reshape(T, n_atoms, 3)                        # [T,N,3]
            X = recenter_positions(X)

            phi_psi = dih_trajs[tid].astype(np.float32, copy=False)  # [T,2]
            phi_psi_deg = _detect_and_convert_to_degrees(phi_psi)    # degrees, wrapped

            idx = select_indices(
                T=T,
                stride=args.stride,
                max_frames=args.max_frames_per_traj,
                seed=args.seed + 1000 * tid,
                random_subset=args.random_subset,
            )
            log.info(f"traj {tid}: T={T}, stride={args.stride}, selected={idx.shape[0]} frames")

            all_pos.append(X[idx])
            all_phi_psi.append(phi_psi_deg[idx])
            all_traj.append(np.full(idx.shape[0], tid, dtype=np.int64))
            all_frame.append(idx.astype(np.int64))

        Xcat = np.concatenate(all_pos, axis=0)          # [M,N,3]
        PPcat = np.concatenate(all_phi_psi, axis=0)    # [M,2]
        Tcat = np.concatenate(all_traj, axis=0)        # [M]
        Fcat = np.concatenate(all_frame, axis=0)       # [M]

        # Deterministic global shuffle (optional later); Day-1 keep stable ordering:
        # (traj0 then traj1 then traj2), stable indices.
        log.info(f"Total frames after subsampling: M={Xcat.shape[0]}")

        # Convert to torch
        positions = torch.from_numpy(Xcat)        # float32
        phi_psi = torch.from_numpy(PPcat)         # float32 degrees
        traj_id = torch.from_numpy(Tcat)
        frame_id = torch.from_numpy(Fcat)

        # Save shards
        save_shards(
            outdir=shard_dir,
            positions=positions,
            atom_types=atom_types,
            phi_psi=phi_psi,
            traj_id=traj_id,
            frame_id=frame_id,
            shard_size=args.shard_size,
        )

        # Save meta
        meta = {
            "n_atoms": n_atoms,
            "atom_types_mapping": {"C": 0, "N": 1, "O": 2, "S": 3, "other": 4},
            "positions_npz": str(positions_npz),
            "dihedrals_npz": str(dihedrals_npz),
            "pdb_path": str(pdb_path),
            "stride": args.stride,
            "max_frames_per_traj": args.max_frames_per_traj,
            "random_subset": bool(args.random_subset),
            "seed": args.seed,
            "total_frames": int(Xcat.shape[0]),
            "units": {"positions": "unknown (mdshare heavy-atom positions)", "phi_psi": "degrees"},
        }
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save(meta, meta_path)
        log.info(f"Wrote metadata: {meta_path}")

        return 0
    except Exception as e:
        log.error(f"FAILED: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

