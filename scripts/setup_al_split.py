from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
import yaml

from metastategen.utils import get_logger, set_deterministic

log = get_logger("setup_al_split")


def _load_shards(shard_dir: Path) -> dict[str, torch.Tensor]:
    shard_paths = sorted(glob.glob(str(shard_dir / "*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {shard_dir}")

    positions = []
    traj_ids = []
    frame_ids = []
    phi_psi = []
    atom_types = None
    has_phi = True

    for p in shard_paths:
        data = torch.load(p, map_location="cpu")
        if atom_types is None:
            atom_types = data["atom_types"]
        elif not torch.equal(atom_types, data["atom_types"]):
            raise ValueError(f"atom_types mismatch in shard {p}")

        positions.append(data["positions"])
        traj_ids.append(data["traj_id"])
        frame_ids.append(data.get("frame_id"))

        if "phi_psi" in data:
            phi_psi.append(data["phi_psi"])
        else:
            has_phi = False

    out = {
        "positions": torch.cat(positions, dim=0),
        "atom_types": atom_types,
        "traj_id": torch.cat(traj_ids, dim=0),
    }

    if all(x is not None for x in frame_ids):
        out["frame_id"] = torch.cat(frame_ids, dim=0)
    if has_phi and phi_psi:
        out["phi_psi"] = torch.cat(phi_psi, dim=0)

    return out


def _slice_data(data: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {"positions": data["positions"][indices], "atom_types": data["atom_types"]}
    for key in ("traj_id", "frame_id", "phi_psi", "source_index"):
        if key in data:
            out[key] = data[key][indices]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None, help="Processed shard directory")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory for AL split")
    ap.add_argument("--seed-size", type=int, default=None)
    ap.add_argument("--val-size", type=int, default=None)
    ap.add_argument("--seed-traj", type=int, default=None)
    ap.add_argument("--val-traj", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-random", action="store_true", help="Randomly sample val indices (default)")
    ap.add_argument("--val-sequential", action="store_true", help="Use first val_size frames in val traj")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    al_cfg = cfg.get("active_learning", {})
    data_cfg = cfg.get("data", {})

    data_dir = args.data_dir or al_cfg.get("oracle_pool_source") or data_cfg.get("data_dir")
    if data_dir is None:
        raise ValueError("data_dir must be provided via --data-dir or config")

    out_dir = Path(args.out_dir or al_cfg.get("split_out_dir") or "data/processed/ala2")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_size = int(args.seed_size or al_cfg.get("initial_seed_size", 5000))
    val_size = int(args.val_size or al_cfg.get("val_size", 2000))
    seed_traj = int(args.seed_traj if args.seed_traj is not None else al_cfg.get("seed_traj", 0))
    val_traj = int(args.val_traj if args.val_traj is not None else al_cfg.get("val_traj", 2))

    if seed_size <= 0:
        raise ValueError("seed_size must be positive")
    if val_size < 0:
        raise ValueError("val_size must be non-negative")

    set_deterministic(args.seed)

    data_dir = Path(data_dir)
    log.info(f"Loading shards from {data_dir}")
    full = _load_shards(data_dir)
    n_total = int(full["positions"].shape[0])
    full["source_index"] = torch.arange(n_total, dtype=torch.long)

    traj_id = full["traj_id"]
    seed_idx_all = torch.where(traj_id == seed_traj)[0]
    val_idx_all = torch.where(traj_id == val_traj)[0]

    if seed_size > seed_idx_all.shape[0]:
        raise ValueError(f"seed_size {seed_size} > available traj {seed_traj} frames {seed_idx_all.shape[0]}")
    if val_size > val_idx_all.shape[0]:
        raise ValueError(f"val_size {val_size} > available traj {val_traj} frames {val_idx_all.shape[0]}")

    seed_idx = seed_idx_all[:seed_size]

    if val_size == 0:
        val_idx = torch.empty(0, dtype=torch.long)
    elif args.val_sequential and not args.val_random:
        val_idx = val_idx_all[:val_size]
    else:
        perm = torch.randperm(val_idx_all.shape[0])
        val_idx = val_idx_all[perm[:val_size]]

    mask = torch.ones(n_total, dtype=torch.bool)
    mask[seed_idx] = False
    if val_idx.numel() > 0:
        mask[val_idx] = False
    pool_idx = torch.where(mask)[0]

    seed_data = _slice_data(full, seed_idx)
    pool_data = _slice_data(full, pool_idx)
    val_data = _slice_data(full, val_idx)

    seed_path = out_dir / "al_seed.pt"
    pool_path = out_dir / "al_pool_ref.pt"
    val_path = out_dir / "al_val.pt"

    torch.save(seed_data, seed_path)
    torch.save(pool_data, pool_path)
    torch.save(val_data, val_path)

    log.info(
        "Wrote AL split: seed=%s pool=%s val=%s",
        seed_path,
        pool_path,
        val_path,
    )
    log.info(
        "Counts: seed=%d pool=%d val=%d total=%d",
        int(seed_data["positions"].shape[0]),
        int(pool_data["positions"].shape[0]),
        int(val_data["positions"].shape[0]),
        n_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
