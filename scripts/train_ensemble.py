import argparse
import time
from copy import deepcopy
from pathlib import Path

import torch
import yaml
import pandas as pd

from metastategen.utils import get_logger, set_deterministic
from metastategen.data import Ala2Dataset
from metastategen.models.ensemble import build_model_from_cfg, build_diffusion_from_cfg

log = get_logger("train_ensemble")


def _resolve_run_root(cfg: dict) -> Path:
    ens_cfg = cfg.get("ensemble", {})
    exp_id = ens_cfg.get("exp_id", "ensemble")
    out_dir = ens_cfg.get("out_dir", f"runs/{exp_id}")
    return Path(out_dir)


def _resolve_seeds(cfg: dict) -> list[int]:
    ens_cfg = cfg.get("ensemble", {})
    seeds = ens_cfg.get("seeds")
    if seeds is not None:
        return [int(s) for s in seeds]
    members = int(ens_cfg.get("members", 1))
    base_seed = int(ens_cfg.get("base_seed", cfg.get("train", {}).get("seed", 0)))
    return [base_seed + i for i in range(members)]


def _build_dataloaders(cfg: dict, seed: int):
    data_cfg = cfg["data"]
    subsample = int(data_cfg.get("frame_subsample", 1))

    ds_train = Ala2Dataset(
        data_cfg["data_dir"],
        trajs=data_cfg.get("train_trajs"),
        subsample=subsample,
    )
    g = torch.Generator()
    g.manual_seed(seed)
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        generator=g,
    )

    dl_val = None
    if data_cfg.get("val_trajs") is not None:
        ds_val = Ala2Dataset(
            data_cfg["data_dir"],
            trajs=data_cfg.get("val_trajs"),
            subsample=subsample,
        )
        dl_val = torch.utils.data.DataLoader(
            ds_val,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
        )

    return ds_train, dl_train, dl_val


def _evaluate(model, diffusion, dl_val, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dl_val:
            x = batch["x"].to(device)
            a = batch["a"].to(device)
            B = x.shape[0]
            t = torch.randint(1, diffusion.cfg.T + 1, (B,), device=device)
            loss, _ = diffusion.training_loss(model, x, a, t, rot_aug=False)
            losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


def train_member(cfg: dict, member_idx: int, seed: int, run_root: Path) -> None:
    set_deterministic(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    member_dir = run_root / "members" / f"m{member_idx:03d}"
    ckpt_dir = member_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg_member = deepcopy(cfg)
    cfg_member.setdefault("ensemble", {})
    cfg_member["ensemble"]["member_idx"] = member_idx
    cfg_member["ensemble"]["member_seed"] = seed
    cfg_member.setdefault("train", {})
    cfg_member["train"]["seed"] = seed

    with (member_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg_member, f, sort_keys=False)

    ds_train, dl_train, dl_val = _build_dataloaders(cfg, seed=seed)
    n_atom_types = int(ds_train.atom_types.max().item()) + 1

    model = build_model_from_cfg(cfg, n_atom_types=n_atom_types).to(device)
    diffusion = build_diffusion_from_cfg(cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    save_every = int(cfg["train"].get("save_every", 0))
    val_every = int(cfg["train"].get("val_every", 0))
    rot_aug = bool(cfg["train"].get("rot_aug", True))
    max_seconds = cfg["train"].get("max_seconds")

    epochs = int(cfg["train"]["epochs"])
    log.info(f"[m{member_idx:03d}] seed={seed} device={device} epochs={epochs}")

    logs = []
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_losses = []
        for batch in dl_train:
            x = batch["x"].to(device)
            a = batch["a"].to(device)
            B = x.shape[0]
            t = torch.randint(1, diffusion.cfg.T + 1, (B,), device=device)
            loss, _ = diffusion.training_loss(model, x, a, t, rot_aug=rot_aug)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_losses.append(loss.item())

        train_loss = float(sum(ep_losses) / max(1, len(ep_losses)))
        val_loss = None
        if dl_val is not None and val_every > 0 and epoch % val_every == 0:
            val_loss = _evaluate(model, diffusion, dl_val, device)

        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_s": time.time() - start_time,
            }
        )
        log.info(f"[m{member_idx:03d}] epoch={epoch} train_loss={train_loss:.6f}")

        metric = val_loss if val_loss is not None else train_loss
        if metric < best_loss:
            best_loss = metric
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "best_metric": best_loss,
                    "config": cfg_member,
                },
                ckpt_dir / "best.pt",
            )
            log.info(f"[m{member_idx:03d}] new best checkpoint (metric={best_loss:.6f})")

        if save_every > 0 and epoch % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "config": cfg_member,
                },
                ckpt_dir / f"ckpt_{epoch:04d}.pt",
            )

        if max_seconds is not None and (time.time() - start_time) > float(max_seconds):
            log.info(f"[m{member_idx:03d}] stopping early at epoch {epoch} due to max_seconds")
            break

    pd.DataFrame(logs).to_csv(member_dir / "train_log.csv", index=False)
    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    log.info(f"[m{member_idx:03d}] training complete; best_metric={best_loss:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/ala2_ensemble.yaml")
    ap.add_argument("--member-idx", type=int, default=None, help="Train a single member index")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_root = _resolve_run_root(cfg)
    run_root.mkdir(parents=True, exist_ok=True)

    seeds = _resolve_seeds(cfg)
    with (run_root / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    if args.member_idx is not None:
        if args.member_idx < 0 or args.member_idx >= len(seeds):
            raise ValueError(f"member_idx {args.member_idx} out of range (0..{len(seeds)-1})")
        train_member(cfg, args.member_idx, seeds[args.member_idx], run_root)
    else:
        for i, seed in enumerate(seeds):
            train_member(cfg, i, seed, run_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
