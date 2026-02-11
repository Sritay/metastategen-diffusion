from typing import List, Dict, Union
import csv
import time
from pathlib import Path
import torch
from metastategen.models.features import compute_chiral_volume_signal

def _resolve_run_root(cfg: dict) -> Path:
    al_cfg = cfg.get("active_learning", {})
    exp_id = al_cfg.get("exp_id", cfg.get("train", {}).get("exp_id", "experiment"))
    out_dir = al_cfg.get("out_dir", cfg.get("train", {}).get("out_dir", f"runs/{exp_id}"))
    return Path(out_dir)


def _build_dataloader(dataset, batch_size: int, num_workers: int, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
    )


def _save_member_logs(member_dir: Path, logs: List[Dict]) -> None:
    if not logs:
        return
    keys = sorted({k for row in logs for k in row.keys()})
    out_path = member_dir / "train_log.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(logs)


def _save_checkpoint(member_dir: Path, state: dict, iter_idx: int, cfg: dict) -> None:
    ckpt_dir = member_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle single model vs member training state structure differences if any
    # active_learning: iter_idx is passed.
    # train.py: might use epoch instead.
    
    fname = f"iter_{iter_idx:02d}.pt" if iter_idx is not None else f"ckpt_{state['epoch']:04d}.pt"
    
    ckpt = {
        "epoch": state["epoch"],
        "model": state["model"].state_dict(),
        "opt": state["opt"].state_dict(),
        "config": cfg,
    }
    if iter_idx is not None:
        ckpt["iter"] = iter_idx
        
    torch.save(ckpt, ckpt_dir / fname)


def _train_member(
    state: dict,
    diffusion,
    dataloader,
    epochs: int,
    grad_clip: float,
    rot_aug: bool,
    iter_idx: int = None,
    chirality_config: list = None,
) -> None:
    model = state["model"]
    opt = state["opt"]
    device = state["device"]
    # Chirality config might be in state or passed explicitly
    chirality_config = state.get("chirality_config", chirality_config)

    start = time.time()
    for _ in range(epochs):
        state["epoch"] += 1
        model.train()
        losses = []
        for batch in dataloader:
            x = batch["x"].to(device)
            a = batch["a"].to(device)
            bsz = x.shape[0]
            
            # Compute Chiral Conditioning Signal (from CLEAN x)
            # x is scaled by scale_factor, so we must tell the function to unscale it
            scale_factor = diffusion.cfg.scale_factor
            condition = None
            if hasattr(model, "cfg") and getattr(model.cfg, "use_chiral_features", False):
                 condition = compute_chiral_volume_signal(x, scale_factor=scale_factor, chirality_config=chirality_config)
            
            t = torch.randint(1, diffusion.cfg.T + 1, (bsz,), device=device)
            loss, _ = diffusion.training_loss(model, x, a, t, rot_aug=rot_aug, condition=condition)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(loss.item())

        train_loss = float(sum(losses) / max(1, len(losses)))
        log_entry = {
                "epoch": state["epoch"],
                "train_loss": train_loss,
                "elapsed_s": time.time() - start,
            }
        if iter_idx is not None:
            log_entry["iter"] = iter_idx
            
        state["logs"].append(log_entry)
