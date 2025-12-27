from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

from metastategen.models.diffusion import DiffusionConfig, GaussianDiffusion, center
from metastategen.models.egnn import EGNN, EGNNConfig
from metastategen.utils import get_logger

log = get_logger("ensemble")


def build_model_from_cfg(cfg: dict, n_atom_types: int) -> EGNN:
    model_cfg = cfg.get("model", {})
    egnn_cfg = EGNNConfig(
        n_layers=int(model_cfg.get("n_layers", 4)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        edge_mlp_layers=int(model_cfg.get("edge_mlp_layers", 2)),
        node_mlp_layers=int(model_cfg.get("node_mlp_layers", 2)),
        coord_mlp_layers=int(model_cfg.get("coord_mlp_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model = EGNN(
        n_atom_types=n_atom_types,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        time_emb_dim=int(model_cfg.get("time_emb_dim", 128)),
        cfg=egnn_cfg,
    )
    return model


def build_diffusion_from_cfg(cfg: dict) -> GaussianDiffusion:
    diff_cfg = cfg.get("diffusion", {})
    dcfg = DiffusionConfig(
        T=int(diff_cfg.get("T", 1000)),
        beta_start=float(diff_cfg.get("beta_start", 1e-4)),
        beta_end=float(diff_cfg.get("beta_end", 2e-2)),
        schedule=str(diff_cfg.get("schedule", "linear")),
        recenter_every_step=bool(diff_cfg.get("recenter_every_step", True)),
        ddim_eta=float(diff_cfg.get("ddim_eta", 0.0)),
    )
    return GaussianDiffusion(dcfg)


def _resolve_checkpoint(member_dir: Path, prefer: str = "best.pt") -> Path:
    ckpt_dir = member_dir / "checkpoints"
    preferred = ckpt_dir / prefer
    if preferred.exists():
        return preferred
    for alt in ("final.pt", "last.pt"):
        alt_path = ckpt_dir / alt
        if alt_path.exists():
            log.warning(f"Checkpoint {preferred} not found; using {alt_path}.")
            return alt_path
    raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}.")


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: Optional[torch.device] = None) -> nn.Module:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model


def center_ensemble(eps_stack: torch.Tensor) -> torch.Tensor:
    """Center stacked tensors shaped [M,B,N,3] across atom dimension."""
    if eps_stack.dim() != 4:
        raise ValueError(f"Expected eps_stack [M,B,N,3], got {eps_stack.shape}")
    m, b, n, c = eps_stack.shape
    flat = eps_stack.reshape(m * b, n, c)
    flat = center(flat)
    return flat.reshape(m, b, n, c)


def per_sample_uncertainty_from_var(var: torch.Tensor) -> torch.Tensor:
    """Convert per-coordinate variance [B,N,3] into scalar uncertainty per sample."""
    if var.dim() != 3:
        raise ValueError(f"Expected var [B,N,3], got {var.shape}")
    return var.mean(dim=(1, 2))


class Ensemble(nn.Module):
    def __init__(self, members: Iterable[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(list(members))

    def predict_eps(
        self, x: torch.Tensor, h: torch.Tensor, t: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        preds = []
        for m in self.members:
            preds.append(m(x, h, t, edge_attr=edge_attr))
        return torch.stack(preds, dim=0)

    def mean_and_var(
        self, x: torch.Tensor, h: torch.Tensor, t: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = self.predict_eps(x, h, t, edge_attr=edge_attr)
        mean = eps.mean(dim=0)
        var = eps.var(dim=0, unbiased=False)
        return mean, var


def load_ensemble(
    member_dirs: Iterable[Path],
    cfg: dict,
    n_atom_types: int,
    device: torch.device,
    prefer_ckpt: str = "best.pt",
) -> Ensemble:
    members = []
    for member_dir in member_dirs:
        ckpt_path = _resolve_checkpoint(Path(member_dir), prefer=prefer_ckpt)
        model = build_model_from_cfg(cfg, n_atom_types).to(device)
        load_checkpoint(model, ckpt_path, device=device)
        model.eval()
        members.append(model)
    return Ensemble(members)
