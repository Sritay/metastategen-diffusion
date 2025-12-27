import argparse
import glob
import math
from pathlib import Path

import torch
import yaml

from metastategen.utils import get_logger, set_deterministic
from metastategen.models.diffusion import center
from metastategen.models.ensemble import (
    Ensemble,
    build_diffusion_from_cfg,
    center_ensemble,
    load_ensemble,
    per_sample_uncertainty_from_var,
)

log = get_logger("sample_ensemble")


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


def _member_dirs(run_root: Path, n_members: int) -> list[Path]:
    return [run_root / "members" / f"m{i:03d}" for i in range(n_members)]


def _load_atom_types(cfg: dict, device: torch.device) -> tuple[int, torch.Tensor]:
    meta = torch.load(cfg["data"]["meta_path"])
    n_atoms = int(meta["n_atoms"])
    shard_path = sorted(glob.glob(f"{cfg['data']['data_dir']}/*.pt"))[0]
    atom_types = torch.load(shard_path)["atom_types"].to(device)
    return n_atoms, atom_types


def _sample_single(
    model: torch.nn.Module,
    diffusion,
    atom_types: torch.Tensor,
    n_atoms: int,
    n_samples: int,
    batch_size: int,
    steps: int,
    eta: float,
) -> torch.Tensor:
    all_samples = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in range(n_batches):
        curr_bs = min(batch_size, n_samples - len(all_samples) * batch_size)
        batch_types = atom_types.unsqueeze(0).expand(curr_bs, -1)
        shape = (curr_bs, n_atoms, 3)
        if eta == 0.0 and steps < diffusion.cfg.T:
            x0 = diffusion.ddim_sample_loop(model, shape, batch_types, steps=steps, eta=eta)
        else:
            x0 = diffusion.p_sample_loop(model, shape, batch_types, steps=None)
        all_samples.append(x0.cpu())
    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def _consensus_ddpm(
    ensemble: Ensemble,
    diffusion,
    shape: tuple[int, ...],
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = diffusion.betas.device
    B = shape[0]
    xt = torch.randn(shape, device=device)
    if diffusion.cfg.recenter_every_step:
        xt = center(xt)

    unc_accum = torch.zeros(B, device=device)
    steps = 0

    for i in reversed(range(1, diffusion.cfg.T + 1)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        eps_stack = ensemble.predict_eps(xt, h, t)
        if diffusion.cfg.recenter_every_step:
            eps_stack = center_ensemble(eps_stack)

        mean_eps = eps_stack.mean(dim=0)
        var_eps = eps_stack.var(dim=0, unbiased=False)
        unc_accum += per_sample_uncertainty_from_var(var_eps)
        steps += 1

        if diffusion.cfg.recenter_every_step:
            mean_eps = center(mean_eps)

        idx = i - 1
        beta = diffusion.betas[idx]
        alpha = diffusion.alphas[idx]
        alpha_bar = diffusion.alphas_cumprod[idx]

        mean = (1 / torch.sqrt(alpha)) * (xt - (beta / torch.sqrt(1 - alpha_bar)) * mean_eps)
        if i > 1:
            sigma = torch.sqrt(diffusion.posterior_variance[idx])
            noise = torch.randn_like(xt)
            xt = mean + sigma * noise
        else:
            xt = mean
        if diffusion.cfg.recenter_every_step:
            xt = center(xt)

    return xt, unc_accum / max(1, steps)


@torch.no_grad()
def _consensus_ddim(
    ensemble: Ensemble,
    diffusion,
    shape: tuple[int, ...],
    h: torch.Tensor,
    steps: int,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = diffusion.betas.device
    B = shape[0]
    xt = torch.randn(shape, device=device)
    if diffusion.cfg.recenter_every_step:
        xt = center(xt)

    unc_accum = torch.zeros(B, device=device)
    n_steps = 0
    times = torch.linspace(diffusion.cfg.T, 1, steps).long().to(device)

    for i, t_val in enumerate(times):
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        prev_t_val = times[i + 1] if i < len(times) - 1 else torch.tensor(0, device=device)

        eps_stack = ensemble.predict_eps(xt, h, t)
        if diffusion.cfg.recenter_every_step:
            eps_stack = center_ensemble(eps_stack)

        mean_eps = eps_stack.mean(dim=0)
        var_eps = eps_stack.var(dim=0, unbiased=False)
        unc_accum += per_sample_uncertainty_from_var(var_eps)
        n_steps += 1

        if diffusion.cfg.recenter_every_step:
            mean_eps = center(mean_eps)

        idx = t_val - 1
        prev_idx = prev_t_val - 1

        alpha_bar = diffusion.alphas_cumprod[idx]
        alpha_bar_prev = diffusion.alphas_cumprod[prev_idx] if prev_t_val > 0 else torch.tensor(1.0, device=device)

        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
        pred_x0 = (xt - torch.sqrt(1 - alpha_bar) * mean_eps) / torch.sqrt(alpha_bar)
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * mean_eps
        noise = torch.randn_like(xt) if eta > 0 else 0.0

        xt = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
        if diffusion.cfg.recenter_every_step:
            xt = center(xt)

    return xt, unc_accum / max(1, n_steps)


@torch.no_grad()
def _score_uncertainty(
    ensemble: Ensemble,
    diffusion,
    samples: torch.Tensor,
    atom_types: torch.Tensor,
    score_steps: int,
) -> torch.Tensor:
    if score_steps <= 0:
        raise ValueError("score_steps must be positive")
    device = diffusion.betas.device
    samples = samples.to(device)
    B = samples.shape[0]
    h = atom_types.unsqueeze(0).expand(B, -1)

    unc_accum = torch.zeros(B, device=device)
    for _ in range(score_steps):
        t = torch.randint(1, diffusion.cfg.T + 1, (B,), device=device)
        x0 = samples
        if diffusion.cfg.recenter_every_step:
            x0 = center(x0)
        noise = torch.randn_like(x0)
        if diffusion.cfg.recenter_every_step:
            noise = center(noise)
        xt = diffusion.q_sample(x0, t, noise)
        if diffusion.cfg.recenter_every_step:
            xt = center(xt)

        eps_stack = ensemble.predict_eps(xt, h, t)
        if diffusion.cfg.recenter_every_step:
            eps_stack = center_ensemble(eps_stack)
        var_eps = eps_stack.var(dim=0, unbiased=False)
        unc_accum += per_sample_uncertainty_from_var(var_eps)

    return unc_accum / float(score_steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/ala2_ensemble.yaml")
    ap.add_argument("--ensemble-dir", type=str, default=None)
    ap.add_argument("--mode", type=str, choices=["naive", "consensus", "score"], default="consensus")
    ap.add_argument("--samples", type=str, default=None, help="Existing samples.pt to score (mode=score)")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--per-member", action="store_true", help="Interpret n_samples as per-member (naive mode)")
    ap.add_argument("--score-steps", type=int, default=1)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sample_cfg = cfg.get("sample", {})
    seed = int(sample_cfg.get("seed", cfg.get("train", {}).get("seed", 0) + 100))
    set_deterministic(seed)

    run_root = Path(args.ensemble_dir) if args.ensemble_dir else _resolve_run_root(cfg)
    seeds = _resolve_seeds(cfg)
    member_dirs = _member_dirs(run_root, len(seeds))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_atoms, atom_types = _load_atom_types(cfg, device=device)

    diffusion = build_diffusion_from_cfg(cfg).to(device)
    ensemble = load_ensemble(member_dirs, cfg, n_atom_types=int(atom_types.max().item()) + 1, device=device)

    out_dir = Path(args.outdir) if args.outdir else run_root / "samples" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "score":
        if args.samples is None:
            raise ValueError("--samples is required for mode=score")
        samples = torch.load(args.samples)
        uncertainty = _score_uncertainty(ensemble, diffusion, samples, atom_types, args.score_steps)
        torch.save(uncertainty.cpu(), out_dir / "uncertainty.pt")
        log.info(f"Wrote: {out_dir / 'uncertainty.pt'}")
        return 0

    n_samples = int(sample_cfg.get("n_samples", 1000))
    batch_size = int(sample_cfg.get("batch_size", 256))
    steps = int(sample_cfg.get("steps", diffusion.cfg.T))
    eta = float(sample_cfg.get("eta", 0.0))

    if args.mode == "naive":
        samples_list = []
        member_ids = []
        per_member = n_samples if args.per_member else math.ceil(n_samples / len(seeds))
        for m_idx, model in enumerate(ensemble.members):
            log.info(f"[m{m_idx:03d}] sampling {per_member} samples")
            member_samples = _sample_single(
                model,
                diffusion,
                atom_types,
                n_atoms,
                per_member,
                batch_size,
                steps,
                eta,
            )
            samples_list.append(member_samples)
            member_ids.append(torch.full((member_samples.shape[0],), m_idx, dtype=torch.long))

        samples = torch.cat(samples_list, dim=0)
        member_ids = torch.cat(member_ids, dim=0)
        if not args.per_member and samples.shape[0] > n_samples:
            samples = samples[:n_samples]
            member_ids = member_ids[:n_samples]

        uncertainty = _score_uncertainty(ensemble, diffusion, samples, atom_types, args.score_steps)
        torch.save(member_ids, out_dir / "member_ids.pt")
        torch.save(samples, out_dir / "samples.pt")
        torch.save(uncertainty.cpu(), out_dir / "uncertainty.pt")
        log.info(f"Wrote: {out_dir / 'samples.pt'}")
        log.info(f"Wrote: {out_dir / 'uncertainty.pt'}")
        log.info(f"Wrote: {out_dir / 'member_ids.pt'}")
        return 0

    if args.mode == "consensus":
        samples_list = []
        unc_list = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        for i in range(n_batches):
            curr_bs = min(batch_size, n_samples - len(samples_list) * batch_size)
            batch_types = atom_types.unsqueeze(0).expand(curr_bs, -1)
            shape = (curr_bs, n_atoms, 3)
            if eta == 0.0 and steps < diffusion.cfg.T:
                x0, unc = _consensus_ddim(ensemble, diffusion, shape, batch_types, steps=steps, eta=eta)
            else:
                if eta != 0.0 and steps < diffusion.cfg.T:
                    log.warning("DDIM steps < T with eta>0 is not supported; using full DDPM steps.")
                x0, unc = _consensus_ddpm(ensemble, diffusion, shape, batch_types)
            samples_list.append(x0.cpu())
            unc_list.append(unc.cpu())
            log.info(f"Batch {i+1}/{n_batches} done.")

        samples = torch.cat(samples_list, dim=0)
        uncertainty = torch.cat(unc_list, dim=0)
        torch.save(samples, out_dir / "samples.pt")
        torch.save(uncertainty, out_dir / "uncertainty.pt")
        log.info(f"Wrote: {out_dir / 'samples.pt'}")
        log.info(f"Wrote: {out_dir / 'uncertainty.pt'}")
        return 0

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
