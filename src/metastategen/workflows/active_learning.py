from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from typing import Union, Optional, Tuple, List

from metastategen.active_learning import select_acquisition
try:
    from metastategen.data import ALDataManager, load_al_data, load_training_data
except ImportError:  # Fallback
    from metastategen.data.manager import ALDataManager, load_al_data, load_training_data
from metastategen.eval.coverage import kl_from_phi_psi
from metastategen.eval.rmsd import greedy_cluster, rmsd_kabsch
from metastategen.models.diffusion import center, constrain_bonds, constrain_chirality
from metastategen.models.ensemble import (
    Ensemble,
    build_diffusion_from_cfg,
    build_model_from_cfg,
    center_ensemble,
    per_sample_uncertainty_from_var,
)
from metastategen.oracles import DatasetOracle
from metastategen.utils import get_logger, set_deterministic
from metastategen.models.features import compute_chiral_volume_signal
from metastategen.utils.geometry import compute_dihedrals, rad2deg
try:
    from metastategen.utils.pdb import get_ala2_heavy_atom_indices
except ImportError:
    get_ala2_heavy_atom_indices = None
from metastategen.data.topology import MoleculeTopology

log = get_logger("run_al_loop")


from metastategen.workflows.common import (
    _resolve_run_root,
    _build_dataloader,
    _save_member_logs,
    _save_checkpoint,
    _train_member,
)


def _resolve_seeds(cfg: dict) -> list[int]:
    ens_cfg = cfg.get("ensemble", {})
    seeds = ens_cfg.get("seeds")
    if seeds is not None:
        return [int(s) for s in seeds]
    members = int(ens_cfg.get("members", 1))
    base_seed = int(ens_cfg.get("base_seed", cfg.get("train", {}).get("seed", 0)))
    return [base_seed + i for i in range(members)]














@torch.no_grad()
def _consensus_ddpm(ensemble: Ensemble, diffusion, shape: tuple[int, ...], h: torch.Tensor, model_kwargs: dict = None, constraints: torch.Tensor = None, chirality_config: list = None):
    device = diffusion.betas.device
    bsz = shape[0]
    xt = torch.randn(shape, device=device)
    if diffusion.cfg.recenter_every_step:
        xt = center(xt)

    unc_accum = torch.zeros(bsz, device=device)
    steps = 0

    for i in reversed(range(1, diffusion.cfg.T + 1)):
        t = torch.full((bsz,), i, device=device, dtype=torch.long)
        eps_stack = ensemble.predict_eps(xt, h, t, **(model_kwargs or {}))
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

        # Stability: clamp pred_x0 to prevent singularity at t=T
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        
        pred_x0 = (xt - sqrt_one_minus_alpha_bar * mean_eps) / (sqrt_alpha_bar + 1e-8)
        pred_x0 = torch.clamp(pred_x0, -10.0, 10.0)
        
        # Recompute effective noise from clamped x0
        mean_eps = (xt - sqrt_alpha_bar * pred_x0) / sqrt_one_minus_alpha_bar
        
        mean = (1 / torch.sqrt(alpha)) * (xt - (beta / sqrt_one_minus_alpha_bar) * mean_eps)
        if i > 1:
            sigma = torch.sqrt(diffusion.posterior_variance[idx])
            noise = torch.randn_like(xt)
            xt = mean + sigma * noise
        else:
            xt = mean
            
        # Generalized Constraints
        xt = constrain_chirality(xt, chirality_config=chirality_config)
        xt = constrain_bonds(xt, constraints=constraints, scale_factor=diffusion.cfg.scale_factor)
        
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
    model_kwargs: dict = None,
    constraints: torch.Tensor = None,
    chirality_config: list = None,
):
    device = diffusion.betas.device
    bsz = shape[0]
    xt = torch.randn(shape, device=device)
    if diffusion.cfg.recenter_every_step:
        xt = center(xt)

    unc_accum = torch.zeros(bsz, device=device)
    n_steps = 0
    times = torch.linspace(diffusion.cfg.T, 1, steps).long().to(device)

    for i, t_val in enumerate(times):
        t = torch.full((bsz,), t_val, device=device, dtype=torch.long)
        prev_t_val = times[i + 1] if i < len(times) - 1 else torch.tensor(0, device=device)

        eps_stack = ensemble.predict_eps(xt, h, t, **(model_kwargs or {}))
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
        pred_x0 = torch.clamp(pred_x0, -10.0, 10.0) # Prevent singularity at t=T
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * mean_eps
        noise = torch.randn_like(xt) if eta > 0 else 0.0

        xt = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
        
        # Generalized Constraints
        xt = constrain_chirality(xt, chirality_config=chirality_config)
        xt = constrain_bonds(xt, constraints=constraints, scale_factor=diffusion.cfg.scale_factor)
        
        if diffusion.cfg.recenter_every_step:
            xt = center(xt)

    return xt, unc_accum / max(1, n_steps)


@torch.no_grad()
@torch.no_grad()
def _sample_candidates(
    ensemble: Ensemble,
    diffusion,
    atom_types: torch.Tensor,
    n_samples: int,
    batch_size: int,
    steps: int,
    eta: float,
    condition: Union[torch.Tensor, dict] = None,
    constraints: torch.Tensor = None,
    chirality_config: list = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ensemble.eval()
    device = diffusion.betas.device
    n_atoms = atom_types.shape[0]
    samples = []
    uncertainty = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in range(n_batches):
        curr_bs = min(batch_size, n_samples - len(samples) * batch_size)
        h = atom_types.unsqueeze(0).expand(curr_bs, -1).to(device)
        shape = (curr_bs, n_atoms, 3)
        model_kwargs = {}
        
        c_batch = None
        if condition is not None:
             # Case 1: Dynamic Sampling (Dict)
             if isinstance(condition, dict):
                 strategy = condition.get("strategy", "fixed")
                 if strategy == "uniform":
                     c_min = condition.get("min", 0.0)
                     c_max = condition.get("max", 0.1)
                     # Sample uniformly [min, max]
                     # Shape: [B, 1]
                     c_vals = torch.rand(curr_bs, 1, device=device) * (c_max - c_min) + c_min
                     # Expand to atoms: [B, N, 1]
                     c_batch = c_vals.unsqueeze(1).expand(-1, n_atoms, -1)
             
             # Case 2: Fixed Tensor
             elif isinstance(condition, torch.Tensor):
                 # Expand fixed condition to batch: [N_atoms, 1] -> [B, N_atoms, 1]
                 # Assuming condition is [N, 1] derived from mean of corpus
                 if condition.dim() == 2:
                     c_batch = condition.unsqueeze(0).expand(curr_bs, -1, -1).to(device)
                 else:
                     c_batch = condition.to(device) # Already batched?
        
        if c_batch is not None:
            model_kwargs["condition"] = c_batch

        if eta == 0.0 and steps < diffusion.cfg.T:
            x0, unc = _consensus_ddim(ensemble, diffusion, shape, h, steps=steps, eta=eta, model_kwargs=model_kwargs, constraints=constraints, chirality_config=chirality_config)
        else:
            if eta != 0.0 and steps < diffusion.cfg.T:
                log.warning("DDIM steps < T with eta>0 is not supported; using full DDPM steps.")
            x0, unc = _consensus_ddpm(ensemble, diffusion, shape, h, model_kwargs=model_kwargs, constraints=constraints, chirality_config=chirality_config)
        samples.append(x0.cpu())
        uncertainty.append(unc.cpu())
        log.info("Sample batch %d/%d done.", i + 1, n_batches)
    return torch.cat(samples, dim=0), torch.cat(uncertainty, dim=0)


def _compute_phi_psi(samples: torch.Tensor, pdb_path: str = None, torsion_indices: Tuple[List[List[int]], List[List[int]]] = None) -> np.ndarray:
    """
    Computes Phi/Psi angles (degrees) for generated samples.
    Args:
        samples: [B, N, 3]
        pdb_path: Ignored if torsion_indices is provided. Backward compat.
        torsion_indices: (phi_indices, psi_indices) from MoleculeTopology.
    """
    device = samples.device
    
    if torsion_indices is not None:
        phi_idx, psi_idx = torsion_indices
    else:
        # Fallback to old behavior (Ala2 specific)
        if pdb_path is None: 
            return None
        # We need to import locally to avoid circular deps if needed or just use the old import
        from metastategen.utils.pdb import get_ala2_heavy_atom_indices
        phi_atoms, psi_atoms = get_ala2_heavy_atom_indices(Path(pdb_path))
        # Convert atom list to flat indices? get_ala2... returns ([i1, i2, i3, i4], [i1...])
        # Actually it returns lists of indices for ONE phi/psi pair.
        # We wrap in list to match structure of Generalized lists
        phi_idx = [phi_atoms]
        psi_idx = [psi_atoms]

    # Flatten logic: We only compute the FIRST pair for now for metric compatibility with Ala2 plots
    # If generalized, we might return ALL phis/psis? 
    # For now, let's just grab the first valid set found to keep return shape [B, 2] consistent with evaluation plots.
    
    if not phi_idx or not psi_idx:
        return None

    # Use first found torsion pair
    first_phi = phi_idx[0]
    first_psi = psi_idx[0]
    
    indices = torch.tensor([first_phi, first_psi], dtype=torch.long, device=device)
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    return degs.cpu().numpy()


def _basin_coverage(samples: torch.Tensor, medoids: torch.Tensor, rmsd_thresh: float) -> int:
    samples = samples.to(dtype=torch.float64, device="cpu")
    medoids = medoids.to(dtype=torch.float64, device="cpu")

    covered = set()
    for i in range(samples.shape[0]):
        cand = samples[i].unsqueeze(0).expand(medoids.shape[0], -1, -1)
        rmsds = rmsd_kabsch(cand, medoids)
        best = int(torch.argmin(rmsds).item())
        if float(rmsds[best].item()) <= rmsd_thresh:
            covered.add(best)
    return len(covered)


def _evaluate(
    ensemble: Ensemble,
    diffusion,
    atom_types: torch.Tensor,
    val_phi_psi: np.ndarray,
    val_medoids: torch.Tensor,
    cfg: dict,
    iter_dir: Path,
    seed: int,
    condition: Union[torch.Tensor, dict] = None,
    constraints: torch.Tensor = None,
    chirality_config: list = None,
    torsion_indices: tuple = None,
) -> dict:
    al_cfg = cfg.get("active_learning", {})
    n_eval = int(al_cfg.get("eval_samples", 1000))
    batch_size = int(al_cfg.get("sample_batch_size", 200))
    steps = int(al_cfg.get("sample_steps", 100))
    eta = float(al_cfg.get("sample_eta", 0.0))
    rmsd_thresh = float(al_cfg.get("rmsd_thresh", 0.75))

    set_deterministic(seed)
    samples, _ = _sample_candidates(ensemble, diffusion, atom_types, n_eval, batch_size, steps, eta, condition=condition, constraints=constraints, chirality_config=chirality_config)
    
    # Scale correction
    scale_factor = float(cfg.get("data", {}).get("scale_factor", 1.0))
    samples = samples / scale_factor

    torch.save(samples, iter_dir / "eval_samples.pt")

    meta_path = Path(cfg["data"].get("meta_path", "data/processed/ala2/meta.pt"))
    meta = torch.load(meta_path)
    pdb_path = Path(meta.get("pdb_path", "data/raw/alanine-dipeptide-nowater.pdb"))

    if pdb_path.exists() or torsion_indices is not None:
        gen_phi_psi = _compute_phi_psi(samples.to(diffusion.betas.device), pdb_path=pdb_path if pdb_path.exists() else None, torsion_indices=torsion_indices)
        if val_phi_psi is not None and gen_phi_psi is not None:
            kl = kl_from_phi_psi(gen_phi_psi, val_phi_psi, bins=int(al_cfg.get("phi_psi_bins", 180)))
        else:
            kl = -1.0
    else:
        log.warning("PDB not found and no torsion indices. Skipping Phi/Psi KL divergence.")
        kl = -1.0

    basin_count = _basin_coverage(samples, val_medoids, rmsd_thresh=rmsd_thresh)
    basin_frac = basin_count / max(1, val_medoids.shape[0])

    return {
        "kl_to_val": kl,
        "basin_count": basin_count,
        "basin_fraction": basin_frac,
        "eval_samples": n_eval,
    }


def run_active_learning(config_path: str) -> int:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    al_cfg = cfg.get("active_learning", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    run_root = _resolve_run_root(cfg)
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    seed_path = Path(data_cfg.get("seed_path", "data/processed/default/seed.pt"))
    pool_path = Path(data_cfg.get("pool_path", "data/processed/default/pool.pt"))
    val_path = Path(data_cfg.get("val_path", "data/processed/default/val.pt"))
    
    # Check if splits exist; if not, try to create them from raw NPZ + PDB
    if not (seed_path.exists() and pool_path.exists() and val_path.exists()):
        log.info("AL split files not found. Checking for raw inputs to auto-generate splits...")
        
        traj_source = data_cfg.get("traj_path") or data_cfg.get("npz_path")
        topo_source = data_cfg.get("topo_path") or data_cfg.get("pdb_path") # Used for topology
        
        # We need at least a Topology source to proceed with generalized loading
        if topo_source and Path(topo_source).exists():
            log.info(f"Ingesting raw data. Topo: {topo_source}, Traj: {traj_source}")
            
            # Load full dataset
            full_data = load_training_data(traj_path=traj_source, topo_path=topo_source)
            n_total = full_data["positions"].shape[0]
            log.info(f"Loaded {n_total} frames. Splitting...")
            
            # Splitting Logic
            n_seed = int(al_cfg.get("initial_seed_size", 100))
            n_val = int(al_cfg.get("val_size", 2000))
            n_pool = n_total - n_seed - n_val
            
            n_pool = n_total - n_seed - n_val
            
            # Low Data Check: AL is disabled if dataset is too small.
            if n_pool <= 0:
                 raise ValueError(
                     f"Dataset size ({n_total}) is insufficient for Active Learning with "
                     f"initial_seed_size={n_seed} and val_size={n_val}. "
                     "Active Learning requires a non-empty pool. "
                     "For small datasets, please use standard training ('msgen train')."
                 )
                
            # Sequential split for simplicity and trajectory coherence (Head=Seed, Tail=Val, Mid=Pool)
            # This mimics 'past' data (Seed) vs 'future' data (Val)
            positions = full_data["positions"]
            atom_types = full_data["atom_types"]
            
            # Use .clone() to decouple storage, ensuring saved files are small
            seed_data = {
                "positions": positions[:n_seed].clone(),
                "atom_types": atom_types.clone()
            }
            
            pool_data = {
                "positions": positions[n_seed:-n_val].clone(),
                "atom_types": atom_types.clone()
            }
            
            val_data = {
                "positions": positions[-n_val:].clone(),
                "atom_types": atom_types.clone()
            }
            
            # Save splits
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(seed_data, seed_path)
            torch.save(pool_data, pool_path)
            torch.save(val_data, val_path)
            log.info(f"Auto-generated splits saved to {seed_path.parent}")
            
        else:
             raise FileNotFoundError(f"Missing AL splits AND missing valid raw inputs (npz_path, pdb_path).")

    seed_data = load_al_data(seed_path)
    val_data = load_al_data(val_path)

    scale_factor = float(data_cfg.get("scale_factor", 1.0))
    if scale_factor != 1.0:
        log.info("Using data scale factor: %f", scale_factor)

    manager = ALDataManager(seed_data, scale_factor=scale_factor)

    atom_types = seed_data["atom_types"]
    n_atom_types = int(atom_types.max().item()) + 1
    
    # 4. Topology Inference
    topo_path = data_cfg.get("topo_path", data_cfg.get("pdb_path")) # Prefer topo_path, fallback to pdb
    constraints = None
    chirality_config = None
    torsion_indices = None # (phi_list, psi_list)

    if topo_path and Path(topo_path).exists():
        topology = MoleculeTopology(topo_path)
        log.info(f"Loaded topology from {topo_path}")
        constraints = topology.infer_constraints().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        chirality_config = topology.infer_chirality_config()
        torsion_indices = topology.infer_torsions() # ([phi_ids...], [psi_ids...])
        log.info(f"Inferred {len(chirality_config)} chiral centers and {len(torsion_indices[0])} phi torsions.")
    else:
        log.warning("No topology file found. Using NO constraints/torsions.")
    
    # Compute Target Chiral Condition
    condition_strategy = al_cfg.get("condition_strategy", "fixed")
    target_condition = None
    
    if condition_strategy == "uniform":
        c_range = al_cfg.get("condition_range", [0.01, 0.08])
        log.info(f"Using Uniform Conditioning: {c_range}")
        target_condition = {
            "strategy": "uniform",
            "min": float(c_range[0]),
            "max": float(c_range[1])
        }
    else:
        # Fixed strategy (Default)
        # Compute from Seed Data using inferred Chirality Config
        with torch.no_grad():
            # Seed data loaded from .pt is typically raw (nm).
            pos_subset = seed_data["positions"][:100]
            computed = compute_chiral_volume_signal(pos_subset, scale_factor=1.0, chirality_config=chirality_config).mean(dim=0)
            target_condition = computed.to(torch.device("cpu"))
            log.info(f"Using Fixed Conditioning: {target_condition.mean().item():.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = build_diffusion_from_cfg(cfg).to(device)

    member_states = []
    seeds = _resolve_seeds(cfg)
    for idx, seed in enumerate(seeds):
        set_deterministic(seed)
        model = build_model_from_cfg(cfg, n_atom_types=n_atom_types).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)))
        member_dir = run_root / "members" / f"m{idx:03d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "model": model,
            "opt": opt,
            "epoch": 0,
            "logs": [],
            "device": device,
            "seed": seed,
            "member_dir": member_dir,
        }
        member_states.append(state)

    ensemble = Ensemble([s["model"] for s in member_states])

    oracle_device = al_cfg.get("oracle_device", "cpu")
    oracle = DatasetOracle(pool_path, device=oracle_device, batch_size=int(al_cfg.get("oracle_batch_size", 100)))

    # Val data is loaded raw (nm). ALDataManager handles scaling for training.
    # Metrics evaluation expects nm.
    val_phi_psi = None
    if "phi_psi" in val_data:
        val_phi_psi = val_data["phi_psi"].cpu().numpy()
    else:
        # PDB Path handling for Reference data check
        pdb_path = Path(data_cfg.get("pdb_path", "data/raw/alanine-dipeptide-nowater.pdb"))
        if torsion_indices is not None:
             val_phi_psi = _compute_phi_psi(val_data["positions"].to(device), torsion_indices=torsion_indices)
        elif pdb_path.exists():
             val_phi_psi = _compute_phi_psi(val_data["positions"].to(device), pdb_path=pdb_path)
        else:
             log.warning("PDB not found and no torsion indices. Val Phi/Psi will be None.")
             val_phi_psi = None

    rmsd_thresh = float(al_cfg.get("rmsd_thresh", 0.75))
    _, val_medoids_idx, _ = greedy_cluster(val_data["positions"], rmsd_thresh=rmsd_thresh)
    val_medoids = val_data["positions"][val_medoids_idx]
    log.info("Val basins: %d", len(val_medoids_idx))

    used_pool_indices = set()

    metrics_path = run_root / "al_metrics.csv"
    metrics_rows = []

    # Phase 0: cold-start training on seed
    init_epochs = int(train_cfg.get("epochs", 5))
    log.info("Cold start: training %d epochs on seed (%d frames)", init_epochs, manager.size())
    for state in member_states:
        dl = _build_dataloader(
            manager.dataset(),
            batch_size=int(data_cfg.get("batch_size", 256)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            seed=state["seed"],
        )
        _train_member(
            state,
            diffusion,
            dl,
            epochs=init_epochs,
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            rot_aug=bool(train_cfg.get("rot_aug", True)),
            iter_idx=0,
            chirality_config=chirality_config,
        )
        _save_checkpoint(state["member_dir"], state, iter_idx=0, cfg=cfg)
        _save_member_logs(state["member_dir"], state["logs"])

    iter_dir = run_root / "iter_00"
    iter_dir.mkdir(parents=True, exist_ok=True)
    eval_seed = int(al_cfg.get("seed", train_cfg.get("seed", 0)))
    metrics = _evaluate(ensemble, diffusion, atom_types, val_phi_psi, val_medoids, cfg, iter_dir, eval_seed, condition=target_condition, constraints=constraints, chirality_config=chirality_config, torsion_indices=torsion_indices)
    metrics.update(
        {
            "iter": 0,
            "oracle_calls": 0,
            "train_size": manager.size(),
        }
    )
    metrics_rows.append(metrics)
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in metrics_rows for k in r.keys()}))
        writer.writeheader()
        writer.writerows(metrics_rows)

    n_iters = int(al_cfg.get("n_iters", 3))
    n_candidates = int(al_cfg.get("n_candidates", 1000))
    n_acquire = int(al_cfg.get("n_acquire", 200))
    strategy = str(al_cfg.get("acquisition_strategy", "uncertainty"))

    for iter_idx in range(1, n_iters + 1):
        log.info("AL iter %d/%d", iter_idx, n_iters)
        iter_dir = run_root / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        sample_seed = int(al_cfg.get("seed", train_cfg.get("seed", 0))) + iter_idx
        set_deterministic(sample_seed)

        samples, uncertainty = _sample_candidates(
            ensemble,
            diffusion,
            atom_types,
            n_samples=n_candidates,
            batch_size=int(al_cfg.get("sample_batch_size", 200)),
            steps=int(al_cfg.get("sample_steps", 100)),
            eta=float(al_cfg.get("sample_eta", 0.0)),
            condition=target_condition,
            constraints=constraints,
            chirality_config=chirality_config,
        )
        
        # UNSCALE generated samples before saving/using
        samples = samples / scale_factor

        torch.save(samples, iter_dir / "candidates.pt")
        torch.save(uncertainty, iter_dir / "uncertainty.pt")

        gen = torch.Generator()
        gen.manual_seed(sample_seed)
        sel_idx = select_acquisition(uncertainty, n_acquire, strategy=strategy, generator=gen)
        sel_idx = sel_idx.cpu()

        selected = samples[sel_idx]
        labels = oracle.query(selected.to(oracle.device))
        pool_idx = oracle.last_indices
        if pool_idx is None:
            raise RuntimeError("Oracle did not populate last_indices")

        unique_mask = []
        for idx in pool_idx.tolist():
            if idx in used_pool_indices:
                unique_mask.append(False)
            else:
                unique_mask.append(True)
                used_pool_indices.add(idx)
        unique_mask = torch.tensor(unique_mask, dtype=torch.bool)

        if unique_mask.sum() < pool_idx.numel():
            log.warning(
                "Filtered %d duplicate pool indices in iter %d",
                int(pool_idx.numel() - unique_mask.sum().item()),
                iter_idx,
            )

        pool_idx = pool_idx[unique_mask]
        selected = selected[unique_mask]
        labels = labels[unique_mask.to(labels.device)]
        selected_unc = uncertainty[sel_idx][unique_mask]

        meta = oracle.get_metadata(pool_idx)
        acquired = {
            "positions": labels.cpu(),
            "atom_types": atom_types,
        }
        acquired.update(meta)
        torch.save(acquired, iter_dir / "acquired.pt")

        log_rows = []
        if pool_idx.numel() > 0:
            kept_sel_idx = sel_idx[unique_mask]
            for i, idx in enumerate(pool_idx.tolist()):
                row = {
                    "iter": iter_idx,
                    "candidate_index": int(kept_sel_idx[i].item()),
                    "pool_index": int(idx),
                    "uncertainty": float(selected_unc[i].item()),
                }
                if "traj_id" in meta:
                    row["traj_id"] = int(meta["traj_id"][i].item())
                if "frame_id" in meta:
                    row["frame_id"] = int(meta["frame_id"][i].item())
                if "source_index" in meta:
                    row["source_index"] = int(meta["source_index"][i].item())
                log_rows.append(row)

        log_path = iter_dir / "acquired_indices.csv"
        with log_path.open("w", newline="") as f:
            fieldnames = sorted({k for r in log_rows for k in r.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames or ["iter", "candidate_index", "pool_index"])
            writer.writeheader()
            if log_rows:
                writer.writerows(log_rows)

        manager.append(acquired)
        torch.save(manager.cumulative_data(), iter_dir / "cumulative.pt")

        finetune_epochs = int(al_cfg.get("finetune_epochs", 5))
        for state in member_states:
            dl = _build_dataloader(
                manager.dataset(),
                batch_size=int(data_cfg.get("batch_size", 256)),
                num_workers=int(data_cfg.get("num_workers", 0)),
                seed=state["seed"] + iter_idx,
            )
            _train_member(
                state,
                diffusion,
                dl,
                epochs=finetune_epochs,
                grad_clip=float(train_cfg.get("grad_clip", 1.0)),
                rot_aug=bool(train_cfg.get("rot_aug", True)),
                iter_idx=iter_idx,
                chirality_config=chirality_config,
            )
            _save_checkpoint(state["member_dir"], state, iter_idx=iter_idx, cfg=cfg)
            _save_member_logs(state["member_dir"], state["logs"])

        metrics = _evaluate(
            ensemble,
            diffusion,
            atom_types,
            val_phi_psi,
            val_medoids,
            cfg,
            iter_dir,
            eval_seed + 1000 + iter_idx,
            condition=target_condition,
            constraints=constraints,
            chirality_config=chirality_config,
        )
        metrics.update(
            {
                "iter": iter_idx,
                "oracle_calls": int(manager.size() - seed_data["positions"].shape[0]),
                "train_size": manager.size(),
            }
        )
        metrics_rows.append(metrics)

        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in metrics_rows for k in r.keys()}))
            writer.writeheader()
            writer.writerows(metrics_rows)

    for state in member_states:
        ckpt_dir = state["member_dir"] / "checkpoints"
        torch.save(state["model"].state_dict(), ckpt_dir / "final.pt")

    log.info("AL loop complete. Metrics: %s", metrics_path)
    return 0
