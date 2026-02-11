from __future__ import annotations

from pathlib import Path
import yaml
import torch
from rich.logging import RichHandler
import logging

from metastategen.utils import get_logger, set_deterministic
from metastategen.models.ensemble import build_diffusion_from_cfg, build_model_from_cfg
from metastategen.data import load_npz_as_al_data, ALDataManager
from metastategen.data.topology import MoleculeTopology
from metastategen.workflows.common import (
    _resolve_run_root,
    _build_dataloader,
    _save_checkpoint,
    _save_member_logs,
    _train_member,
)

log = get_logger("train")

def run_training(config_path: str) -> int:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Use resolve_run_root for consistent output directory handling
    # train.py config usually has 'train' -> 'out_dir' or similar. 
    # _resolve_run_root checks active_learning.exp_id, active_learning.out_dir, 
    # then falls back to train.exp_id, train.out_dir.
    run_root = _resolve_run_root(cfg)
    run_root.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with (run_root / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    
    seed = int(train_cfg.get("seed", 0))
    set_deterministic(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # --- Data Loading (Generalized) ---
    # We use load_npz_as_al_data to support arbitrary NPZ + PDB inputs
    # Just like AL loop, but we won't do splitting, we just load 'train' data.
    # 
    # If users provide explicit paths in data_cfg, we use them.
    # If they use the old 'data_dir' + 'train_trajs' style (Ala2Dataset), we might need to support that?
    #
    # Proposal: Support BOTH for backward compatibility, or switch to Generalized?
    # The user request implies "make AL optional", suggesting we want the SAME capabilities as AL (generalized) but without the loop.
    # So we should prioritize the Generalized Loader.
    
    npz_path = data_cfg.get("npz_path")
    pdb_path = data_cfg.get("pdb_path")
    
    if npz_path and pdb_path:
        log.info(f"Loading generalized data from {npz_path} and {pdb_path}")
        scale_factor = float(data_cfg.get("scale_factor", 1.0))
        raw_data = load_npz_as_al_data(Path(npz_path), Path(pdb_path))
        
        # In a normal training script, we might want to split into Train/Val manually or assume provided data is Train.
        # For simplicity, let's treat the entire NPZ as training data.
        # If user wants a split, they should pre-split or we add a 'val_split' arg?
        # AL loop auto-splits. Here let's just train on provided data.
        
        # --- Augmentation for Low Data ---
        augment_low_data = bool(data_cfg.get("augment_low_data", True))
        min_frames = int(data_cfg.get("min_aug_frames", 100)) # If fewer frames than this, augment.
        
        # Or explicit 'n_copies' control
        n_copies = int(data_cfg.get("aug_n_copies", 1000))
        noise_scale = float(data_cfg.get("aug_noise_scale", 0.05))
        
        n_frames = raw_data["positions"].shape[0]
        
        if augment_low_data and n_frames < min_frames:
            log.info(f"Low data detected ({n_frames} < {min_frames}). Applying Thermal Augmentation.")
            from metastategen.data.augmentation import augment_with_noise_and_rotations
            raw_data = augment_with_noise_and_rotations(
                raw_data, 
                n_copies=n_copies, 
                noise_scale=noise_scale
            )
            log.info(f"Data augmented to {raw_data['positions'].shape[0]} frames.")
        
        # ALDataManager wrappers are useful for scale handling, so we reuse it?
        # Yes, ALDataManager handles scaling and "PositionsDataset" creation.
        manager = ALDataManager(raw_data, scale_factor=scale_factor)
        
        # Infer topology for n_atom_types and constraints
        top_file = data_cfg.get("topology_extension", None) # Or just 'topology_path' distinct from 'pdb_path'?
        # Let's standardize on: 'pdb_path' is main structure/coords. 'topology_path' is optional extra.
        topology_path_arg = data_cfg.get("topology_path")
        
        # If user provided 'topo_path' in old config, it might mean the PDB itself?
        # Let's clean this up.
        
        topology = MoleculeTopology(pdb_path, topology_path=topology_path_arg)
        chirality_config = topology.infer_chirality_config()
        # Explicitly infer constraints to log them (and check rings)
        constraints = topology.infer_constraints()
        
        n_atom_types = int(raw_data["atom_types"].max().item()) + 1
        
    else:
        # Fallback? No, require updated config for generalized training.
        raise ValueError("Missing data config. Provide 'npz_path' & 'pdb_path' (Generalized). Legacy Ala2Dataset support has been removed.")

    # --- Model ---
    model = build_model_from_cfg(cfg, n_atom_types=n_atom_types).to(device)
    diffusion = build_diffusion_from_cfg(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)))
    
    # Setup State (similar to AL member state)
    # But here we only have ONE model (no ensemble, or rather, just 1 "member")
    state = {
        "model": model,
        "opt": opt,
        "epoch": 0,
        "logs": [],
        "device": device,
        "chirality_config": chirality_config
    }
    
    training_dataset = manager.dataset()
    
    batch_size = int(data_cfg.get("batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 0))
    
    dl = _build_dataloader(
        training_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed
    )
    
    epochs = int(train_cfg.get("epochs", 100))
    save_every = int(train_cfg.get("save_every", 10))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    rot_aug = bool(train_cfg.get("rot_aug", True))
    
    log.info(f"Starting Normal Training for {epochs} epochs...")
    
    # We can reuse _train_member, but it loops for 'epochs'. 
    # We want to save every X epochs.
    # So we call _train_member for 1 epoch at a time inside our loop?
    # Or strict reuse? _train_member runs for N epochs and saves logs.
    # It does NOT save checkpoints internally during the loop (only afterwards in AL loop).
    # So we should call it in a loop of our own.
    
    for epoch in range(1, epochs + 1):
        # Run 1 epoch
        _train_member(
            state,
            diffusion,
            dl,
            epochs=1,
            grad_clip=grad_clip,
            rot_aug=rot_aug,
            iter_idx=None, # Normal training doesn't use AL iters
            chirality_config=chirality_config
        )
        
        # Log to console
        last_log = state["logs"][-1]
        log.info(f"Epoch {epoch}/{epochs} | Loss: {last_log['train_loss']:.6f}")
        
        # Save Checkpoint
        if epoch % save_every == 0:
            _save_checkpoint(run_root, state, iter_idx=None, cfg=cfg)
            # Also save logs periodically
            _save_member_logs(run_root, state["logs"])
            log.info(f"Saved checkpoint for epoch {epoch}")

    # Final Save
    final_path = run_root / "checkpoints" / "final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    _save_member_logs(run_root, state["logs"])
    
    log.info(f"Done. Results in {run_root}")
    return 0
