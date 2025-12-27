import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional
import glob

from metastategen.utils import get_logger

log = get_logger("dataset")

class Ala2Dataset(Dataset):
    def __init__(
        self, 
        shard_dir: str, 
        trajs: Optional[List[int]] = None,
        subsample: int = 1
    ):
        super().__init__()
        self.shard_paths = sorted(glob.glob(f"{shard_dir}/*.pt"))
        if not self.shard_paths:
            log.warning(f"No shards found in {shard_dir}")
            
        self.data = []
        
        # Load all shards into memory (Ala2 is small enough)
        # Filters by trajectory ID if provided
        loaded_count = 0
        for p in self.shard_paths:
            d = torch.load(p)
            # Check if any frame in this shard belongs to requested trajs
            # (Assumes shards might be mixed, though usually contiguous)
            traj_ids = d['traj_id']
            
            mask = torch.zeros_like(traj_ids, dtype=torch.bool)
            if trajs is None:
                mask[:] = True
            else:
                for t in trajs:
                    mask |= (traj_ids == t)
            
            if not mask.any():
                continue

            # Apply subsampling
            indices = torch.where(mask)[0]
            if subsample > 1:
                indices = indices[::subsample]
                
            self.data.append({
                "positions": d['positions'][indices],
                "atom_types": d['atom_types'], # Shared across frames
                "traj_id": d['traj_id'][indices]
            })
            loaded_count += len(indices)

        if len(self.data) > 0:
            self.positions = torch.cat([x['positions'] for x in self.data], dim=0)
            self.atom_types = self.data[0]['atom_types'] # Assume constant topology
            self.traj_ids = torch.cat([x['traj_id'] for x in self.data], dim=0)
        else:
            self.positions = torch.empty(0)
            self.atom_types = torch.empty(0)
            self.traj_ids = torch.empty(0)
            
        log.info(f"Loaded {len(self.positions)} frames from {len(self.shard_paths)} shards. Trajs={trajs}, Subsample={subsample}")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return {
            "x": self.positions[idx],       # [N, 3]
            "a": self.atom_types,           # [N]
            "t": self.traj_ids[idx]         # Scalar
        }
