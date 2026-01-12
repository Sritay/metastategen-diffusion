import torch
import glob
from pathlib import Path

def main():
    shards_dir = Path("data/processed/ala2/shards")
    shard_paths = sorted(list(shards_dir.glob("*.pt")))
    
    counts = {}
    
    print(f"Scanning {len(shard_paths)} shards...")
    
    for p in shard_paths:
        data = torch.load(p)
        t_ids = data["traj_id"]
        uniques, c = torch.unique(t_ids, return_counts=True)
        
        for u, n in zip(uniques, c):
            uid = int(u.item())
            counts[uid] = counts.get(uid, 0) + int(n.item())
            
    print("\n--- Trajectory Counts ---")
    for tid, count in sorted(counts.items()):
        print(f"Traj ID {tid}: {count:,} frames")

if __name__ == "__main__":
    main()
