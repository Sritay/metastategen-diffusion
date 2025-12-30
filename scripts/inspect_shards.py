import torch
import glob
import os
from collections import Counter

def main():
    shard_dir = "data/processed/ala2/shards"
    shards = sorted(glob.glob(f"{shard_dir}/*.pt"))
    print(f"Found {len(shards)} shards in {shard_dir}")
    
    traj_counts = Counter()
    
    for i, s in enumerate(shards):
        try:
            d = torch.load(s, map_location="cpu")
            if "traj_id" in d:
                t_ids = d["traj_id"]
                # Count unique traj IDs in this shard
                trajs = t_ids.unique().tolist()
                for t in trajs:
                    count = (t_ids == t).sum().item()
                    traj_counts[t] += count
            
            if i % 50 == 0:
                print(f"Processed {i} shards...")
        except Exception as e:
            print(f"Error loading {s}: {e}")
            
    print("\nTrajectory Counts:")
    for t, count in sorted(traj_counts.items()):
        print(f"Traj {t}: {count} frames")

if __name__ == "__main__":
    main()
