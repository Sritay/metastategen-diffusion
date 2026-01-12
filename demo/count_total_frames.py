import torch
from pathlib import Path

def count_frames(path):
    if not path.exists():
        return 0, f"File not found: {path}"
    data = torch.load(path)
    if isinstance(data, dict):
        if 'pos' in data: return len(data['pos']), "dict['pos']"
        if 'positions' in data: return len(data['positions']), "dict['positions']"
        if 'data' in data: return len(data['data']), "dict['data']"
    if hasattr(data, 'shape'):
        return data.shape[0], "tensor"
    if isinstance(data, list):
        return len(data), "list"
    return 0, "unknown format"

def main():
    base_dir = Path("data/processed/ala2")
    split_dir = base_dir / "split_5"
    shards_dir = base_dir / "shards"

    # 1. Seed
    seed_count, seed_fmt = count_frames(split_dir / "al_seed.pt")
    
    # 2. Val
    val_count, val_fmt = count_frames(split_dir / "al_val.pt")
    
    # 3. Pool (Shards)
    pool_count = 0
    shards = list(shards_dir.glob("shard_*.pt"))
    first_shard_count = 0
    if shards:
        # Check first shard
        first_shard_count, _ = count_frames(shards[0])
        pool_count = first_shard_count * len(shards)
    
    print(f"--- Data Stats ---")
    print(f"Seed: {seed_count:,} frames ({seed_fmt})")
    print(f"Val:  {val_count:,} frames ({val_fmt})")
    print(f"Pool: ~{pool_count:,} frames ({len(shards)} shards x {first_shard_count})")
    print(f"Total Universe: ~{seed_count + val_count + pool_count:,} frames")

if __name__ == "__main__":
    main()
