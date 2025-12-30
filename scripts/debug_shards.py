
import torch
import glob
import os

def main():
    shard_dir = "data/processed/ala2_all_atom/shards"
    shards = sorted(glob.glob(f"{shard_dir}/*.pt"))
    print(f"Found {len(shards)} shards.")
    
    total_frames = 0
    for s in shards:
        try:
            d = torch.load(s)
            n = d['positions'].shape[0]
            total_frames += n
            # print(f"{os.path.basename(s)}: {n}")
        except Exception as e:
            print(f"Error loading {s}: {e}")
            
    print(f"Total frames in shards: {total_frames}")
    
    force_path = "data/processed/ala2_all_atom/al_forces_ref.pt"
    if os.path.exists(force_path):
        f = torch.load(force_path)
        print(f"Total frames in forces: {f.shape[0]}")
        print(f"Difference: {total_frames - f.shape[0]}")
    else:
        print("Force file not found.")

if __name__ == "__main__":
    main()
