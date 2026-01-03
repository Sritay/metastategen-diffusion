import torch
from pathlib import Path

def check_file(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"MISSING: {p}")
        return
        
    data = torch.load(p, map_location='cpu')
    if isinstance(data, dict):
        if 'pos' in data: data = data['pos']
        elif 'data' in data: data = data['data']
        elif 'positions' in data: data = data['positions']
        
    if hasattr(data, 'shape'):
        print(f"{p}: {data.shape}")
    else:
        print(f"{p}: Type {type(data)} (len {len(data)})")

print("--- Seed ---")
check_file("data/processed/ala2/split_5/al_seed.pt")

print("\n--- Iter 01 ---")
check_file("runs/day10_al_5_hpc/iter_01/cumulative.pt")
check_file("runs/day10_al_5_hpc/iter_01/acquired.pt")

print("\n--- Iter 02 ---")
check_file("runs/day10_al_5_hpc/iter_02/cumulative.pt")
check_file("runs/day10_al_5_hpc/iter_02/acquired.pt")
