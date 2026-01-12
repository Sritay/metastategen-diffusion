
import torch
path = "runs/day10_al_9_hpc/members/m000/checkpoints/final.pt"
try:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        print("Keys:", data.keys())
    else:
        print("Data is not a dict, it is type:", type(data))
except Exception as e:
    print(e)
