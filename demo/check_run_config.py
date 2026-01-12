
import yaml
import torch
from pathlib import Path

def main():
    print("Checking run configuration for Loop 19 features...")
    
    # 1. Check YAML config
    yaml_path = Path("runs/day11_al_18_hpc/config.yaml")
    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
            al_cfg = cfg.get("active_learning", {})
            print(f"YAML Config Strategy: {al_cfg.get('condition_strategy', 'Not Found')}")
            print(f"YAML Config Range: {al_cfg.get('condition_range', 'Not Found')}")
    else:
        print("config.yaml not found.")
        
    # 2. Check Checkpoint Config (Truth)
    ckpt_path = Path("runs/day11_al_18_hpc/members/m000/checkpoints/final.pt")
    if ckpt_path.exists():
        # Checkpoints typically contain model state dict, not full config in our current saving logic?
        # Let's check iter_00.pt which we verify saves config
        iter0_path = Path("runs/day11_al_18_hpc/members/m000/checkpoints/iter_00.pt")
        if iter0_path.exists():
             ckpt = torch.load(iter0_path, map_location="cpu")
             saved_cfg = ckpt.get("config", {})
             al_cfg = saved_cfg.get("active_learning", {})
             print(f"Checkpoint Strategy: {al_cfg.get('condition_strategy', 'Not Found')}")
             print(f"Checkpoint Range: {al_cfg.get('condition_range', 'Not Found')}")
        else:
            print("iter_00.pt checkpoint not found.")
    else:
        print("final.pt not found.")

if __name__ == "__main__":
    main()
