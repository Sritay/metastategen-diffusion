
import torch
from pathlib import Path

def main():
    path = "data/timewarp/train/positions.pt"
    if not Path(path).exists():
        print("Template not found.")
        return

    # Load 1st frame
    pos = torch.load(path)[0] # [22, 3]
    print(f"Loaded template shape: {pos.shape}")
    
    # Check distances between "assumed" backbone atoms
    # N=3, CA=5, C=11?
    
    indices = [
        (3, 5, "3-5 (N-CA?)"),
        (5, 11, "5-11 (CA-C?)"),
        (0, 1, "0-1"),
        (1, 2, "1-2"),
        (1, 3, "1-3"), # Acetyl C-N?
        (4, 5, "4-5?"),
        (5, 6, "5-6?"),
        (6, 7, "6-7?"),
    ]
    
    print("\n--- Checking Distances in Template (nm) ---")
    for i, j, label in indices:
        dist = torch.norm(pos[i] - pos[j]).item()
        print(f"{label}: {dist:.4f}")
        
    print("\nIf 3-5 is ~0.146 and 5-11 is ~0.151, then indices are correct.")

if __name__ == "__main__":
    main()
