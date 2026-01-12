import torch
from pathlib import Path

def main():
    path = "data/processed/ala2/shards/shard_00000.pt"
    print(f"Loading {path}...")
    data = torch.load(path)
    
    types = data['atom_types']
    print(f"Atom Types ({len(types)} atoms): {types}")
    
    # Try to decode if they are atomic numbers
    # 1=H, 6=C, 7=N, 8=O
    elements = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
    decoded = []
    for t in types:
        elem = elements.get(int(t.item()), f"Z={int(t.item())}")
        decoded.append(elem)
        
    print(f"Decoded Elements: {decoded}")
    
    # Verify mapping list from export script
    export_types = ["C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "H", "H"]
    
    match = True
    for i, (d, e) in enumerate(zip(decoded, export_types)):
        if d != e:
            print(f"MISMATCH at index {i}: Data has {d}, Export Script expects {e}")
            match = False
            
    if match:
        print("Atom Types MATCH export script.")
    else:
        print("Atom Types DO NOT MATCH export script!")

if __name__ == "__main__":
    main()
