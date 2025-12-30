from pathlib import Path

def main():
    pdb_path = Path("data/timewarp/train/ad1-traj-state0.pdb")
    indices = []
    
    with open(pdb_path, 'r') as f:
        idx = 0
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Element symbol is usually column 76-77, or inferred from name.
                # Standard PDB name is col 12-16.
                name = line[12:16].strip()
                element = line[76:78].strip()
                if not element:
                    # Infer from name
                    element = name[0]
                
                if element != 'H':
                    indices.append(idx)
                    print(f"Index {idx}: {name} ({element})")
                
                idx += 1
                
    print(f"Heavy Indices: {indices}")
    print(f"Count: {len(indices)}")

if __name__ == "__main__":
    main()
