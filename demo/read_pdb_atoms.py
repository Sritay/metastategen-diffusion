
def main():
    pdb_path = "data/timewarp/train/ad1-traj-state0.pdb"
    print(f"Reading atoms from {pdb_path}")
    atoms = []
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    parts = line.split()
                    # Keep it simple, just store the line or extract name
                    # Standard PDB: Atom Name is parts[2] usually
                    atoms.append(line.strip())
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Index | Line")
    for i, line in enumerate(atoms):
        print(f"{i:4d} | {line}")

    heavy_indices = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
    print("\n--- Scheduled Heavy Indices ---")
    for idx in heavy_indices:
        if 0 <= idx < len(atoms):
            print(f"{idx:4d} | {atoms[idx]}")
        else:
            print(f"{idx:4d} | OUT OF BOUNDS")

if __name__ == "__main__":
    main()
