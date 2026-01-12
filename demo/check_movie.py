
def analyze_movie(pdb_path):
    print(f"Analyzing {pdb_path}")
    
    current_model = None
    frames = {}
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("MODEL"):
                current_model = int(line.split()[1])
                frames[current_model] = []
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                # PDB coordinates are columns 30-38, 38-46, 46-54
                # Or just split() if standard spacing isn't guaranteed, but PDB is fixed width.
                # Let's try split first as it's more robust to simple files
                parts = line.split()
                # Finding x, y, z. Usually parts[6], parts[7], parts[8] for ATOM
                # Example: ATOM      1  N   ALA A   1       2.000   1.000   0.000
                x = float(parts[6])
                y = float(parts[7])
                z = float(parts[8])
                frames[current_model].append((x, y, z))
                
    # Analyze first 4 frames
    for i in range(1, 5):
        if i not in frames: break
        
        coords = frames[i]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        
        min_c = min(min(xs), min(ys), min(zs))
        max_c = max(max(xs), max(ys), max(zs))
        
        type_str = "GENERATED" if i % 2 != 0 else "REFINED"
        print(f"Model {i} ({type_str}): Min={min_c:.2f}, Max={max_c:.2f}")
        
        if i % 2 != 0:
             # Print a few atoms to see deviations
             print("  Sample Atomic Coords:")
             for j, (x,y,z) in enumerate(coords[:5]):
                 print(f"    Atom {j}: {x:.2f}, {y:.2f}, {z:.2f}")

if __name__ == "__main__":
    analyze_movie("demo/refinement_movie.pdb")
