
import numpy as np
from pathlib import Path

def parse_pdb_models(path):
    models = []
    current_model = []
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("MODEL"):
                if current_model:
                    models.append(np.array(current_model))
                    current_model = []
            elif line.startswith("ATOM"):
                parts = line.split()
                # x, y, z are usually columns 30-38, 38-46, 46-54
                # Standard PDB: 
                # x: line[30:38]
                # y: line[38:46]
                # z: line[46:54]
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    current_model.append([x, y, z])
                except:
                    pass
            elif line.startswith("ENDMDL"):
                if current_model:
                    models.append(np.array(current_model))
                    current_model = []
                    
    if current_model:
         models.append(np.array(current_model))
         
    return models

def check_structure(pos, name):
    # PDB is in Angstroms.
    # N(6), CA(8), C(14) in our 0-indexed list (from 22 atoms)
    # But wait. My script 'create_aligned_movie' wrote 22 atoms sequentially.
    # So indices 0..21 map to the names in the list.
    # The list was:
    # ["C1", "H1", "H2", "H3", "C2", "O2", "N3", "H4", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C4", "O4", "N5", "H5", "C6", "H6", "H7", "H8"]
    # So:
    # N3 is index 6 (7th atom)
    # CA is index 8 (9th atom)
    # CB is index 10 (11th atom)
    # C4 is index 14 (15th atom)
    
    idx_N = 6
    idx_CA = 8
    idx_CB = 10
    idx_C = 14
    
    n = pos[idx_N]
    ca = pos[idx_CA]
    cb = pos[idx_CB]
    c = pos[idx_C]
    
    # Bond Lengths (Angstrom)
    d_n_ca = np.linalg.norm(n - ca)
    d_ca_c = np.linalg.norm(ca - c)
    d_ca_cb = np.linalg.norm(ca - cb)
    
    # Angle N-CA-C
    v1 = n - ca
    v2 = c - ca
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # Chiral Volume
    # V = (N-CA) . ((CB-CA) x (C-CA))
    v_n = n - ca
    v_cb = cb - ca
    v_c = c - ca
    vol = np.dot(v_n, np.cross(v_cb, v_c))
    
    return {
        "N-CA": d_n_ca,
        "CA-C": d_ca_c,
        "CA-CB": d_ca_cb,
        "N-CA-C": theta,
        "Volume": vol
    }

def main():
    path = Path("demo/aligned_final.pdb")
    if not path.exists():
        print("PDB not found")
        return
        
    print(f"Loading {path}...")
    models = parse_pdb_models(path)
    print(f"Loaded {len(models)} models.")
    
    stats_list = []
    
    print(f"Loaded {len(models)} models.")
    
    init_stats = []
    ref_stats = []
    
    for i, m in enumerate(models):
        stats = check_structure(m, f"Model {i+1}")
        if (i % 2) == 0:
            init_stats.append(stats)
        else:
            ref_stats.append(stats)
            
    # Helper to print summary
    def print_summary(label, s_list):
        n_ca = [s["N-CA"] for s in s_list]
        angles = [s["N-CA-C"] for s in s_list]
        vols = [s["Volume"] for s in s_list]
        
        print(f"\n--- {label} (N={len(s_list)}) ---")
        print(f"N-CA Length: {np.mean(n_ca):.4f} +/- {np.std(n_ca):.4f}")
        print(f"N-CA-C Angle: {np.mean(angles):.2f} +/- {np.std(angles):.2f} (Target ~110.0)")
        print(f"Chiral Vol: {np.mean(vols):.4f}")
        
    print_summary("Initial (Generated)", init_stats)
    print_summary("Refined (Final)", ref_stats)

if __name__ == "__main__":
    main()
