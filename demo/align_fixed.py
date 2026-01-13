
import numpy as np
from pathlib import Path

def parse_pdb_models(path):
    models = []
    current_model_lines = []
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("MODEL"):
                if current_model_lines:
                    models.append(current_model_lines)
                current_model_lines = [line] # Keep header
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                current_model_lines.append(line)
            elif line.startswith("ENDMDL"):
                current_model_lines.append(line)
                models.append(current_model_lines)
                current_model_lines = []
            elif line.startswith("TER"):
                current_model_lines.append(line)
                
    if current_model_lines:
        models.append(current_model_lines)
        
    return models

def get_coords(lines):
    # Extracts coordinates from ATOM lines into Nx3 array
    coords = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    return np.array(coords)

def align_coords(coords):
    # Indices (0-based) from grep:
    # N is 7th atom -> Index 6
    # CA is 9th atom -> Index 8
    # C is 15th atom -> Index 14
    
    idx_N = 6
    idx_CA = 8
    idx_C = 14
    
    # 1. Centering on CA
    ca = coords[idx_CA]
    centered = coords - ca
    
    # 2. Construct Basis
    # N-CA vector becomes Z axis
    n_vec = centered[idx_N]
    z_new = n_vec / np.linalg.norm(n_vec)
    
    # C-CA vector helps define the plane
    c_vec = centered[idx_C]
    
    # Normal to plane (N-CA-C) becomes Y axis? or X?
    # User asked for "N-CA fixed in space" (Z) and others changing.
    # Usually we put the plane in XZ or YZ.
    # Let's put Normal in Y. Then N-CA-C plane is XZ.
    
    y_new_raw = np.cross(z_new, c_vec) # Normal to N-CA and C-CA
    y_new = y_new_raw / np.linalg.norm(y_new_raw)
    
    # X axis is cross(Y, Z)
    x_new = np.cross(y_new, z_new)
    # Already normalized since Y, Z orthogonal and unit.
    
    # Rotation Matrix (Rows are new basis vectors IF mapping TO standard basis?)
    # We want Basis {x_new, y_new, z_new} to map to {e_1, e_2, e_3}.
    # coordinate v_new = R * v_old
    # R * x_new = e_1 (1,0,0)
    # R * y_new = e_2 (0,1,0)
    # R * z_new = e_3 (0,0,1)
    # This implies R has rows x_new, y_new, z_new.
    
    R = np.vstack([x_new, y_new, z_new])
    
    # Apply R to all coordinates (v_aligned = R @ v_centered.T).T
    aligned = (R @ centered.T).T
    
    return aligned

def write_model(f, lines, new_coords, model_num):
    atom_idx = 0
    f.write(f"MODEL     {model_num}\n")
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            x, y, z = new_coords[atom_idx]
            # Replace coords in line
            # 30-38, 38-46, 46-54
            prefix = line[:30]
            suffix = line[54:]
            f.write(f"{prefix}{x:8.3f}{y:8.3f}{z:8.3f}{suffix}")
            atom_idx += 1
        elif line.startswith("MODEL") or line.startswith("ENDMDL"):
            pass # We write explicit MODEL header
        else:
            f.write(line)
    f.write("ENDMDL\n")

def main():
    in_pdb = Path("root_archive/demo/refined_structures_23_fixed.pdb")
    out_pdb = Path("demo/aligned_final.pdb")
    
    print(f"Reading {in_pdb}...")
    models_lines = parse_pdb_models(in_pdb)
    print(f"Found {len(models_lines)} models.")
    
    print("Aligning...")
    with open(out_pdb, 'w') as f:
        for i, m_lines in enumerate(models_lines):
            coords = get_coords(m_lines)
            aligned = align_coords(coords)
            write_model(f, m_lines, aligned, i+1)
            
    print(f"Saved {out_pdb}")

if __name__ == "__main__":
    main()
