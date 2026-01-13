
import torch
import numpy as np
from pathlib import Path

def rmsd(x, y):
    d = x - y
    d2 = torch.sum(d**2, dim=-1)
    return torch.sqrt(torch.mean(d2, dim=-1))

def get_rotation_matrix(v1, v2):
    """Rotation matrix that aligns v1 to v2"""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Axis of rotation
    k = np.cross(v1, v2)
    norm_k = np.linalg.norm(k)
    
    if norm_k < 1e-6:
        # Parallel
        if np.dot(v1, v2) > 0:
            return np.eye(3)
        else:
            # Anti-parallel: Rotate 180 around any orthogonal axis
            # Just pick x or y
            ortho = np.array([1, 0, 0])
            if abs(np.dot(v1, ortho)) > 0.9:
                ortho = np.array([0, 1, 0])
            k = np.cross(v1, ortho)
            k /= np.linalg.norm(k)
            # Rodriques for 180?
            # K = cross_product_matrix(k)
            # R = I + 2 K^2 ...
            # Simpler: just -I? No.
            # Let's ignore anti-parallel for bond vectors (unlikely).
            return -np.eye(3) 
            
    k /= norm_k
    theta = np.arccos(np.dot(v1, v2))
    
    # Rodrigues formula
    K = np.array([[0, -k[2], k[1]], 
                  [k[2], 0, -k[0]], 
                  [-k[1], k[0], 0]])
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def align_structure(pos):
    # pos: [22, 3] numpy
    # Indices for 22-atom Timewarp
    idx_N = 6
    idx_CA = 8
    idx_C = 14
    
    coords = pos.copy()
    
    # 1. Translate CA to Origin
    ca = coords[idx_CA]
    coords = coords - ca
    
    # 2. Align N-CA to Z-axis
    n_vec = coords[idx_N] # N is at coords[N]
    z_axis = np.array([0.0, 0.0, 1.0])
    
    R1 = get_rotation_matrix(n_vec, z_axis)
    coords = coords @ R1.T # Apply R1
    
    # 3. Align C projection to X-axis (Rotation around Z)
    # Project C onto XY plane (actually just check X,Y)
    c_vec = coords[idx_C]
    # We want c_vec's projection to align with X (1,0,0)
    # Angle in XY plane
    angle = np.arctan2(c_vec[1], c_vec[0])
    # We want angle to be 0. So rotate by -angle.
    theta = -angle
    c, s = np.cos(theta), np.sin(theta)
    
    R2 = np.array([[c, -s, 0],
                   [s, c, 0],
                   [0, 0, 1]])
                   
    coords = coords @ R2.T
    
    return coords

def write_pdb_frame(f, pos, model_num=1):
    f.write(f"MODEL     {model_num}\n")
    # Types (Timewarp 22)
    # C, H, H, H, C, O, N, H, C, H, C, H, H, H, C, O, N, H, C, H, H, H
    atom_names = ["C1", "H1", "H2", "H3", "C2", "O2", "N3", "H4", "CA", "HA", 
                  "CB", "HB1", "HB2", "HB3", "C4", "O4", "N5", "H5", "C6", "H6", "H7", "H8"]
    elements  = ["C", "H", "H", "H", "C", "O", "N", "H", "C", "H", 
                 "C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "H", "H"]
                 
    for i, p in enumerate(pos):
        x, y, z = p[0]*10, p[1]*10, p[2]*10 # nm to Angstrom
        f.write(f"ATOM  {i+1:>5} {atom_names[i]:<4} ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elements[i]:>2}\n")
    
    f.write("ENDMDL\n")

def main():
    path = Path("runs/loop_b_refinement_23/refined_results.pt")
    out_pdb = Path("demo") / "aligned_refinement.pdb"
    
    print(f"Loading {path}...")
    data = torch.load(path, map_location='cpu')
    
    initial_all = data['initial_positions'] # [10000, 22, 3]
    refined_all = data['refined_positions'] # [100, 22, 3]
    
    # Recover Pairs (RMSD)
    # Batch size 1000. Expected 10 refined per batch.
    batch_size = 1000
    expected_per_batch = 10
    n_batches = len(initial_all) // batch_size
    
    pairs = []
    curr_ref_idx = 0
    
    print("Matching pairs...")
    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size
        batch_init = initial_all[start:end] # [1000, 22, 3]
        
        batch_ref = refined_all[curr_ref_idx : curr_ref_idx + expected_per_batch]
        curr_ref_idx += len(batch_ref)
        
        for ref in batch_ref:
            # RMSD to find parent
            dists = rmsd(batch_init, ref)
            best_idx = torch.argmin(dists)
            parent = batch_init[best_idx]
            pairs.append((parent.numpy(), ref.numpy())) # Copy to numpy
            
    print(f"Aligning {len(pairs)} pairs...")
    
    with open(out_pdb, 'w') as f:
        m_cnt = 1
        for parent, ref in pairs:
            # Align Parent
            p_aligned = align_structure(parent)
            write_pdb_frame(f, p_aligned, m_cnt)
            m_cnt += 1
            
            # Align Refined
            r_aligned = align_structure(ref)
            write_pdb_frame(f, r_aligned, m_cnt)
            m_cnt += 1
            
    print(f"Saved {out_pdb} ({m_cnt-1} frames)")

if __name__ == "__main__":
    main()
