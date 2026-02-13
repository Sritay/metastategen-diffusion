import mdtraj as md
import numpy as np
import torch

def create_l0_npz():
    # Load L0 structure
    # Use PSF for topology to ensure atoms are correct?
    # Actually PDB should have coords.
    traj = md.load("data/miscanthus/L0.pdb")
    
    # Extract positions [1, N_atoms, 3]
    # mdtraj uses nanometers.
    positions = traj.xyz
    
    # Atom types map?
    # Note: load_npz_as_al_data re-infers atom types from PDB/Topology anyway.
    # But it expects 'atom_types' in NPZ? No, it loads from PDB.
    # It expects 'positions' or 'coords' in NPZ.
    
    # Filter heavy? 
    # load_npz_as_al_data logic:
    # 1. Load PDB to get Topology (and infer atom types for heavy atoms).
    # 2. Load NPZ to get positions. 
    # 3. If NPZ has all atoms, slice to heavy.
    
    # So we can just save all atoms positions.
    
    print(f"Saving L0.npz with shape {positions.shape}")
    np.savez("data/miscanthus/L0.npz", positions=positions)

if __name__ == "__main__":
    create_l0_npz()
