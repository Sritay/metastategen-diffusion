from pathlib import Path
from typing import Dict, List, Tuple
from metastategen.utils import get_logger

log = get_logger("pdb_utils")

def get_ala2_heavy_atom_indices(pdb_path: Path) -> Tuple[List[int], List[int]]:
    """
    Parses PDB to find heavy atom indices for Phi and Psi.
    Assumes heavy atoms only are indexed 0..N-1 in the order they appear in PDB.
    
    Phi: C(prev) - N - CA - C
    Psi: N - CA - C - N(next)
    
    Returns: (phi_indices_4, psi_indices_4)
    """
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    heavy_atoms = []
    idx_counter = 0
    
    # Read PDB and map "ResidueName-AtomName" to index
    # We only increment index for heavy atoms (not H)
    
    atom_map = {} # Key: (residue_seq, atom_name), Value: index
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_seq = int(line[22:26])
                element = line[76:78].strip()
                
                # Crude element check
                if not element:
                     element = atom_name[0]
                
                if element.upper().startswith('H'):
                    continue
                    
                atom_map[(res_seq, atom_name)] = idx_counter
                heavy_atoms.append((res_seq, res_name, atom_name))
                idx_counter += 1
                
    log.info(f"Found {len(heavy_atoms)} heavy atoms in PDB.")

    # Hardcoded logic for ACE(1) - ALA(2) - NME(3)
    # Adjust residue numbers if your PDB is different
    # Standard Ala2: Res 1 (ACE), Res 2 (ALA), Res 3 (NME)
    
    try:
        # Phi: C(ACE) - N(ALA) - CA(ALA) - C(ALA)
        # Note: ACE C often named C.
        phi_atoms = [
            (1, 'C'), 
            (2, 'N'),
            (2, 'CA'),
            (2, 'C')
        ]
        
        # Psi: N(ALA) - CA(ALA) - C(ALA) - N(NME)
        psi_atoms = [
            (2, 'N'),
            (2, 'CA'),
            (2, 'C'),
            (3, 'N')
        ]
        
        phi_idx = [atom_map[k] for k in phi_atoms]
        psi_idx = [atom_map[k] for k in psi_atoms]
        
        log.info(f"Phi indices: {phi_idx}")
        log.info(f"Psi indices: {psi_idx}")
        
        return phi_idx, psi_idx
        
    except KeyError as e:
        log.error(f"Could not find atom {e} in PDB heavy atoms. Available: {atom_map.keys()}")
        raise
