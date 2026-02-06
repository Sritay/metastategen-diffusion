from __future__ import annotations
import mdtraj as md
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from metastategen.utils import get_logger

log = get_logger("topology")

class MoleculeTopology:
    """
    Handles molecule topology inference from file.
    Generalizes beyond Alanine Dipeptide.
    """
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Topology file not found: {file_path}")
            
        try:
            self.traj = md.load(str(file_path))
        except Exception as e:
            log.error(f"Failed to load topology from {file_path}: {e}")
            raise

        self.top = self.traj.topology
        self._heavy_indices = None
        self._constraints = None
        self._atom_types = None
        
        log.info(f"Loaded topology for {self.file_path.name}. Atoms: {self.top.n_atoms}, Residues: {self.top.n_residues}")

    @property
    def heavy_indices(self) -> List[int]:
        """Returns indices of non-hydrogen atoms."""
        if self._heavy_indices is None:
            self._heavy_indices = [
                atom.index for atom in self.top.atoms 
                if atom.element.symbol != "H"
            ]
        return self._heavy_indices
    
    @property
    def n_atoms(self) -> int:
        """Total number of atoms in the full structure."""
        return self.top.n_atoms
    
    @property
    def n_heavy_atoms(self) -> int:
        """Number of heavy atoms."""
        return len(self.heavy_indices)

    def get_atom_types(self) -> torch.Tensor:
        """
        Returns atom types (integers) for the heavy atoms only.
        Mapping should be consistent with the diffusion model's vocabulary.
        Simple mapping: H=1, C=6, N=7, O=8, S=16
        """
        if self._atom_types is None:
            types = []
            for idx in self.heavy_indices:
                elem = self.top.atom(idx).element.symbol
                if elem == "C": types.append(0) # Standardize these indices later?
                elif elem == "N": types.append(1)
                elif elem == "O": types.append(2)
                elif elem == "S": types.append(3)
                else: types.append(4) # Other
            self._atom_types = torch.tensor(types, dtype=torch.long)
        return self._atom_types

    def infer_constraints(self) -> torch.Tensor:
        """
        Infers bond length constraints for the heavy atom backbone.
        Returns:
            Tensor [K, 3] where each row is (atom_i, atom_j, length_nm)
            Indices are RELATIVE to the heavy_indices list (0..N_heavy-1).
        """
        if self._constraints is not None:
            return self._constraints

        heavy_indices = self.heavy_indices
        mapping = {original: new for new, original in enumerate(heavy_indices)}
        
        constraints = []
        
        # Iterate over all bonds in topology
        for bond in self.top.bonds:
            a1, a2 = bond.atom1.index, bond.atom2.index
            
            # Keep only bonds between heavy atoms
            if a1 in mapping and a2 in mapping:
                idx1 = mapping[a1]
                idx2 = mapping[a2]
                
                # Calculate equilibrium length from the reference frame
                xyz = self.traj.xyz[0] # [N_total, 3] in nm
                dist = np.linalg.norm(xyz[a1] - xyz[a2])
                
                constraints.append([idx1, idx2, dist])
                
        self._constraints = torch.tensor(constraints, dtype=torch.float32)
        log.info(f"Inferred {len(constraints)} bond constraints for {self.n_heavy_atoms} heavy atoms.")
        return self._constraints

    def get_template_structure(self) -> torch.Tensor:
        """Returns the full template structure [Total_Atoms, 3]"""
        return torch.tensor(self.traj.xyz[0], dtype=torch.float32)

    def infer_torsions(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Infers Phi and Psi torsion indices using MDTraj.
        Returns:
            (phi_indices, psi_indices)
            Each is a list of lists, where each inner list is [a, b, c, d] atom INDICES.
            Indices are mapped to the HEAVY atom subset (0..N_heavy-1).
        """
        heavy_to_subset = {original: i for i, original in enumerate(self.heavy_indices)}

        try:
            phi_ind, _ = md.compute_phi(self.traj)
            psi_ind, _ = md.compute_psi(self.traj)
            
            # Filter and map
            mapped_phi = []
            for quad in phi_ind:
                if all(idx in heavy_to_subset for idx in quad):
                    mapped_phi.append([heavy_to_subset[idx] for idx in quad])
            
            mapped_psi = []
            for quad in psi_ind:
                if all(idx in heavy_to_subset for idx in quad):
                    mapped_psi.append([heavy_to_subset[idx] for idx in quad])

            return mapped_phi, mapped_psi
        except Exception as e:
            log.warning(f"Could not infer protein torsions (Phi/Psi): {e}")
            return [], []

    def infer_chirality_config(self) -> List[Dict]:
        """
        Infers chiral centers configuration for Amino Acid residues.
        Target: CA atoms with neighbors (N, C, CB).
        Returns a list of configs with indices mapped to HEAVY atom subset.
        """
        config = []
        heavy_to_subset = {original: i for i, original in enumerate(self.heavy_indices)}
        
        for res in self.top.residues:
            names = {a.name: a.index for a in res.atoms}
            
            if "CA" in names and "N" in names and "C" in names and "CB" in names:
                idx_CA = names["CA"]
                idx_N = names["N"]
                idx_C = names["C"]
                idx_CB = names["CB"]

                # Ensure all are heavy (they should be, but good to check)
                if (idx_CA in heavy_to_subset and idx_N in heavy_to_subset and 
                    idx_C in heavy_to_subset and idx_CB in heavy_to_subset):
                    
                    config.append({
                        "center_idx": heavy_to_subset[idx_CA],
                        "neighbors": [
                            heavy_to_subset[idx_N], 
                            heavy_to_subset[idx_CB], 
                            heavy_to_subset[idx_C]
                        ], # For features.py (N, CB, C)
                        "chiral_plane": [
                            heavy_to_subset[idx_N],
                            heavy_to_subset[idx_C]
                        ], # For diffusion.py (Plane definition)
                        "chiral_target": heavy_to_subset[idx_CB], # For diffusion.py (Atom to move)
                        "expected_sign": 1.0 
                    })
        
        log.info(f"Inferred {len(config)} chiral centers (CA with CB).")
        return config

