import torch
import numpy as np
from metastategen.utils import get_logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

log = get_logger("rdkit_refine")

def build_rdkit_molecule(topology_path: str):
    """
    Loads a PDB file into an RDKit molecule, preserving exact connectivity and hydrogens.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is not installed. Please install it to use rdkit refinement mode.")
        
    import mdtraj as md
    
    # 1. Load with mdtraj (which is much more robust to non-standard names)
    try:
        traj = md.load(topology_path)
        
        # 2. Force MDTraj to write a strictly compliant PDB string
        # RDKit needs the element column to be correct. MDTraj infers it from mass/name.
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_name = tmp.name
        
        traj.save_pdb(tmp_name)
        
        # 3. Load the cleaned PDB into RDKit
        mol = Chem.MolFromPDBFile(tmp_name, removeHs=False, proximityBonding=False)
        if mol is None:
            log.warning(f"RDKit standard load failed on cleaned PDB. Attempting with proximityBonding=True.")
            mol = Chem.MolFromPDBFile(tmp_name, removeHs=False, proximityBonding=True)
            
        os.remove(tmp_name)
    except Exception as e:
        log.error(f"Failed to clean PDB with MDTraj: {e}")
        mol = None

    if mol is None:
        raise ValueError(f"RDKit failed to load molecule from {topology_path}")
    return mol

def rdkit_reconstruct_and_refine(
    x_gen: torch.Tensor,
    mol,
    heavy_indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a batch of generated heavy atom coordinates [B, N_heavy, 3] (in nm),
    reconstructs the hydrogens and relaxes the structure using RDKit's UFF.
    
    Args:
        x_gen: Tensor of generated heavy atom positions [B, N_heavy, 3] in nm.
        mol: RDKit molecule object (must match the topology of x_gen).
        heavy_indices: List of atom indices corresponding to the heavy atoms.
        
    Returns:
        x_recon: Tensor [B, N_total, 3] in nm (After hydrogen reconstruction with fixed backbone)
        x_refined: Tensor [B, N_total, 3] in nm (After full relaxation)
    """
    B = x_gen.shape[0]
    N_total = mol.GetNumAtoms()
    device = x_gen.device
    
    # RDKit operates in Angstroms, so we convert nm to A
    x_gen_A = (x_gen * 10.0).detach().cpu().numpy()
    
    # Store results
    x_recon_list = []
    x_refined_list = []
    
    for i in range(B):
        # Create a working copy of the molecule for this sample
        m_curr = Chem.Mol(mol)
        conf = m_curr.GetConformer()
        
        # 1. Update heavy atom positions
        for local_idx, global_idx in enumerate(heavy_indices):
            pos = x_gen_A[i, local_idx]
            conf.SetAtomPosition(global_idx, (float(pos[0]), float(pos[1]), float(pos[2])))
            
        # Optional: Sanitize to ensure valences are happy before FF
        try:
            Chem.SanitizeMol(m_curr)
        except Exception as e:
            log.warning(f"RDKit SanitizeMol failed for sample {i}: {e}. Proceeding anyway.")
            
        # 2. Hydrogen Reconstruction (Fix Heavy Atoms, Minimize H)
        try:
            # UFF setup
            ff_recon = AllChem.UFFGetMoleculeForceField(m_curr, confId=0)
            if ff_recon is None:
                raise ValueError("UFF parameters could not be assigned.")
                
            # Fix all heavy atoms
            for global_idx in heavy_indices:
                ff_recon.AddFixedPoint(global_idx)
                
            # Minimize (max 1000 iter should be plenty for H placement)
            ff_recon.Minimize(maxIts=1000)
            
            # Extract coordinates after H-reconstruction
            pos_recon = np.array([list(conf.GetAtomPosition(idx)) for idx in range(N_total)])
            x_recon_list.append(torch.tensor(pos_recon, dtype=torch.float32) / 10.0)
            
        except Exception as e:
            log.warning(f"RDKit H-reconstruction failed for sample {i}: {e}. Skipping refinement.")
            pos_recon = np.array([list(conf.GetAtomPosition(idx)) for idx in range(N_total)])
            x_recon_list.append(torch.tensor(pos_recon, dtype=torch.float32) / 10.0)
            x_refined_list.append(torch.tensor(pos_recon, dtype=torch.float32) / 10.0)
            continue
            
        # 3. Full Relaxation (No fixed atoms)
        try:
            ff_refine = AllChem.UFFGetMoleculeForceField(m_curr, confId=0)
            if ff_refine is not None:
                ff_refine.Minimize(maxIts=2000)
            
            pos_refined = np.array([list(conf.GetAtomPosition(idx)) for idx in range(N_total)])
            x_refined_list.append(torch.tensor(pos_refined, dtype=torch.float32) / 10.0)
            
        except Exception as e:
            log.warning(f"RDKit final relaxation failed for sample {i}: {e}.")
            # Fallback to recon positions
            x_refined_list.append(x_recon_list[-1])
            
    x_recon = torch.stack(x_recon_list).to(device)
    x_refined = torch.stack(x_refined_list).to(device)
    
    return x_recon, x_refined
