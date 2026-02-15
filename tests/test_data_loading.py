import mdtraj as md
import numpy as np
import torch
import pytest
from pathlib import Path
from metastategen.data.manager import load_training_data

# Mock data creation helpers
def create_dummy_pdb(path):
    # simple 3-atom ALA
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("ALA", chain)
    top.add_atom("C", md.element.carbon, res)
    top.add_atom("O", md.element.oxygen, res)
    top.add_atom("N", md.element.nitrogen, res)
    
    xyz = np.array([[[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.15, 0.0]]])
    t = md.Trajectory(xyz, top)
    t.save(str(path))
    return path

def create_lammps_dump(path):
    # LAMMPS dump format
    with open(path, 'w') as f:
        f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n-5 5\n-5 5\n-5 5\nITEM: ATOMS id type x y z\n")
        f.write("1 1 0.0 0.0 0.0\n")
        f.write("2 1 1.5 0.0 0.0\n") # Angstroms (1.5 A = 0.15 nm)
        f.write("3 1 0.0 1.5 0.0\n")

def test_load_pdb_single_frame(tmp_path):
    """Test loading a PDB as both topology and trajectory (1 frame fallback)."""
    pdb_path = tmp_path / "test.pdb"
    create_dummy_pdb(pdb_path)
    
    # Load with only pdb_path provided for both logic
    # In new signature: load_training_data(traj_path=None, topo_path=pdb)
    data = load_training_data(traj_path=None, topo_path=pdb_path)
    
    assert "positions" in data
    assert "atom_types" in data
    assert data["positions"].shape == (1, 3, 3) # 1 frame, 3 atoms
    
def test_load_pdb_explicit_traj(tmp_path):
    """Test loading PDB explicitly as trajectory."""
    pdb_path = tmp_path / "test.pdb"
    create_dummy_pdb(pdb_path)
    
    data = load_training_data(traj_path=pdb_path, topo_path=pdb_path)
    assert data["positions"].shape == (1, 3, 3)

def test_load_lammps_dump(tmp_path):
    """Test loading LAMMPS dump with PDB topology."""
    pdb_path = tmp_path / "topo.pdb"
    create_dummy_pdb(pdb_path)
    
    dump_path = tmp_path / "traj.lammpstrj"
    create_lammps_dump(dump_path)
    
    data = load_training_data(traj_path=dump_path, topo_path=pdb_path)
    
    # MDTraj loading of .lammpstrj usually returns Angstroms converted to nm automatically?
    # Or does it depend? Standard mdtraj behavior is 1 unit = 1 nm usually.
    # Let's check shape first.
    assert data["positions"].shape == (1, 3, 3)
    
def test_load_npz_legacy(tmp_path):
    """Test legacy NPZ loading behavior."""
    pdb_path = tmp_path / "topo.pdb"
    create_dummy_pdb(pdb_path)
    
    npz_path = tmp_path / "data.npz"
    pos = np.random.randn(10, 3, 3).astype(np.float32)
    np.savez(npz_path, positions=pos)
    
    data = load_training_data(traj_path=npz_path, topo_path=pdb_path)
    assert data["positions"].shape == (10, 3, 3)
