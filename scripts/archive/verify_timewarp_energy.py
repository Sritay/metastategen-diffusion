import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import os

# Define the implicit solvent setup as reversed from md.py
def create_system(pdb_path):
    pdb = app.PDBFile(pdb_path)
    # amber99-implicit setup from md.py matching 'amber99-implicit-old' preset
    forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    system = forcefield.createSystem(pdb.topology,
                                   nonbondedMethod=app.CutoffNonPeriodic,
                                   nonbondedCutoff=2.0*unit.nanometer,
                                   constraints=None)
    return system, pdb

def check_energy_match():
    data_dir = "data/timewarp/train"
    npz_path = os.path.join(data_dir, "ad1-traj-arrays.npz")
    pdb_path = os.path.join(data_dir, "ad1-traj-state0.pdb")
    
    print("Loading data...")
    data = np.load(npz_path)
    positions = data['positions'] # (T, 22, 3) in nm
    energies = data['energies']   # (T, 2) [PE, KE]
    
    print("Creating OpenMM system (Amber99SB-ILDN + OBC)...")
    try:
        system, pdb = create_system(pdb_path)
        integrator = mm.VerletIntegrator(0.001*unit.picoseconds)
        simulation = app.Simulation(pdb.topology, system, integrator)
        
        # Check a few random frames
        indices = [0, 100, 1000, 50000]
        for idx in indices:
            if idx >= len(positions): continue
            
            pos_frame = positions[idx]
            simulation.context.setPositions(pos_frame * unit.nanometer)
            state = simulation.context.getState(getEnergy=True)
            pe_omm = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            
            pe_data = energies[idx, 0]
            diff = abs(pe_omm - pe_data)
            
            print(f"Frame {idx}: Data PE={pe_data:.4f}, OpenMM PE={pe_omm:.4f}, Diff={diff:.4f}")
            if diff > 1.0:
                print("  WARNING: Large difference!")
            else:
                print("  Match confirmed.")
                
    except Exception as e:
        print(f"Error running OpenMM verification: {e}")

if __name__ == "__main__":
    check_energy_match()
