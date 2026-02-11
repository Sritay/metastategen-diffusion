from metastategen.data.topology import MoleculeTopology
import networkx as nx

def verify_l0():
    print("Loading L0...")
    topo = MoleculeTopology("data/miscanthus/L0.pdb", topology_path="data/miscanthus/L0.psf")
    
    print(f"Atoms: {topo.n_atoms}")
    print(f"Heavy Atoms: {topo.n_heavy_atoms}")
    
    # Rings
    rings = topo._infer_rings()
    print(f"Rings detected: {len(rings)}")
    for i, r in enumerate(rings):
        print(f"  Ring {i}: {r}")
        
    # Constraints
    constraints = topo.infer_constraints()
    print(f"Total Constraints: {len(constraints)}")
    
if __name__ == "__main__":
    verify_l0()
