import numpy as np
import glob

def main():
    files = glob.glob("data/timewarp/**/*.npz", recursive=True)
    if not files: return
    
    f = files[0]
    print(f"Loading {f}...")
    d = np.load(f)
    E = d['energies'] # [T, 2]
    
    print(f"Shape: {E.shape}")
    
    col0 = E[:, 0]
    col1 = E[:, 1]
    
    print(f"Col 0: Mean={col0.mean():.2f}, Std={col0.std():.2f}, Min={col0.min():.2f}, Max={col0.max():.2f}")
    print(f"Col 1: Mean={col1.mean():.2f}, Std={col1.std():.2f}, Min={col1.min():.2f}, Max={col1.max():.2f}")
    
    # Heuristic: Potential energy is typically negative (for stable systems) or at least consistently offset.
    # Kinetic energy (at 300K) is positive approx 3/2 N kT.
    # N=22 atoms. KE ~ 22 * 1.5 * 2.5 kJ/mol ~ 82 kJ/mol.
    
if __name__ == "__main__":
    main()
