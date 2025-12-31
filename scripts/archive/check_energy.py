import mdshare
from pathlib import Path

def main():
    print("Checking for ALA2 energy data...")
    try:
        # Common name for this dataset
        filename = "alanine-dipeptide-3x250ns-energies.npz"
        print(f"Attempting to fetch {filename}...")
        local = mdshare.fetch(filename, working_directory="data/raw")
        print(f"Success! Found at: {local}")
        
        # Verify content briefly
        import numpy as np
        with np.load(local) as f:
            print(f"Keys: {list(f.keys())}")
            for k in f.keys():
                print(f"{k}: {f[k].shape}")
                
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
