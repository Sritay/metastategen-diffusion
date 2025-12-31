import numpy as np
import glob
from pathlib import Path

def main():
    # Look for any .npz in data/timewarp
    files = glob.glob("data/timewarp/**/*.npz", recursive=True)
    if not files:
        print("No .npz files found in data/timewarp")
        return

    f = files[0]
    print(f"Inspecting {f}...")
    try:
        d = np.load(f)
        print("Keys:", list(d.keys()))
        for k in d.keys():
            if k in ['potential_energy', 'energies', 'energy']:
                print(f"FOUND ENERGY KEY: {k}, Shape: {d[k].shape}")
            else:
                 print(f"  {k}: {d[k].shape}")
    except Exception as e:
        print(f"Error loading {f}: {e}")

if __name__ == "__main__":
    main()
