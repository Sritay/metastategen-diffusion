import numpy as np
import glob
import os

def check_raw_npz():
    files = glob.glob("data/timewarp/train/*.npz")
    if not files:
        print("No raw timewarp files found.")
        return

    f = files[0]
    print(f"Loading {f}...")
    d = np.load(f)
    pos = d['positions']
    frc = d['forces']
    ene = d['energies'][:, 0] # Assume col 0 is Potential Energy

    print(f"Pos shape: {pos.shape}, Min/Max: {pos.min():.2f}/{pos.max():.2f}")
    print(f"Frc shape: {frc.shape}, Min/Max: {frc.min():.2f}/{frc.max():.2f}")
    print(f"Ene shape: {ene.shape}, Min/Max: {ene.min():.2f}/{ene.max():.2f}")

    # Check physics
    # dE ~ -F * dx
    # Using adjacent frames? Timewarp might not be sequential frames, but "samples".
    # If they are independent samples, dE vs F.dx is meaningless between frames.
    # We need to know if it's a trajectory.
    # Usually 'positions' in npz are sequential.
    
    # Check both columns
    for col in [0, 1]:
        print(f"\n--- Checking Energy Column {col} ---")
        ene = d['energies'][:, col]
        
        dx = pos[1:] - pos[:-1]
        de = ene[1:] - ene[:-1]
        f_avg = (frc[1:] + frc[:-1]) / 2
        
        w = (f_avg * dx).sum(axis=(1, 2))
        dot = -w
        
        # Filter small steps
        dx_norm = np.linalg.norm(dx, axis=-1).max(axis=-1)
        mask = dx_norm < 0.2 
        
        valid_de = de[mask]
        valid_dot = dot[mask]
        
        if len(valid_de) > 0:
            # Correlation
            corr = np.corrcoef(valid_de, valid_dot)[0, 1]
            print(f"Correlation: {corr:.4f}")
            
            # Regression Slope
            slope_reg = np.mean(valid_de * valid_dot) / np.mean(valid_dot**2)
            print(f"Regression Slope (dE = s * -Fdx): {slope_reg:.4f}")
            
            # Ratio Mean
            ratio = valid_de / (valid_dot + 1e-9)
            slope_mean = np.mean(ratio)
            print(f"Mean Ratio: {slope_mean:.4f}")
        else:
            print("No valid steps.")

if __name__ == "__main__":
    check_raw_npz()
