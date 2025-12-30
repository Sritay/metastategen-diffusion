import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load data
    try:
        data = np.loadtxt("runs/day8_9_al_3/iter_03/phi_psi_values.csv", delimiter=",", skiprows=1)
    except OSError:
        # Fallback to the artifact path if the run path isn't accessible directly in this env context
        data = np.loadtxt("/Users/sritaymistry/.gemini/antigravity/brain/f3fbf2ab-df00-4397-a351-31c5ce7b207e/phi_psi_values.csv", delimiter=",", skiprows=1)

    phi = data[:, 0]
    psi = data[:, 1]
    
    # Calculate radial distance
    r = np.sqrt(phi**2 + psi**2)
    
    # Define bins (e.g., every 10 degrees)
    bin_width = 10
    bins = np.arange(0, 190, bin_width)
    
    print(f"Radial Density Analysis around (0,0)")
    print(f"{'Bin Range (deg)':<20} | {'Count':<6} | {'Density (points/deg^2)':<20}")
    print("-" * 60)
    
    densities = []
    centers = []

    for i in range(len(bins) - 1):
        r_inner = bins[i]
        r_outer = bins[i+1]
        
        mask = (r >= r_inner) & (r < r_outer)
        count = np.sum(mask)
        
        # Area of annulus = pi * (r_outer^2 - r_inner^2)
        area = np.pi * (r_outer**2 - r_inner**2)
        
        # Density
        density = count / area if area > 0 else 0
        densities.append(density)
        centers.append((r_inner + r_outer) / 2)
        
        print(f"{r_inner:3d} - {r_outer:3d}            | {count:6d} | {density:.6f}")

    # Check for peak near 0
    # If the density in the first bin (0-10) is significantly higher than subsequent bins,
    # it indicates clustering at the origin.
    
    first_bin_density = densities[0]
    mean_background_density = np.mean(densities[4:]) # Compare to background (e.g. > 40 deg)
    
    print("-" * 60)
    print(f"Density (0-10): {first_bin_density:.6f}")
    print(f"Mean Density (>40): {mean_background_density:.6f}")
    
    ratio = first_bin_density / (mean_background_density + 1e-9)
    print(f"Ratio (Peak/Background): {ratio:.2f}")
    
    if ratio > 5.0:
        print("\nCONCLUSION: Significant clustering detected around (0,0).")
    elif ratio > 2.0:
        print("\nCONCLUSION: Mild clustering detected around (0,0).")
    else:
        print("\nCONCLUSION: No significant clustering around (0,0). Distribution is diffuse away from origin.")

if __name__ == "__main__":
    main()
