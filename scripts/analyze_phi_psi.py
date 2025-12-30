import numpy as np

def main():
    data = np.loadtxt("runs/day8_9_al_3/iter_03/phi_psi_values.csv", delimiter=",", skiprows=1)
    phi = data[:, 0]
    psi = data[:, 1]
    
    print(f"Total samples: {len(data)}")
    print(f"Phi Mean: {np.mean(phi):.2f}, Std: {np.std(phi):.2f}")
    print(f"Psi Mean: {np.mean(psi):.2f}, Std: {np.std(psi):.2f}")
    
    # Check for Beta Sheet Region: Phi ~ -120 to -60, Psi ~ 120 to 180 (and -180)
    beta_mask = (phi > -150) & (phi < -50) & ((psi > 90) | (psi < -170))
    beta_count = np.sum(beta_mask)
    
    # Check for Alpha Helix Region: Phi ~ -90 to -50, Psi ~ -60 to -40
    alpha_mask = (phi > -100) & (phi < -50) & (psi > -70) & (psi < -30)
    alpha_count = np.sum(alpha_mask)

    # Check for "Collapse" (near 0,0)
    collapse_mask = (np.abs(phi) < 30) & (np.abs(psi) < 30)
    collapse_count = np.sum(collapse_mask)

    print(f"Beta Sheet-like count: {beta_count} ({beta_count/len(data)*100:.1f}%)")
    print(f"Alpha Helix-like count: {alpha_count} ({alpha_count/len(data)*100:.1f}%)")
    print(f"Near (0,0) count: {collapse_count} ({collapse_count/len(data)*100:.1f}%)")

if __name__ == "__main__":
    main()
