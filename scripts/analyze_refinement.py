import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_dihedral(p1, p2, p3, p4):
    """
    Compute dihedral angle between 4 points (batch mode).
    Inputs: [B, 3]
    Output: [B] (radians)
    """
    b0 = -1.0*(p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= torch.norm(b1, dim=1, keepdim=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 - (b0 dot b1)*b1
    v = b0 - torch.sum(b0 * b1, dim=1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=1, keepdim=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v * w, dim=1)
    y = torch.sum(torch.cross(b1, v, dim=1) * w, dim=1)

    return torch.atan2(y, x)

def main():
    path = "runs/loop_b_final/refined_samples.pt"
    if not Path(path).exists():
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    coords = torch.load(path) # [B, 22, 3]
    if coords.dim() == 2:
         # Maybe just one sample?
         coords = coords.unsqueeze(0)
    
    print(f"Shape: {coords.shape}")
    
    # Indices for Alanine Dipeptide (based on PDB analysis)
    # Phi: C(prev)-N-CA-C  => 4, 6, 8, 14
    # Psi: N-CA-C-N(next)  => 6, 8, 14, 16
    
    phi_indices = [4, 6, 8, 14]
    psi_indices = [6, 8, 14, 16]
    
    p = coords
    
    # Calculate Phi
    phi = compute_dihedral(
        p[:, phi_indices[0]], 
        p[:, phi_indices[1]], 
        p[:, phi_indices[2]], 
        p[:, phi_indices[3]]
    )
    
    # Calculate Psi
    psi = compute_dihedral(
        p[:, psi_indices[0]], 
        p[:, psi_indices[1]], 
        p[:, psi_indices[2]], 
        p[:, psi_indices[3]]
    )
    
    phi_deg = np.degrees(phi.numpy())
    psi_deg = np.degrees(psi.numpy())
    
    # Metrics
    in_alpha = ((phi_deg > -100) & (phi_deg < -30) & (psi_deg > -70) & (psi_deg < -10))
    in_beta = ((phi_deg > -180) & (phi_deg < -100) & (psi_deg > 90) & (psi_deg < 180)) | \
              ((phi_deg > -180) & (phi_deg < -100) & (psi_deg > -180) & (psi_deg < -140))
              
    print(f"Samples: {len(phi_deg)}")
    print(f"In Alpha Basin: {np.sum(in_alpha)} ({100*np.mean(in_alpha):.2f}%)")
    print(f"In Beta Basin:  {np.sum(in_beta)}  ({100*np.mean(in_beta):.2f}%)")
    print(f"Total Valid:    {100*(np.mean(in_alpha) + np.mean(in_beta)):.2f}%")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist2d(phi_deg, psi_deg, bins=100, range=[[-180, 180], [-180, 180]], cmap='viridis', cmin=1)
    plt.colorbar(label='Count')
    plt.title(f"Refinement (Loop B) Ramachandran Plot\nN={len(phi_deg)}")
    plt.xlabel('Phi (degrees)')
    plt.ylabel('Psi (degrees)')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.grid(alpha=0.3)
    
    out_img = "runs/loop_b_final/refined_ramachandran.png"
    plt.savefig(out_img, dpi=150)
    print(f"Saved plot to {out_img}")
    
    # Save raw values
    raw_df = np.stack([phi_deg, psi_deg], axis=1)
    np.savetxt("runs/loop_b_final/refined_phi_psi.csv", raw_df, delimiter=",", header="phi,psi", comments="")
    print("Saved values to runs/loop_b_final/refined_phi_psi.csv")

if __name__ == "__main__":
    main()
