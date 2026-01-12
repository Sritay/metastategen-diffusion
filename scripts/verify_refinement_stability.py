import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from metastategen.models.pairwise import PairwiseEnergyModel
from metastategen.utils import set_deterministic, get_logger

log = get_logger("verify_stability")

def compute_dihedrals(pos, indices):
    """
    Computes batch dihedral angles.
    pos: [B, N, 3]
    indices: [4]
    """
    p0 = pos[:, indices[0]]
    p1 = pos[:, indices[1]]
    p2 = pos[:, indices[2]]
    p3 = pos[:, indices[3]]
    
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    
    n1 = torch.cross(b0, b1, dim=-1)
    n2 = torch.cross(b1, b2, dim=-1)
    
    b1_norm = torch.norm(b1, dim=-1, keepdim=True) + 1e-8
    u = b1 / b1_norm
    
    m1 = torch.cross(n1, n2, dim=-1)
    y = torch.sum(m1 * u, dim=-1)
    x = torch.sum(n1 * n2, dim=-1)
    
    return torch.atan2(y, x)

def load_pairwise_model(ckpt_path, device):
    model = PairwiseEnergyModel(n_atoms=22).to(device)
    stats = {}
    if ckpt_path.exists():
        log.info(f"Loading pairwise checkpoint from {ckpt_path}")
        d = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(d['model'])
        stats['e_mean'] = d['e_mean'].to(device)
        stats['e_std'] = d['e_std'].to(device)
        stats['f_std'] = d['f_std'].to(device)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return model, stats

def constrain_bonds_22(x):
    """
    Projects backbone bonds (N-CA, CA-C) to target lengths for 22-atom Timewarp structure.
    Indices: N=6, CA=8, C=14.
    """
    t1 = 0.146 # N-CA
    t2 = 0.151 # CA-C
    
    constraints = [
        (6, 8, t1),
        (8, 14, t2)
    ]
    
    for _ in range(5):
        for i1, i2, dist_target in constraints:
            p1 = x[:, i1]
            p2 = x[:, i2]
            diff = p2 - p1
            dist = torch.norm(diff, dim=1, keepdim=True) + 1e-8
            delta = diff * (dist_target / dist - 1.0)
            x[:, i1] -= 0.5 * delta
            x[:, i2] += 0.5 * delta
    return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_deterministic(42)
    
    # Paths
    pos_path = Path("data/timewarp/train/positions.pt")
    model_path = Path("runs/energy_pairwise/best_model.pt")
    out_dir = Path("runs/verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    log.info("Loading positions...")
    pos = torch.load(pos_path).to(device) # [N, 22, 3]
    
    # 2. Compute Dihedrals (Degrees)
    phi_idx = [4, 6, 8, 14]
    psi_idx = [6, 8, 14, 16]
    
    phi = torch.rad2deg(compute_dihedrals(pos, phi_idx))
    psi = torch.rad2deg(compute_dihedrals(pos, psi_idx))
    
    # 3. Select Frames
    # Alpha: Phi ~ -60, Psi ~ -45 (Tol 10)
    alpha_mask = (torch.abs(phi - (-60)) < 10) & (torch.abs(psi - (-45)) < 10)
    # Beta: Phi ~ -140, Psi ~ 150 (Tol 10)
    beta_mask = (torch.abs(phi - (-140)) < 10) & (torch.abs(psi - 150) < 10)
    
    alpha_indices = torch.where(alpha_mask)[0][:5]
    beta_indices = torch.where(beta_mask)[0][:5]
    
    selected_indices = torch.cat([alpha_indices, beta_indices])
    log.info(f"Selected {len(selected_indices)} frames (5 Alpha, 5 Beta)")
    
    x_curr = pos[selected_indices].clone().requires_grad_(True)
    
    # 4. Load Model
    model, stats = load_pairwise_model(model_path, device)
    model.eval()
    
    # 5. Run Refinement
    n_steps = 10000
    step_size = 1e-7
    energies = []
    phis = []
    psis = []
    
    log.info(f"Running {n_steps} refinement steps...")
    
    for k in range(n_steps):
        e_norm = model(x_curr)
        
        # Track metrics
        if k % 100 == 0:
            # Energy
            e_val = e_norm * stats['e_std'] + stats['e_mean']
            energies.append(e_val.detach().cpu().numpy())
            
            # Dihedrals
            with torch.no_grad():
                ph = torch.rad2deg(compute_dihedrals(x_curr, phi_idx)).cpu().numpy()
                ps = torch.rad2deg(compute_dihedrals(x_curr, psi_idx)).cpu().numpy()
                phis.append(ph)
                psis.append(ps)
            
        grad = torch.autograd.grad(e_norm.sum(), x_curr)[0]
        f_pred = -grad * stats['e_std']
        
        # Clip
        f_norm = f_pred.norm(dim=-1, keepdim=True)
        clip_coef = torch.clamp(10.0 / (f_norm + 1e-6), max=1.0)
        f_pred = f_pred * clip_coef
        
        # Update
        with torch.no_grad():
            x_curr.data += step_size * f_pred
            x_curr.data = constrain_bonds_22(x_curr.data)
            
    energies = np.array(energies) # [Steps/100, B]
    phis = np.array(phis)         # [Steps/100, B]
    psis = np.array(psis)         # [Steps/100, B]
    
    # 6. Plotting Energy
    plt.figure(figsize=(10, 6))
    steps = np.arange(0, n_steps, 100)
    
    for i in range(energies.shape[1]):
        label = "Alpha" if i < 5 else "Beta"
        color = 'blue' if i < 5 else 'orange'
        plt.plot(steps, energies[:, i], label=label if (i==0 or i==5) else "", color=color, alpha=0.6)
        
    plt.xlabel("Step")
    plt.ylabel("PD (kJ/mol)")
    plt.title("Refinement Stability Check (Energy)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "energy_drift.png")
    log.info(f"Saved energy plot to {out_dir / 'energy_drift.png'}")

    # 7. Plotting Ramachandran Trajectories
    plt.figure(figsize=(8, 8))
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel("Phi (deg)")
    plt.ylabel("Psi (deg)")
    plt.title("Refinement Trajectories (Phi/Psi)")
    
    # Draw boxes for reference
    # Alpha ~ (-60, -45), Beta ~ (-140, 150)
    plt.scatter([-60], [-45], c='blue', marker='x', s=100, label='Alpha Center')
    plt.scatter([-140], [150], c='orange', marker='x', s=100, label='Beta Center')

    for i in range(phis.shape[1]):
        color = 'blue' if i < 5 else 'orange'
        # Plot full trajectory
        plt.plot(phis[:, i], psis[:, i], color=color, alpha=0.5, linewidth=1)
        # Mark start and end
        plt.scatter(phis[0, i], psis[0, i], c=color, marker='o', s=30) # Start
        plt.scatter(phis[-1, i], psis[-1, i], c=color, marker='>', s=50, edgecolors='black') # End
        
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "rama_drift.png")
    log.info(f"Saved Rama plot to {out_dir / 'rama_drift.png'}")
    
    # Analysis
    drift = energies[-1] - energies[0]
    phi_drift = phis[-1] - phis[0]
    psi_drift = psis[-1] - psis[0]
    
    log.info("--- Analysis ---")
    log.info("Energy Drift (Mean):")
    log.info(f"  Alpha: {drift[:5].mean():.2f}")
    log.info(f"  Beta:  {drift[5:].mean():.2f}")
    
    log.info("Angular Drift (Mean Abs deg):")
    log.info(f"  Alpha Phi: {np.abs(phi_drift[:5]).mean():.2f}, Psi: {np.abs(psi_drift[:5]).mean():.2f}")
    log.info(f"  Beta  Phi: {np.abs(phi_drift[5:]).mean():.2f}, Psi: {np.abs(psi_drift[5:]).mean():.2f}")


if __name__ == "__main__":
    main()
