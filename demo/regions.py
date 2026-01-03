import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from metastategen.utils.geometry import compute_dihedrals, rad2deg
from metastategen.utils.pdb import get_ala2_heavy_atom_indices
from scipy.ndimage import gaussian_filter

# Centers for Labeling (kept from literature/approx)
LABELS = {
    "C7eq": {"x": -75, "y": 70, "color": "white"},
    "Alpha_R": {"x": -70, "y": -45, "color": "white"},
    "Beta": {"x": -140, "y": 150, "color": "white"},
    "Alpha_L": {"x": 60, "y": 50, "color": "white"},
    "C7ax": {"x": 60, "y": -50, "color": "white"},
}

# Cache for ground truth data to avoid reloading every time
_GT_DATA = None

def get_ground_truth_data():
    global _GT_DATA
    if _GT_DATA is not None:
        return _GT_DATA
        
    # Load Validation Set from Loop 5 (covers all basins)
    val_path = Path("data/processed/ala2/split_5/al_val.pt")
    pdb_path = Path("data/raw/alanine-dipeptide-nowater.pdb")
    
    if not val_path.exists():
        print(f"Warning: Ground truth file {val_path} not found.")
        return None
        
    samples = torch.load(val_path, map_location='cpu')
    if isinstance(samples, dict):
        if 'positions' in samples:
            samples = samples['positions']
        elif 'pos' in samples:
            samples = samples['pos']
        else:
            print(f"Warning: Unexpected dict keys: {samples.keys()}")
            return None
        
    phi_idx, psi_idx = get_ala2_heavy_atom_indices(pdb_path)
    indices = torch.tensor([phi_idx, psi_idx], dtype=torch.long)
    rads = compute_dihedrals(samples, indices)
    degs = rad2deg(rads)
    degs = (degs + 180.0) % 360.0 - 180.0
    
    _GT_DATA = degs.numpy()
    return _GT_DATA

def plot_regions(ax):
    """Overlays Ground Truth Contours and Labels."""
    data = get_ground_truth_data()
    
    if data is not None:
        bins = 100
        H, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins, range=[[-180, 180], [-180, 180]])
        H = gaussian_filter(H, sigma=1.5)
        
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(xcenters, ycenters)
        
        # Filled Contours (Colored Background)
        # zorder=0 places this BEHIND the main plot data (which usually has zorder=1+)
        levels = np.linspace(H.min(), H.max(), 10)
        
        # Opaque-ish colored contours
        ax.contourf(X, Y, H.T, levels=levels, cmap='viridis', alpha=0.7, antialiased=True, zorder=-1)
        
        # Add a thin contour line for sharper definition
        ax.contour(X, Y, H.T, levels=levels[1::2], colors='white', linewidths=0.5, alpha=0.5, zorder=-1)
        
    # Plot Text Labels (Always)
    for name, p in LABELS.items():
        ax.text(
            p['x'], p['y'], 
            name, 
            color='black',
            ha='center', va='center', 
            fontsize=9, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
        )
