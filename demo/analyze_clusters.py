import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN
from regions import plot_regions

def get_cluster_centers(phi, psi, eps=0.4, min_samples=20, top_n=5):
    # Convert to circular coordinates
    phi_rad = np.radians(phi)
    psi_rad = np.radians(psi)
    
    X = np.column_stack([
        np.cos(phi_rad), np.sin(phi_rad),
        np.cos(psi_rad), np.sin(psi_rad)
    ])
    
    # DBSCAN
    scan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = scan.labels_
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
        
    clusters = []
    
    for lab in unique_labels:
        mask = labels == lab
        count = np.sum(mask)
        
        # Mean vector
        mean_vec = np.mean(X[mask], axis=0)
        mean_phi = np.degrees(np.arctan2(mean_vec[1], mean_vec[0]))
        mean_psi = np.degrees(np.arctan2(mean_vec[3], mean_vec[2]))
        
        clusters.append({
            "label": lab,
            "count": count,
            "phi": mean_phi,
            "psi": mean_psi
        })
        
    # Sort by count descending
    clusters.sort(key=lambda x: x["count"], reverse=True)
    return clusters[:top_n], labels

def main():
    csv_path = Path("demo/evolution_density.csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    iters = sorted(df['Iter'].unique())
    
    fig, axes = plt.subplots(1, len(iters), figsize=(6 * len(iters), 6), sharex=True, sharey=True)
    if len(iters) == 1: axes = [axes]
    
    print(f"{'Iter':<5} | {'Cluster':<7} | {'Phi':>7} | {'Psi':>7} | {'Count':>5} | {'Description'}")
    print("-" * 65)
    
    for i, ax in zip(iters, axes):
        subset = df[df['Iter'] == i]
        phi = subset['Phi'].values
        psi = subset['Psi'].values
        
        # Run Clustering
        # Relaxed threshold: eps=0.28 (approx 16 degrees radius)
        top_clusters, labels = get_cluster_centers(phi, psi, eps=0.28, min_samples=15)
        
        # PLOT
        # Plot noise first
        noise_mask = labels == -1
        ax.scatter(phi[noise_mask], psi[noise_mask], c='lightgrey', s=5, alpha=0.3, label='Noise')
        
        # Plot clusters
        # We need a color map for labels
        # Only color the top N clusters distinctively? Or all valid clusters?
        # User asked for "found cluster points", let's color all valid clusters using a cmap
        # but maybe highlight centroids of top ones.
        
        # Map labels to colors: discrete map
        # valid labels
        unique_valid = sorted(list(set(labels) - {-1}))
        
        # Use a colormap
        cmap = plt.cm.get_cmap('tab10', len(unique_valid) if unique_valid else 1)
        
        for lab in unique_valid:
            mask = labels == lab
            # Check if this label is in top_clusters (to know order)
            rank = -1
            for r, tc in enumerate(top_clusters):
                if tc['label'] == lab:
                    rank = r
                    break
            
            # If it's a major cluster, plot visibly
            if rank != -1:
                 marker='o'
                 alpha=0.6
                 label_text = f"C{rank+1}" 
            else:
                 # Minor cluster not in top N
                 marker='.'
                 alpha=0.2
                 label_text = None

            ax.scatter(phi[mask], psi[mask], s=10, alpha=alpha, label=label_text)
            
        ax.set_title(f"Iter {i}")
        ax.set_xlabel("Phi")
        if i == iters[0]: ax.set_ylabel("Psi")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        
        # Print and plot centroids
        for rank, c in enumerate(top_clusters):
            cx, cy = c['phi'], c['psi']
            
            # Annotate plot
            ax.plot(cx, cy, 'kX', markersize=10, markeredgecolor='white')
            ax.text(cx+5, cy+5, str(rank+1), fontsize=12, fontweight='bold', color='black')
            
            # Improved Annotations
            desc = "Unknown"
            
            # Alpha Right (-75, -45)
            if -100 <= cx <= -40 and -80 <= cy <= -10: 
                desc = "Alpha_R"
            
            # Beta / C5 / Extended (Top Left & Bottom Left)
            elif (cx <= -100 or cx >= 150) and (cy >= 90 or cy <= -90):
                desc = "Beta/C5"
                
            # C7eq / Polyproline II (-80, +70)
            elif -120 <= cx <= -40 and 0 <= cy <= 100:
                desc = "C7eq"
                
            # C7ax / Alpha Left (+60, -70 is Ax, +60, +40 is Alpha_L)
            elif 30 <= cx <= 90 and -90 <= cy <= -30:
                desc = "C7ax"
            elif 30 <= cx <= 90 and 0 <= cy <= 80:
                desc = "Alpha_L"
            
            # High Energy Barrier (0,0)
            elif -25 <= cx <= 25 and -25 <= cy <= 25:
                desc = "Barrier"

            print(f"{i:<5} | #{rank+1:<6} | {cx:>7.1f} | {cy:>7.1f} | {int(c['count']):>5} | {desc}")

        # Overlay Regions
        plot_regions(ax)
        
    plt.suptitle("Cluster Analysis (Top Clusters per Iteration)")
    
    # Save PNG
    out_file_png = Path("demo") / "cluster_analysis.png"
    plt.savefig(out_file_png, dpi=150)
    print(f"\nSaved plot to {out_file_png}")

    # Save PDF
    out_file_pdf = Path("demo") / "cluster_analysis.pdf"
    plt.savefig(out_file_pdf, dpi=300, format='pdf')
    print(f"Saved plot to {out_file_pdf}")

if __name__ == "__main__":
    main()
