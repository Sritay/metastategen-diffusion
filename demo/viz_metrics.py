import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="runs/day10_al_5_hpc")
    args = parser.parse_args()

    path = Path(args.run) / "al_metrics.csv"
    if not path.exists():
        print("Metrics file not found.")
        return
        
    df = pd.read_csv(path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Basin Fraction
    # Note: 'basin_fraction' might be 0, 0, 0, 1.0 based on previous head output
    # Let's plot it anyway.
    axes[0].plot(df['iter'], df['basin_fraction'], marker='o', linestyle='-', color='green')
    axes[0].set_title("Basin Discovery fraction")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Fraction of Validation Basins Found")
    axes[0].set_ylim(-0.1, 1.1)
    
    # Plot 2: KL Divergence (if available)
    if 'kl_to_val' in df.columns:
        axes[1].plot(df['iter'], df['kl_to_val'], marker='s', linestyle='--', color='purple')
        axes[1].set_title("KL Divergence to Val Density (Phi/Psi)")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("KL Divergence")
    
    plt.suptitle("Active Learning Performance Metrics")
    
    # Save PNG
    out_file_png = Path("demo") / "learning_curves.png"
    plt.savefig(out_file_png, dpi=150)
    print(f"Saved {out_file_png}")

    # Save PDF
    out_file_pdf = Path("demo") / "learning_curves.pdf"
    plt.savefig(out_file_pdf, dpi=300, format='pdf')
    print(f"Saved {out_file_pdf}")

if __name__ == "__main__":
    main()
