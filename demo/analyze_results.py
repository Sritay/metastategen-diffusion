import pandas as pd
import numpy as np

def analyze():
    print("--- AL Analysis ---")
    
    # 1. Density
    try:
        df_dens = pd.read_csv("demo/evolution_density.csv")
        print("\n[Density Evolution]")
        print(df_dens.groupby("Iter").size().to_frame("Count"))
        # Simple coverage check: discretize phi/psi
        df_dens['phi_bin'] = (df_dens['Phi'] // 20).astype(int)
        df_dens['psi_bin'] = (df_dens['Psi'] // 20).astype(int)
        coverage = df_dens.groupby("Iter")[['phi_bin', 'psi_bin']].apply(lambda x: len(x.drop_duplicates()))
        print("Grid Coverage (20deg bins):")
        print(coverage)
    except Exception as e:
        print(f"Density analysis failed: {e}")

    # 2. Acquisition
    try:
        df_acq = pd.read_csv("demo/acquired_points.csv")
        print("\n[Acquisition Stats]")
        print(df_acq.groupby(["Iter", "Type"]).size())
    except Exception as e:
        print(f"Acquisition analysis failed: {e}")

    # 3. Uncertainty
    try:
        df_unc = pd.read_csv("demo/uncertainty_map.csv")
        print("\n[Uncertainty Stats]")
        print(df_unc.groupby("Iter")["Uncertainty"].describe()[['mean', 'std', 'max']])
    except Exception as e:
        print(f"Uncertainty analysis failed: {e}")
        
    # 4. Metrics
    try:
        df_met = pd.read_csv("runs/day10_al_5_hpc/al_metrics.csv")
        print("\n[Metrics Summary]")
        # basin_fraction is constantly 1.0, so let's check KL
        if 'kl_to_val' in df_met.columns:
            print(df_met[['iter', 'basin_fraction', 'kl_to_val']])
        else:
            print(df_met[['iter', 'basin_fraction']])
    except Exception as e:
        print(f"Metrics analysis failed: {e}")

if __name__ == "__main__":
    analyze()
