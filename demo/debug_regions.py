from demo.regions import get_ground_truth_data
import numpy as np

data = get_ground_truth_data()

if data is None:
    print("ERROR: get_ground_truth_data() returned None")
else:
    print(f"Data shape: {data.shape}")
    print(f"Phi range: {data[:, 0].min()} to {data[:, 0].max()}")
    print(f"Psi range: {data[:, 1].min()} to {data[:, 1].max()}")
    
    # Check histogram density
    H, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=100, range=[[-180, 180], [-180, 180]])
    print(f"Max density: {H.max()}")
    print(f"Sum density: {H.sum()}")
