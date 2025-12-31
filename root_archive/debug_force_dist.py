import torch
import matplotlib.pyplot as plt

def main():
    try:
        forces = torch.load("data/processed/ala2/al_forces_ref.pt")
        print(f"Force shape: {forces.shape}") # [N_frames, N_atoms, 3]
        
        # Calculate magnitudes
        mags = forces.norm(dim=-1)
        print(f"Force magnitudes stats:")
        print(f"  Max: {mags.max().item()}")
        print(f"  Mean: {mags.mean().item()}")
        print(f"  Std: {mags.std().item()}")
        
        # Flatten and check large components
        flat_f = forces.flatten()
        print(f"Component stats:")
        print(f"  Max: {flat_f.max().item()}")
        print(f"  Min: {flat_f.min().item()}")
        print(f"  Std: {flat_f.std().item()}")
        
        q99 = torch.quantile(mags, 0.99)
        q999 = torch.quantile(mags, 0.999)
        print(f"  99th percentile magnitude: {q99.item()}")
        print(f"  99.9th percentile magnitude: {q999.item()}")
        
        # Check energy stats if possible
        try:
            energies = torch.load("data/processed/ala2/al_energies_ref.pt")
            print(f"Energy shape: {energies.shape}")
            print(f"Energy stats:")
            print(f"  Max: {energies.max().item()}")
            print(f"  Min: {energies.min().item()}")
            print(f"  Mean: {energies.mean().item()}")
            print(f"  Std: {energies.std().item()}")
        except Exception as e:
            print(f"Could not load energies: {e}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
