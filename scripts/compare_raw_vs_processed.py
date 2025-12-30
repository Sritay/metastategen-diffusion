import torch
import numpy as np
import glob

def main():
    # 1. Load Raw
    raw_files = sorted(glob.glob("data/timewarp/train/*.npz"))
    print(f"Loading raw: {raw_files[0]}")
    d = np.load(raw_files[0])
    raw_pos = d['positions'][0]
    raw_frc = d['forces'][0]
    raw_ene = d['energies'][0, 0] # Pot Energy
    
    print(f"Raw Frame 0:")
    print(f"  Pos[0]: {raw_pos[0]}")
    print(f"  Frc[0]: {raw_frc[0]}")
    print(f"  Ene:    {raw_ene}")
    
    # 2. Load Processed Shard (Pos)
    shard = torch.load("data/processed/ala2_all_atom/shards/shard_00000.pt")
    proc_pos = shard['positions'][0].numpy()
    
    # 3. Load Processed Force/Energy (Global)
    proc_frc_all = torch.load("data/processed/ala2_all_atom/al_forces_ref.pt")
    proc_frc = proc_frc_all[0].numpy()
    
    proc_ene_all = torch.load("data/processed/ala2_all_atom/al_energies_ref.pt")
    proc_ene = proc_ene_all[0].numpy()
    
    print(f"\nProcessed Frame 0:")
    print(f"  Pos[0]: {proc_pos[0]}")
    print(f"  Frc[0]: {proc_frc[0]}")
    print(f"  Ene:    {proc_ene}")
    
    # Diff
    diff_pos = np.abs(raw_pos - proc_pos).max()
    diff_frc = np.abs(raw_frc - proc_frc).max()
    diff_ene = np.abs(raw_ene - proc_ene).max()
    
    print(f"\nMax Diff:")
    print(f"  Pos: {diff_pos}")
    print(f"  Frc: {diff_frc}")
    print(f"  Ene: {diff_ene}")
    
    if diff_pos > 1e-5 or diff_frc > 1e-3:
        print("MISMATCH DETECTED!")
    else:
        print("Data is IDENTICAL.")

if __name__ == "__main__":
    main()
