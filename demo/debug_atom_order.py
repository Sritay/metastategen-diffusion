import numpy as np
import torch
from pathlib import Path

# Load 1 frame from MDShare (Diffusion Target)
mdshare_path = Path("data/raw/alanine-dipeptide-3x250ns-heavy-atom-positions.npz")
if mdshare_path.exists():
    with np.load(mdshare_path) as f:
        # keys like 'arr_0'
        arr = f['arr_0']
        mdshare_frame = arr[0]
        print(f"MDShare Frame Shape: {mdshare_frame.shape}") # Should be 30 flattened -> 10,3
else:
    print("MDShare not found")

# Load 1 frame from Timewarp (Surrogate Template)
timewarp_path = Path("data/timewarp/train/positions.pt")
if timewarp_path.exists():
    timewarp_frame = torch.load(timewarp_path)[0].numpy()
    print(f"Timewarp Frame Shape: {timewarp_frame.shape}") # Should be 22,3
else:
    print("Timewarp not found")

# Compare Distances
import scipy.spatial.distance

# MDShare: [30] -> [10, 3]
md_pos = mdshare_frame.reshape(10, 3)

# Timewarp: [22, 3]
tw_pos = timewarp_frame

print("\nMDShare Heavy Atoms (First 5):")
print(md_pos[:5])
print("\nTimewarp All Atoms (First 5):")
print(tw_pos[:5])

# We need to find the subset of 10 atoms in Timewarp that minimizes RMSD to MDShare
# Since they are different frames, we can't compare positions directly.
# BUT, we can compare internal distances (pairwise distance matrix).

md_dists = scipy.spatial.distance.pdist(md_pos)
print(f"\nMDShare Pairwise Dists shape: {md_dists.shape}")

# Brute force? 22 choose 10 is 646,646. A bit large for python script if slow.
# Heuristic: Timewarp should follow PDB order mostly.
# Common indices?
# Heavy indices used before: [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]

candidate_indices = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
tw_subset = tw_pos[candidate_indices]
tw_dists = scipy.spatial.distance.pdist(tw_subset)

# Compare distance histograms or correlation
corr = np.corrcoef(md_dists, tw_dists)[0, 1]
print(f"Correlation with candidate [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]: {corr:.4f}")

# Try to find best match greedily or check atom types if available?
# Timewarp atom types?
# Load Timewarp metadata if possible? 
# or just print masses/types if implied.

# Search for best match
import itertools

# We know MDShare has 10 atoms. Timewarp has 22.
# We need to find 10 indices in Timewarp [0..21] such that the pairwise distances
# of those 10 atoms match the pairwise distances of the 10 MDShare atoms.

# Since MDShare order is fixed (0..9), we need to find an ordered list of 10 indices in Timewarp
# [idx_0, idx_1, ..., idx_9] where idx_i corresponds to MDShare atom i.
# 22 P 10 is too huge.
# BUT, we can assume they are heavy atoms.
# Let's identify heavy atoms in Timewarp first? Not explicit.
# But heavy atoms shouldn't be close to each other like H. C-H is ~1.09.
# The MDShare atoms are heavy.
# Let's assume indices are roughly sorted or at least heavy atoms are a subset.

print("\nSearching for matching indices...")
best_corr = -1
best_indices = None

# Optimization:
# MDShare atom 0 is likely ACE CH3 or C.
# Atom 1 is ... 
# MDShare is 10 atoms.
# Timewarp is 22 atoms.
# Let's try to match purely by distance matrix signature.
# This is a Subgraph Isomorphism problem where edges are distances.
# Since graph is small and fully connected (distance matrix), can we use a heuristic?

# Match sequential distances first.
# md_pos[i] to md_pos[i+1] distances are:
md_seq_dists = [np.linalg.norm(md_pos[i] - md_pos[i+1]) for i in range(9)]
# We want to find a chain of 10 indices in Timewarp that matches these distances.

def find_chain(current_chain, remaining_indices):
    if len(current_chain) == 10:
        return current_chain
    
    last_idx = current_chain[-1]
    target_dist = md_seq_dists[len(current_chain) - 1]
    
    # Tolerances
    tol = 0.05 # 0.05 Angstrom tolerance? Wait, units? nm or A?
    # MDShare is usually nm. Timewarp might be nm or A.
    # Check bounds.
    # MDShare dists: 1->2 is 1.5ish?
    
    # Try all reasonable next steps
    for next_idx in remaining_indices:
        d = np.linalg.norm(tw_pos[last_idx] - tw_pos[next_idx])
        # If units don't match, this fails.
        # Let's check scaling factor first.
        yield from [] # Dummy

# Let's guess units.
md_d1 = np.linalg.norm(md_pos[0]-md_pos[1])
# tw_subset_d1 = np.linalg.norm(tw_pos[candidate_indices[0]]-tw_pos[candidate_indices[1]])

# If MDShare is nm, C-C is ~0.15. If Angstrom, ~1.5.
print(f"MDShare d(0,1): {md_d1:.4f}")

# Timewarp check
# Find ANY distance in Timewarp comparable to md_d1.
# If md_d1 is 0.15 and Timewarp has 1.5s, then factor is 10.
tw_pdist = scipy.spatial.distance.pdist(tw_pos)
print(f"Timewarp min/max/mean pairwise dist: {tw_pdist.min():.4f}, {tw_pdist.max():.4f}, {tw_pdist.mean():.4f}")
print(f"MDShare min/max/mean pairwise dist: {md_dists.min():.4f}, {md_dists.max():.4f}, {md_dists.mean():.4f}")

# If factor differs, normalize?
ratio = np.mean(tw_pdist) / np.mean(md_dists)
print(f"Ratio (TW/MD): {ratio:.4f}")

# Re-run correlation with assumed indices?
# Maybe the order in candidate_indices was just wrong?
# What if Timewarp heavy atoms are just 0,1,2...9 ? Or some other subset?
# Heavy atoms are usually: C, O, N, CA, CB...
# H are usually lighter/shorter bonds.
# Let's find 10 atoms in Timewarp that are "heavy like".
# i.e., bonded neighbors are > 1.2 distance (if A).
# Actually, just matching the full distance matrix of 10 atoms is robust.

# Let's try to find a permutation of candidate_indices that gives good correlation?
# Maybe just shuffled?
import itertools
p_iter = itertools.permutations(candidate_indices)
# Too many (3.6M).

# But maybe the set of atoms is correct, just order is wrong?
# Let's check correlation of Sorted Distance Matrix (histogram matching).
# If the SET of pairwise distances is the same, the atoms are the same (rigid body).
md_dists_sorted = np.sort(md_dists)
tw_dists_subset_sorted = np.sort(tw_dists)

print("\nMDShare Sorted Dists (first 5):", md_dists_sorted[:5])
print("Timewarp Candidate Sorted Dists (first 5):", tw_dists_subset_sorted[:5])

# If they don't match, then the set of atoms is wrong.



