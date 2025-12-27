import argparse
import csv
import json
from pathlib import Path

import torch

from metastategen.utils import get_logger
from metastategen.eval.rmsd import greedy_cluster

log = get_logger("cluster_samples")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=str, required=True, help="Path to samples.pt")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (defaults to samples parent)")
    ap.add_argument("--rmsd-thresh", type=float, default=1.50, help="RMSD threshold for greedy clustering")
    args = ap.parse_args()

    samples_path = Path(args.samples)
    samples = torch.load(samples_path)
    if samples.dim() != 3 or samples.shape[-1] != 3:
        raise ValueError(f"Expected samples [M,N,3], got {samples.shape}")

    out_dir = Path(args.outdir) if args.outdir else samples_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Clustering {samples.shape[0]} samples with rmsd_thresh={args.rmsd_thresh}")
    assignments, medoid_indices, cluster_sizes = greedy_cluster(samples, rmsd_thresh=args.rmsd_thresh)

    clusters_json = out_dir / "clusters.json"
    with clusters_json.open("w") as f:
        json.dump(
            {
                "rmsd_thresh": args.rmsd_thresh,
                "n_samples": int(samples.shape[0]),
                "n_clusters": int(len(medoid_indices)),
                "medoid_indices": medoid_indices,
                "assignments": assignments,
            },
            f,
            indent=2,
        )

    medoids = samples[medoid_indices]
    torch.save(medoids, out_dir / "cluster_medoids.pt")

    sizes_csv = out_dir / "cluster_sizes.csv"
    with sizes_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "size", "medoid_index"])
        for cid, size in enumerate(cluster_sizes):
            writer.writerow([cid, size, medoid_indices[cid]])

    log.info(f"Wrote: {clusters_json}")
    log.info(f"Wrote: {out_dir / 'cluster_medoids.pt'}")
    log.info(f"Wrote: {sizes_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
