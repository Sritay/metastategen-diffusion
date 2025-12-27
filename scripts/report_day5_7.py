import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from metastategen.utils import get_logger

log = get_logger("report_day5_7")


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _to_float(arr):
    out = []
    for v in arr:
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return np.array(out, dtype=np.float64)


def _load_assignments(path: Path) -> list[int]:
    with path.open("r") as f:
        data = json.load(f)
    return data["assignments"]


def _coverage_curve(assignments: list[int]) -> tuple[np.ndarray, np.ndarray]:
    seen = set()
    xs = []
    ys = []
    for i, cid in enumerate(assignments):
        seen.add(cid)
        xs.append(i + 1)
        ys.append(len(seen))
    return np.array(xs), np.array(ys)


def _plot_uncertainty_vs_error(rows, out_path: Path):
    unc_mean = _to_float([r["uncertainty_mean"] for r in rows])
    nn_dist = _to_float([r["nn_phi_psi_dist_mean"] for r in rows])
    kl = _to_float([r["kl_to_ref"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(unc_mean, nn_dist, marker="o", color="tab:blue", label="NN phi/psi distance")
    ax1.set_xlabel("Uncertainty (bin mean)")
    ax1.set_ylabel("NN phi/psi distance (deg)")

    ax2 = ax1.twinx()
    ax2.plot(unc_mean, kl, marker="s", color="tab:orange", label="KL to reference")
    ax2.set_ylabel("KL divergence")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_triage_tradeoff(rows, out_path: Path):
    rejection = _to_float([r["rejection_rate"] for r in rows])
    kl = _to_float([r["kl_to_ref"] for r in rows])
    basins = _to_float([r["basin_count"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(rejection, kl, marker="o", color="tab:blue", label="KL to reference")
    ax1.set_xlabel("Rejection rate")
    ax1.set_ylabel("KL divergence")

    ax2 = ax1.twinx()
    ax2.plot(rejection, basins, marker="s", color="tab:green", label="Basin count")
    ax2.set_ylabel("Basin count")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_basin_coverage(baseline_assign, pooled_assign, consensus_assign, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    x0, y0 = _coverage_curve(baseline_assign)
    ax.plot(x0, y0, label="single-model baseline")

    x1, y1 = _coverage_curve(pooled_assign)
    ax.plot(x1, y1, label="ensemble pooled")

    if consensus_assign is not None:
        x2, y2 = _coverage_curve(consensus_assign)
        ax.plot(x2, y2, label="ensemble consensus")

    ax.set_xlabel("# samples")
    ax.set_ylabel("# unique basins")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncertainty-bins", type=str, required=True)
    ap.add_argument("--triage-curve", type=str, required=True)
    ap.add_argument("--baseline-clusters", type=str, required=True)
    ap.add_argument("--ensemble-pooled-clusters", type=str, required=True)
    ap.add_argument("--ensemble-consensus-clusters", type=str, default=None)
    ap.add_argument("--exp-id", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    if args.outdir:
        out_dir = Path(args.outdir)
    elif args.exp_id:
        out_dir = Path("runs") / args.exp_id / "reports"
    else:
        out_dir = Path("reports") / "day5_7"
    out_dir.mkdir(parents=True, exist_ok=True)

    unc_rows = _load_csv(Path(args.uncertainty_bins))
    triage_rows = _load_csv(Path(args.triage_curve))
    baseline_assign = _load_assignments(Path(args.baseline_clusters))
    pooled_assign = _load_assignments(Path(args.ensemble_pooled_clusters))
    consensus_assign = None
    if args.ensemble_consensus_clusters:
        consensus_assign = _load_assignments(Path(args.ensemble_consensus_clusters))

    _plot_uncertainty_vs_error(unc_rows, out_dir / "uncertainty_vs_error.png")
    _plot_triage_tradeoff(triage_rows, out_dir / "triage_tradeoff.png")
    _plot_basin_coverage(baseline_assign, pooled_assign, consensus_assign, out_dir / "basin_coverage_budget.png")

    log.info(f"Wrote reports to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
