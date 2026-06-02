#!/usr/bin/env python3
"""Evaluation comparison pipeline for visual odometry.

Compares estimated trajectories against ground truth and/or baselines.
Supports fast mock mode for CI testing without VO inference.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slam_dnn.eval import align_umeyama, compute_ape, compute_rte, evaluate
from slam_dnn.trajectory import extract_translations


def get_mock_poses(n_poses: int = 10) -> list[np.ndarray]:
    """Pre-computed hardcoded poses for testing.
    
    Generates a simple trajectory with linear X translation and Z drift.
    No VO inference required - completes instantly.
    
    Args:
        n_poses: Number of poses to generate.
        
    Returns:
        List of 4x4 SE(3) matrices.
    """
    poses = []
    for i in range(n_poses):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = 0.1 * i  # Simple X translation
        T[2, 3] = 0.05 * i  # Z drift
        poses.append(T)
    return poses


def get_mock_ground_truth(n_poses: int = 10) -> list[np.ndarray]:
    """Generate mock ground truth poses (perfect trajectory, no drift).
    
    Args:
        n_poses: Number of poses to generate.
        
    Returns:
        List of 4x4 SE(3) matrices.
    """
    poses = []
    for i in range(n_poses):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = 0.1 * i  # X translation only
        poses.append(T)
    return poses


def compute_metrics(
    est_poses: list[np.ndarray],
    gt_poses: list[np.ndarray],
    label: str = "estimated",
) -> dict:
    """Compute evaluation metrics for a trajectory.
    
    Uses slam_dnn.eval functions: align_umeyama, compute_ape, compute_rte.
    
    Args:
        est_poses: Estimated 4x4 SE(3) poses.
        gt_poses: Ground truth 4x4 SE(3) poses.
        label: Label for this trajectory (e.g., "ours", "baseline").
        
    Returns:
        Dictionary with metrics: ape_rmse, ape_mean, rte_rmse, rte_mean, scale, num_frames.
    """
    report = evaluate(est_poses, gt_poses, with_scale=True)
    
    return {
        "label": label,
        "ape_rmse": report["ape_rmse"],
        "ape_mean": report["ape_mean"],
        "rte_rmse": report["rte_rmse"],
        "rte_mean": report["rte_mean"],
        "scale": report["scale"],
        "num_frames": report["num_frames"],
    }


def generate_report(
    results: list[dict],
    output_path: str,
    baseline_available: bool = False,
) -> None:
    """Generate markdown comparison report.
    
    Args:
        results: List of metrics dictionaries from compute_metrics().
        output_path: Path to write the markdown report.
        baseline_available: Whether baseline results are included.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Trajectory Comparison Report",
        "",
        "## Results Summary",
        "",
    ]
    
    if not baseline_available:
        lines.extend([
            "**Baseline unavailable** - Baseline results not included in this run.",
            "",
        ])
    
    lines.extend([
        "| Metric | Value |",
        "|--------|-------|",
    ])
    
    for result in results:
        lines.append(f"### {result['label']}")
        lines.extend([
            f"- **APE RMSE**: {result['ape_rmse']:.6f}",
            f"- **APE Mean**: {result['ape_mean']:.6f}",
            f"- **RTE RMSE**: {result['rte_rmse']:.6f}",
            f"- **RTE Mean**: {result['rte_mean']:.6f}",
            f"- **Scale**: {result['scale']:.6f}",
            f"- **Frames**: {result['num_frames']}",
            "",
        ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Report written to {output_path}")


def plot_trajectory_comparison(
    ours: list[np.ndarray],
    baseline: list[np.ndarray] | None,
    gt: list[np.ndarray],
    output_path: str,
) -> None:
    """Plot trajectory comparison using matplotlib.
    
    Args:
        ours: Our estimated poses.
        baseline: Baseline poses (can be None).
        gt: Ground truth poses.
        output_path: Path to save the PNG plot.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract translations
    gt_centers = extract_translations(gt)
    ours_centers = extract_translations(ours)
    
    # Plot GT
    if len(gt_centers) > 0:
        ax.plot(gt_centers[:, 0], gt_centers[:, 2], 'k--', label='Ground Truth', linewidth=2)
        ax.plot(gt_centers[0, 0], gt_centers[0, 2], 'go', markersize=10, label='GT Start')
        ax.plot(gt_centers[-1, 0], gt_centers[-1, 2], 'gs', markersize=10, label='GT End')
    
    # Plot ours
    if len(ours_centers) > 0:
        ax.plot(ours_centers[:, 0], ours_centers[:, 2], 'b-', label='Ours', linewidth=2, alpha=0.7)
        ax.plot(ours_centers[0, 0], ours_centers[0, 2], 'bo', markersize=10, label='Ours Start')
        ax.plot(ours_centers[-1, 0], ours_centers[-1, 2], 'bs', markersize=10, label='Ours End')
    
    # Plot baseline if available
    if baseline is not None and len(baseline) > 0:
        baseline_centers = extract_translations(baseline)
        ax.plot(baseline_centers[:, 0], baseline_centers[:, 2], 'r-', 
                label='Baseline', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Trajectory Comparison (Top-Down View)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluation comparison pipeline for visual odometry"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "ci"],
        default="mock",
        help="Run mode: 'mock' for fast testing with hardcoded poses, 'ci' for synthetic data",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline comparison (report will note 'Baseline unavailable')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/reports",
        help="Output directory for reports and plots",
    )
    
    args = parser.parse_args()
    
    if args.mode == "mock":
        print("Running in mock mode (fast, no VO inference)...")
        
        # Generate mock poses
        est_poses = get_mock_poses(n_poses=10)
        gt_poses = get_mock_ground_truth(n_poses=10)
        
        # Compute metrics
        results = []
        ours_metrics = compute_metrics(est_poses, gt_poses, label="Ours (Mock)")
        results.append(ours_metrics)
        
        # Generate report
        report_path = os.path.join(args.output_dir, "comparison_report.md")
        baseline_available = not args.skip_baseline
        generate_report(results, report_path, baseline_available=baseline_available)
        
        # Plot trajectory
        plot_path = os.path.join(args.output_dir, "trajectory_comparison.png")
        baseline_poses = None if args.skip_baseline else get_mock_poses(n_poses=10)
        plot_trajectory_comparison(ours=est_poses, baseline=baseline_poses, gt=gt_poses, output_path=plot_path)
        
        print("Mock mode completed successfully.")
        
    elif args.mode == "ci":
        print("CI mode not implemented in this minimal version.")
        print("Use --mode mock for fast testing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
