#!/usr/bin/env python3
"""Evaluation comparison pipeline for visual odometry.

Compares estimated trajectories against ground truth and/or baselines.
Supports:
- mock: Fast mock mode for testing without PyTorch dependencies
- ci: Fast synthetic end-to-end VO comparison pipeline on generated trajectories
- full: Real-dataset visual odometry execution on KITTI 05 image sequences
"""

import argparse
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slam_dnn.eval import align_umeyama, align_trajectories, compute_ape, compute_rte, evaluate
from slam_dnn.trajectory import extract_translations
from slam_dnn.camera import PinholeCamera
from slam_dnn.vo import VisualOdometry
from slam_dnn.kitti_loader import KITTIFrameLoader
from slam_dnn.testdata.synthetic import SyntheticVODataset
from baselines.minislam_wrapper import run_minislam_on_kitti, check_minislam_available
from baselines.pyslam_wrapper import run_pyslam_on_kitti, check_pyslam_available


# Try to import evo components
try:
    from evo.core.trajectory import PoseTrajectory3D, se3_poses_to_xyz_quat_wxyz
    from evo.core.metrics import APE, RPE, PoseRelation
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False


def get_mock_poses(n_poses: int = 10) -> list[np.ndarray]:
    """Pre-computed hardcoded poses for testing.

    Generates a simple trajectory with linear X translation and Z drift.
    No VO inference required.
    """
    poses = []
    for i in range(n_poses):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = 0.1 * i  # Simple X translation
        T[2, 3] = 0.05 * i  # Z drift
        poses.append(T)
    return poses


def get_mock_ground_truth(n_poses: int = 10) -> list[np.ndarray]:
    """Generate mock ground truth poses (perfect trajectory, no drift)."""
    poses = []
    for i in range(n_poses):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = 0.1 * i  # X translation only
        poses.append(T)
    return poses


def run_slam_dnn_on_loader(
    loader,
    extractor: str = 'superpoint',
    matcher: str = 'lightglue',
    max_keypoints: int = 2048,
    scale: float = 1.0,
    device: str = 'cpu',
    min_matches: int = 20,
    target_resolution: int | None = None,
    use_joint_ba: bool = False,
    ba_window_size: int = 5,
    use_depth_prior: bool = False,
    depth_source: str = 'directory',
    depth_directory: str = 'data/kitti/05/depth',
    depth_scale_factor: float = 256.0,
    depth_model_name: str = 'LiheYoung/depth-anything-small-hf',
    depth_scale_mode: str = 'median_ratio',
    min_parallax: float = 8.0,
    max_overlap: float = 0.85,
    max_keyframe_interval: int = 10,
) -> list[np.ndarray]:
    """Run slam_dnn visual odometry on the loaded sequence.

    Args:
        loader: An iterable KITTIFrameLoader.
        matcher: 'lightglue' or 'classic'.
        max_keypoints: Max keypoints per frame.
        scale: Scaling factor for translation magnitude.
        device: 'cpu', 'cuda', or 'mps'.
        min_matches: Minimum matches required for pose estimation.
        use_joint_ba: Enable 3D-2D joint Bundle Adjustment tracking mode.
        ba_window_size: Number of keyframes in sliding window.
        use_depth_prior: Enable 3D-2D depth-prior tracking mode.
        depth_source: 'directory' or 'model'.
        depth_directory: Path to pre-computed depth maps.
        depth_scale_factor: Convert depth values to meters.
        depth_model_name: Hugging Face repository ID.
        depth_scale_mode: Method of scaling relative depth.

    Returns:
        List of 4x4 estimated poses.
    """
    K = loader.get_intrinsics()
    
    # Read first frame to initialize camera size
    frame_iter = iter(loader)
    first_frame = next(frame_iter)
    h, w = first_frame['image'].shape[:2]
    
    # Initialize PinholeCamera with matching intrinsics
    camera = PinholeCamera(width=w, height=h, fov_deg=63.0)
    camera.K = K.copy()
    camera.K_inv = np.linalg.inv(K)
    
    from slam_dnn.config import VOConfig
    config = VOConfig(
        extractor=extractor,
        matcher=matcher,
        max_keypoints=max_keypoints,
        scale=scale,
        device=device,
        min_matches=min_matches,
        target_resolution=target_resolution,
        use_joint_ba=use_joint_ba,
        ba_window_size=ba_window_size,
        use_depth_prior=use_depth_prior,
        depth_source=depth_source,
        depth_directory=depth_directory,
        depth_scale_factor=depth_scale_factor,
        depth_model_name=depth_model_name,
        depth_scale_mode=depth_scale_mode,
        min_parallax=min_parallax,
        max_overlap=max_overlap,
        max_keyframe_interval=max_keyframe_interval,
    )
    
    vo = VisualOdometry(
        camera=camera,
        config=config,
    )
    
    print(f"Running slam_dnn VisualOdometry ({matcher} matcher)...")
    t_start = time.time()
    
    # Iterate from frame 0
    for frame in loader:
        vo.process_frame(frame['image'])
        
    elapsed = time.time() - t_start
    fps = len(loader) / elapsed if elapsed > 0 else 0.0
    print(f"slam_dnn processed {len(loader)} frames in {elapsed:.2f}s ({fps:.2f} FPS)")
    
    # Timing and Statistics breakdown
    stats = vo.get_stats()
    timings = vo.get_timings()
    total_tracked_time = timings.get("total", 0.0)
    print(f"--- VO System Stats & Timings Breakdown ---")
    print(f"Total Keyframes: {stats.get('keyframes', 0)}")
    print(f"Successful PnP frames: {stats.get('successful', 0)}")
    print(f"Pose-failed PnP frames: {stats.get('pose_failed', 0)}")
    print(f"Tracking lost count: {stats.get('tracking_lost', 0)}")
    print(f"Motion model fallbacks: {stats.get('motion_model_fallbacks', 0)}")
    
    if total_tracked_time > 0:
        ext_time = timings.get("extraction", 0.0)
        match_time = timings.get("matching", 0.0)
        pnp_time = timings.get("pose_estimation", 0.0)
        other_time = max(0.0, total_tracked_time - ext_time - match_time - pnp_time)
        
        print(f"Accumulated Processing Time: {total_tracked_time:.3f}s")
        print(f"  - Keypoint Extraction: {ext_time:.3f}s ({ext_time/total_tracked_time*100.1:.1f}%)")
        print(f"  - Keypoint Matching:   {match_time:.3f}s ({match_time/total_tracked_time*100.1:.1f}%)")
        print(f"  - Pose Estimation PnP: {pnp_time:.3f}s ({pnp_time/total_tracked_time*100.1:.1f}%)")
        print(f"  - Other (e.g. depth):  {other_time:.3f}s ({other_time/total_tracked_time*100.1:.1f}%)")
    print(f"-------------------------------------------")
    
    return vo.get_trajectory().get_poses()


def compute_metrics_all(
    est_poses: list[np.ndarray],
    gt_poses: list[np.ndarray],
    label: str = "estimated",
) -> dict:
    """Compute metrics using both slam_dnn/eval.py and evo if available.

    Args:
        est_poses: List of 4x4 estimated poses.
        gt_poses: List of 4x4 ground truth poses.
        label: Label identifier (e.g., 'ours', 'baseline').

    Returns:
        A unified report dictionary.
    """
    # 1. Compute using our eval.py
    report = evaluate(est_poses, gt_poses, with_scale=True)
    
    n = report["num_frames"]
    aligned_est = report["aligned_poses"]
    aligned_centers = np.array([T[:3, 3] for T in aligned_est])
    gt_centers = np.array([T[:3, 3] for T in gt_poses[:n]])
    
    ape_values = compute_ape(aligned_centers, gt_centers).tolist()
    rte_values = compute_rte(aligned_centers, gt_centers).tolist()
    
    result = {
        "label": label,
        "ape_rmse": report["ape_rmse"],
        "ape_mean": report["ape_mean"],
        "rte_rmse": report["rte_rmse"],
        "rte_mean": report["rte_mean"],
        "scale": report["scale"],
        "num_frames": report["num_frames"],
        "aligned_poses": report["aligned_poses"],
        "ape_values": ape_values,
        "rte_values": rte_values,
        "evo_cross_val": False,
        "evo_ape_rmse": None,
        "evo_ape_mean": None,
    }
    
    # 2. Compute using evo library (if installed)
    if EVO_AVAILABLE:
        n = report["num_frames"]
        if n >= 3:
            try:
                # Extract camera centers in world coordinates for aligned estimated and ground truth
                aligned_est = report["aligned_poses"]
                aligned_centers = np.array([T[:3, 3] for T in aligned_est])
                gt_centers = np.array([T[:3, 3] for T in gt_poses[:n]])
                
                # Build evo Trajectory3D objects (already aligned, so orientations can be identity)
                timestamps = np.arange(n, dtype=np.float64)
                identity_quats = np.array([[0.0, 0.0, 0.0, 1.0] for _ in range(n)])
                
                traj_ref = PoseTrajectory3D(gt_centers, identity_quats, timestamps)
                traj_est = PoseTrajectory3D(aligned_centers, identity_quats, timestamps)
                
                ape_metric = APE(PoseRelation.translation_part)
                ape_metric.process_data((traj_ref, traj_est))
                evo_stats = ape_metric.get_all_statistics()
                
                result["evo_cross_val"] = True
                result["evo_ape_rmse"] = float(evo_stats["rmse"])
                result["evo_ape_mean"] = float(evo_stats["mean"])
                
                # Sanity check
                diff = abs(report["ape_rmse"] - evo_stats["rmse"])
                if diff > 1e-3:
                    print(f"Warning: APE RMSE discrepancy between slam_dnn ({report['ape_rmse']:.6f}) and evo ({evo_stats['rmse']:.6f})")
            except Exception as e:
                print(f"Error during evo metrics calculation: {e}")
                
    return result


def generate_report(
    results: list[dict],
    output_path: str,
    active_baselines: list[str] = None,
    mode_name: str = "mock",
) -> None:
    """Generate detailed markdown trajectory comparison report."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Trajectory Comparison Report",
        "",
        f"**Execution Mode**: `{mode_name.upper()}`",
        f"**Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Evaluation Metrics",
        "",
    ]
    
    # 1. Main APE/RTE Comparison Table
    lines.extend([
        "| Tracker | Frames | APE RMSE (m) | APE Mean (m) | RTE RMSE (m) | RTE Mean (m) | Umeyama Scale |",
        "|---------|--------|--------------|--------------|--------------|--------------|---------------|",
    ])
    
    for r in results:
        lines.append(
            f"| **{r['label']}** | {r['num_frames']} | {r['ape_rmse']:.6f} | {r['ape_mean']:.6f} | "
            f"{r['rte_rmse']:.6f} | {r['rte_mean']:.6f} | {r['scale']:.6f} |"
        )
    lines.append("")
    
    # 1.5 Visualizations section
    lines.extend([
        "## Visualizations",
        "",
        "### Trajectory Overlays",
        "The following plots show top-down and side-view comparisons of the trajectories:",
        "",
        "| Top-Down Trajectory Comparison | Side-View Trajectory Comparison |",
        "|--------------------------------|---------------------------------|",
        "| ![Top-Down](trajectory_comparison.png) | ![Side-View](trajectory_comparison_side.png) |",
        "",
        "### Trajectory Metric Drift (Error plots)",
        "The absolute pose error (APE) and relative trajectory error (RTE) over the frame index:",
        "",
        "![Error Plot](error_comparison.png)",
        "",
    ])
    
    # 2. Evo Cross-Validation section
    lines.extend([
        "## Evo Library Cross-Validation",
        "",
    ])
    
    if not EVO_AVAILABLE:
        lines.extend([
            "> [!NOTE]",
            "> **Evo library not installed/importable.** Cross-validation was skipped.",
            "> Install it using `pip install evo` to enable full mathematical validation.",
            "",
        ])
    else:
        lines.extend([
            "To confirm our mathematical correctness, we cross-validated our pure-NumPy evaluation metrics against the academic benchmark **`evo`** package:",
            "",
            "| Tracker | slam_dnn APE RMSE | evo APE RMSE | Discrepancy | Status |",
            "|---------|-------------------|--------------|-------------|--------|",
        ])
        for r in results:
            if r["evo_cross_val"]:
                diff = abs(r["ape_rmse"] - r["evo_ape_rmse"])
                status = "✅ PASS" if diff < 1e-4 else "⚠️ WARNING"
                lines.append(
                    f"| {r['label']} | {r['ape_rmse']:.6f} | {r['evo_ape_rmse']:.6f} | {diff:.2e} | {status} |"
                )
            else:
                lines.append(f"| {r['label']} | {r['ape_rmse']:.6f} | N/A | N/A | Skipped |")
        lines.append("")
        
    # 3. Methodological Summary section
    if not active_baselines:
        lines.extend([
            "## Methodological Summary",
            "",
            "> [!WARNING]",
            "> **No baselines were evaluated in this run.**",
            "",
        ])
    else:
        lines.extend([
            "## Methodological Summary",
            "",
            "- **Ours (slam_dnn)**: Uses learned deep features (**SuperPoint**) and **LightGlue** neural network matcher (or Classic matcher) along with an Essential Matrix 5-point RANSAC pipeline.",
        ])
        
        # Add baseline descriptions
        for r in results:
            label = r["label"]
            if "minislam" in label.lower():
                lines.append("- **Baseline (minislam)**: Uses traditional **ORB** keypoint extraction and OpenCV **BFMatcher** with ratio test for Essential Matrix recovery.")
            elif "pySLAM-ORB2" in label:
                lines.append("- **Baseline (pySLAM-ORB2)**: pySLAM monocular pipeline using classical binary **ORB2/FAST** tracking.")
            elif "pySLAM-SIFT" in label:
                lines.append("- **Baseline (pySLAM-SIFT)**: pySLAM monocular pipeline using robust gradient-based **SIFT/ROOT_SIFT** tracking.")
            elif "pySLAM-SP" in label:
                lines.append("- **Baseline (pySLAM-SP)**: pySLAM monocular pipeline using **SuperPoint** deep feature tracking for direct control comparison.")
            elif "pySLAM-XFeat" in label:
                lines.append("- **Baseline (pySLAM-XFeat)**: pySLAM monocular pipeline using lightweight real-time **XFeat** deep feature tracking.")
        lines.append("")
        
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Detailed report successfully written to {output_path}")


def plot_trajectory_comparison(
    ours: list[np.ndarray],
    baselines_dict: dict[str, list[np.ndarray]],
    gt: list[np.ndarray],
    output_path: str,
) -> None:
    """Plot XY (Top-down) and XZ (Side-view) trajectory comparisons using Matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
        
    # 1. XY Top-down plot
    fig_xy, ax_xy = plt.subplots(figsize=(10, 8))
    
    gt_centers = np.array([T[:3, 3] for T in gt])
    ours_centers = np.array([T[:3, 3] for T in ours])
    
    if len(gt_centers) > 0:
        ax_xy.plot(gt_centers[:, 0], gt_centers[:, 2], 'g--', label='Ground Truth', linewidth=2)
        ax_xy.plot(gt_centers[0, 0], gt_centers[0, 2], 'go', markersize=10, label='Start')
        ax_xy.plot(gt_centers[-1, 0], gt_centers[-1, 2], 'gs', markersize=10, label='End')
        
    if len(ours_centers) > 0:
        ax_xy.plot(ours_centers[:, 0], ours_centers[:, 2], 'b-', label='Ours (slam_dnn)', linewidth=2, alpha=0.8)
        
    for name, poses in baselines_dict.items():
        if poses is not None and len(poses) > 0:
            b_centers = np.array([T[:3, 3] for T in poses])
            color = BASELINES[name]["color"]
            label = BASELINES[name]["label"]
            ax_xy.plot(b_centers[:, 0], b_centers[:, 2], color=color, linestyle='-', label=label, linewidth=2, alpha=0.8)
        
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Z (m)')
    ax_xy.set_title('Trajectory Comparison (Top-Down View)')
    ax_xy.legend(loc='best')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect('equal')
    
    fig_xy.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig_xy)
    print(f"Top-down plot saved to {output_path}")
    
    # 2. XZ Side-view plot
    fig_xz, ax_xz = plt.subplots(figsize=(10, 6))
    if len(gt_centers) > 0:
        ax_xz.plot(gt_centers[:, 0], gt_centers[:, 1], 'g--', label='Ground Truth', linewidth=2)
        ax_xz.plot(gt_centers[0, 0], gt_centers[0, 1], 'go', markersize=10)
    if len(ours_centers) > 0:
        ax_xz.plot(ours_centers[:, 0], ours_centers[:, 1], 'b-', label='Ours (slam_dnn)', linewidth=2, alpha=0.8)
        
    for name, poses in baselines_dict.items():
        if poses is not None and len(poses) > 0:
            b_centers = np.array([T[:3, 3] for T in poses])
            color = BASELINES[name]["color"]
            label = BASELINES[name]["label"]
            ax_xz.plot(b_centers[:, 0], b_centers[:, 1], color=color, linestyle='-', label=label, linewidth=2, alpha=0.8)
        
    ax_xz.set_xlabel('X (m)')
    ax_xz.set_ylabel('Y (m)')
    ax_xz.set_title('Trajectory Comparison (Side View)')
    ax_xz.legend(loc='best')
    ax_xz.grid(True, alpha=0.3)
    
    xz_path = str(Path(output_path).with_name("trajectory_comparison_side.png"))
    fig_xz.savefig(xz_path, dpi=150, bbox_inches='tight')
    plt.close(fig_xz)
    print(f"Side-view plot saved to {xz_path}")


def plot_errors_comparison(
    results: list[dict],
    output_path: str,
) -> None:
    """Plot APE (Absolute Pose Error) and RTE (Relative Trajectory Error) over frame index."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping error plots")
        return
        
    fig, (ax_ape, ax_rte) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Map each label to its corresponding color
    color_map = {"Ours (Mock)": "b", "Ours (slam_dnn)": "b"}
    for name, info in BASELINES.items():
        color_map[info["label"]] = info["color"]
    color_map["Baseline (Mock)"] = "r"
    
    default_colors = ['c', 'm', 'y', 'k', 'g']
    
    for i, r in enumerate(results):
        label = r["label"]
        ape_vals = r["ape_values"]
        rte_vals = r["rte_values"]
        n_frames = len(ape_vals)
        frames = np.arange(n_frames)
        
        color = color_map.get(label, default_colors[i % len(default_colors)])
        
        ax_ape.plot(frames, ape_vals, color=color, linestyle='-', linewidth=2, label=f"{label} (Mean: {r['ape_mean']:.4f}m)")
        
        # Filter valid RTE entries for cleaner plotting
        rte_array = np.array(rte_vals)
        valid_indices = np.where(rte_array > 0.0)[0]
        if len(valid_indices) > 0:
            ax_rte.plot(valid_indices, rte_array[valid_indices], color=color, linestyle='-', linewidth=2, label=f"{label} (Mean: {r['rte_mean']:.4f}m)")
            
    ax_ape.set_ylabel('APE (m)')
    ax_ape.set_title('Absolute Pose Error (APE) over Frame Index')
    ax_ape.legend(loc='best')
    ax_ape.grid(True, alpha=0.3)
    
    ax_rte.set_xlabel('Frame Index')
    ax_rte.set_ylabel('RTE (m)')
    ax_rte.set_title('Relative Trajectory Error (RTE, 5m window) over Frame Index')
    ax_rte.legend(loc='best')
    ax_rte.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Error plots saved to {output_path}")


# Baseline system registry
BASELINES = {
    "minislam": {
        "run": run_minislam_on_kitti,
        "check": check_minislam_available,
        "label": "Baseline (minislam)",
        "color": "r",
    },
    "pyslam-orb2": {
        "run": lambda d, o, cf, m: run_pyslam_on_kitti(d, o, "orb2", cf, m),
        "check": check_pyslam_available,
        "label": "Baseline (pySLAM-ORB2)",
        "color": "orange",
    },
    "pyslam-sift": {
        "run": lambda d, o, cf, m: run_pyslam_on_kitti(d, o, "sift", cf, m),
        "check": check_pyslam_available,
        "label": "Baseline (pySLAM-SIFT)",
        "color": "purple",
    },
    "pyslam-superpoint": {
        "run": lambda d, o, cf, m: run_pyslam_on_kitti(d, o, "superpoint", cf, m),
        "check": check_pyslam_available,
        "label": "Baseline (pySLAM-SP)",
        "color": "brown",
    },
    "pyslam-xfeat": {
        "run": lambda d, o, cf, m: run_pyslam_on_kitti(d, o, "xfeat", cf, m),
        "check": check_pyslam_available,
        "label": "Baseline (pySLAM-XFeat)",
        "color": "deeppink",
    },
}


def get_active_baselines(args) -> list[str]:
    """Resolve active baselines list from parser arguments."""
    if args.skip_baseline:
        return []
        
    requested = args.baselines
    if not requested:
        return []
        
    if "none" in requested:
        return []
        
    if "all" in requested:
        # Return all baselines that can be run
        return list(BASELINES.keys())
        
    active = []
    for r in requested:
        if r in BASELINES:
            active.append(r)
        else:
            print(f"Warning: Unknown baseline requested: {r}")
    return active


def main():
    """CLI entry point."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(
        description="Evaluation comparison pipeline for visual odometry"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "ci", "full"],
        default="mock",
        help="Run mode: 'mock' for fast testing, 'ci' for synthetic VO pipeline, 'full' for real KITTI",
    )
    parser.add_argument(
        "--dataset",
        choices=["kitti05", "parking"],
        default="kitti05",
        help="Dataset sequence to run on in full mode (default: kitti05)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Max frames to process in ci or full mode (default: 100)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline execution entirely (backwards compatibility)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["all"],
        help="Baselines to run: minislam, pyslam-orb2, pyslam-sift, pyslam-superpoint, pyslam-xfeat, all, or none",
    )
    parser.add_argument(
        "--extractor",
        choices=["superpoint", "xfeat"],
        default="superpoint",
        help="Feature extractor to use for slam_dnn (default: superpoint)",
    )
    parser.add_argument(
        "--matcher",
        choices=["lightglue", "classic", "xfeat"],
        default="classic",
        help="Matcher to use for slam_dnn (default: classic for CPU speed)",
    )
    parser.add_argument(
        "--target-resolution",
        type=int,
        default=None,
        help="Resize image so max(H, W) is at most this value (default: None for original resolution)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device for slam_dnn (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/reports",
        help="Output directory for reports and plots",
    )
    parser.add_argument(
        "--use-joint-ba",
        action="store_true",
        help="Enable 3D-2D Joint BA tracking mode for slam_dnn",
    )
    parser.add_argument(
        "--ba-window-size",
        type=int,
        default=5,
        help="Maximum keyframes in active Joint BA sliding window (default: 5)",
    )
    parser.add_argument(
        "--use-depth-prior",
        action="store_true",
        help="Enable 3D-2D depth-prior tracking mode for slam_dnn",
    )
    parser.add_argument(
        "--depth-source",
        type=str,
        default="directory",
        help="Source of depth maps: 'directory' or 'model'",
    )
    parser.add_argument(
        "--depth-directory",
        type=str,
        default="data/kitti/05/depth",
        help="Directory containing pre-computed depth maps",
    )
    parser.add_argument(
        "--depth-scale-factor",
        type=float,
        default=256.0,
        help="Scale factor to convert depth map values to meters (default: 256.0)",
    )
    parser.add_argument(
        "--depth-model-name",
        type=str,
        default="LiheYoung/depth-anything-small-hf",
        help="Hugging Face repo name for depth estimation (default: LiheYoung/depth-anything-small-hf)",
    )
    parser.add_argument(
        "--depth-scale-mode",
        choices=["median_ratio", "fixed"],
        default="median_ratio",
        help="Scale alignment mode: 'median_ratio' (dynamic scale) or 'fixed' (multiply by depth-scale-factor)",
    )
    parser.add_argument(
        "--min-parallax",
        type=float,
        default=8.0,
        help="Minimum median parallax in pixels to trigger a keyframe (default: 8.0)",
    )
    parser.add_argument(
        "--max-overlap",
        type=float,
        default=0.85,
        help="Maximum overlap ratio before keyframe is forced (default: 0.85)",
    )
    parser.add_argument(
        "--max-keyframe-interval",
        type=int,
        default=10,
        help="Maximum consecutive frames before forcing a keyframe (default: 10)",
    )
    
    args = parser.parse_args()
    
    results = []
    
    if args.mode == "mock":
        print("Running in mock mode (fast, no VO inference)...")
        est_poses = get_mock_poses(n_poses=args.max_frames)
        gt_poses = get_mock_ground_truth(n_poses=args.max_frames)
        
        # Calculate ours
        ours_metrics = compute_metrics_all(est_poses, gt_poses, label="Ours (Mock)")
        results.append(ours_metrics)
        
        # Calculate baselines if requested
        active_baselines = get_active_baselines(args)
        baseline_poses_dict = {}
        
        for name in active_baselines:
            # Generate different mock drift per baseline
            base_poses = get_mock_poses(n_poses=args.max_frames)
            drift_factor = 0.01 * (active_baselines.index(name) + 1)
            for i, p in enumerate(base_poses):
                p[0, 3] = (0.1 - drift_factor) * i
                p[2, 3] = drift_factor * i
            
            label = BASELINES[name]["label"]
            metrics = compute_metrics_all(base_poses, gt_poses, label=label)
            results.append(metrics)
            baseline_poses_dict[name] = base_poses
            
        report_path = os.path.join(args.output_dir, "comparison_report.md")
        generate_report(results, report_path, active_baselines=active_baselines, mode_name=args.mode)
        
        plot_path = os.path.join(args.output_dir, "trajectory_comparison.png")
        # Align mock baselines for plotting
        baselines_aligned = {}
        for r in results[1:]:
            for name, b_info in BASELINES.items():
                if b_info["label"] == r["label"]:
                    baselines_aligned[name] = r["aligned_poses"]
                    
        plot_trajectory_comparison(ours=est_poses, baselines_dict=baselines_aligned, gt=gt_poses, output_path=plot_path)
        
        error_plot_path = os.path.join(args.output_dir, "error_comparison.png")
        plot_errors_comparison(results, error_plot_path)
        print("Mock mode completed successfully.")
        
    elif args.mode == "ci":
        print("Running in CI mode (Synthetic VO pipeline)...")
        
        # 1. Generate Synthetic circular trajectory
        n_frames = min(args.max_frames, 20)
        print(f"Generating synthetic 'mixed' orbit trajectory ({n_frames} frames)...")
        dataset = SyntheticVODataset(scenario="mixed", n_frames=n_frames, n_points=600)
        
        # Save to temp directory
        temp_dir = tempfile.mkdtemp(prefix="vo_synth_ci_")
        try:
            dataset.save(temp_dir)
            print(f"Synthetic dataset saved to temp directory: {temp_dir}")
            
            # 2. Load dataset
            loader = KITTIFrameLoader(temp_dir, max_frames=n_frames, use_calib_intrinsics=True)
            gt_poses = loader.get_ground_truth()
            
            # 3. Run slam_dnn
            est_poses = run_slam_dnn_on_loader(
                loader, extractor=args.extractor, matcher=args.matcher, max_keypoints=300, device=args.device, min_matches=8, target_resolution=args.target_resolution,
                use_joint_ba=args.use_joint_ba, ba_window_size=args.ba_window_size,
                use_depth_prior=args.use_depth_prior, depth_source=args.depth_source,
                depth_directory=args.depth_directory, depth_scale_factor=args.depth_scale_factor,
                depth_model_name=args.depth_model_name, depth_scale_mode=args.depth_scale_mode,
                min_parallax=args.min_parallax, max_overlap=args.max_overlap, max_keyframe_interval=args.max_keyframe_interval
            )
            
            ours_report = compute_metrics_all(est_poses, gt_poses, label="Ours (slam_dnn)")
            results.append(ours_report)
            
            # 4. Run active baselines
            active_baselines = get_active_baselines(args)
            evaluated_baselines = []
            
            for name in active_baselines:
                b_info = BASELINES[name]
                if b_info["check"]():
                    print(f"Running baseline {b_info['label']}...")
                    b_out = os.path.join(temp_dir, f"{name}_out")
                    b_poses = b_info["run"](
                        temp_dir, b_out, True, n_frames
                    )
                    if len(b_poses) >= 3:
                        b_report = compute_metrics_all(b_poses, gt_poses, label=b_info["label"])
                        results.append(b_report)
                        evaluated_baselines.append(name)
                    else:
                        print(f"Warning: {b_info['label']} produced insufficient poses.")
                else:
                    print(f"Note: {b_info['label']} is not installed or importable. Skipping.")
            
            # 5. Generate outputs
            report_path = os.path.join(args.output_dir, "comparison_report.md")
            generate_report(results, report_path, active_baselines=evaluated_baselines, mode_name=args.mode)
            
            plot_path = os.path.join(args.output_dir, "trajectory_comparison.png")
            aligned_ours = ours_report["aligned_poses"]
            
            baselines_aligned = {}
            for r in results[1:]:
                for name, b_info in BASELINES.items():
                    if b_info["label"] == r["label"]:
                        baselines_aligned[name] = r["aligned_poses"]
                        
            plot_trajectory_comparison(ours=aligned_ours, baselines_dict=baselines_aligned, gt=gt_poses, output_path=plot_path)
            
            error_plot_path = os.path.join(args.output_dir, "error_comparison.png")
            plot_errors_comparison(results, error_plot_path)
            
            print("CI mode completed successfully.")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    elif args.mode == "full":
        print(f"Running in Full Mode (Real Dataset: {args.dataset})...")
        
        # 1. Resolve dataset path
        if args.dataset == "kitti05":
            data_path = os.path.join("data", "kitti", "05")
        else:
            data_path = os.path.join("data", "parking")
            
        if not os.path.isdir(data_path):
            print(f"Error: Dataset directory not found at {data_path}")
            print("Please ensure real datasets are downloaded or download them using:")
            print("  python scripts/download_data.py --dataset kitti05")
            sys.exit(1)
            
        # 2. Initialize loader
        print(f"Loading dataset from {data_path} (max_frames={args.max_frames})...")
        loader = KITTIFrameLoader(data_path, max_frames=args.max_frames, use_calib_intrinsics=True)
        gt_poses = loader.get_ground_truth()
        
        if gt_poses is None:
            print("Error: Poses file poses.txt is missing from dataset directory. Trajectory cannot be evaluated.")
            sys.exit(1)
            
        # 3. Run slam_dnn
        est_poses = run_slam_dnn_on_loader(
            loader, extractor=args.extractor, matcher=args.matcher, device=args.device, target_resolution=args.target_resolution,
            use_joint_ba=args.use_joint_ba, ba_window_size=args.ba_window_size,
            use_depth_prior=args.use_depth_prior, depth_source=args.depth_source,
            depth_directory=args.depth_directory, depth_scale_factor=args.depth_scale_factor,
            depth_model_name=args.depth_model_name, depth_scale_mode=args.depth_scale_mode,
            min_parallax=args.min_parallax, max_overlap=args.max_overlap, max_keyframe_interval=args.max_keyframe_interval
        )
        
        ours_report = compute_metrics_all(est_poses, gt_poses, label="Ours (slam_dnn)")
        results.append(ours_report)
        
        # 4. Run active baselines
        active_baselines = get_active_baselines(args)
        evaluated_baselines = []
        
        for name in active_baselines:
            b_info = BASELINES[name]
            if b_info["check"]():
                print(f"Running baseline {b_info['label']}...")
                b_out = os.path.join(args.output_dir, f"{name}_full")
                b_poses = b_info["run"](
                    data_path, b_out, True, args.max_frames
                )
                if len(b_poses) >= 3:
                    b_report = compute_metrics_all(b_poses, gt_poses, label=b_info["label"])
                    results.append(b_report)
                    evaluated_baselines.append(name)
                else:
                    print(f"Warning: {b_info['label']} produced insufficient poses.")
            else:
                print(f"Note: {b_info['label']} is not installed or importable. Skipping.")
                
        # 5. Generate outputs
        report_path = os.path.join(args.output_dir, "comparison_report.md")
        generate_report(results, report_path, active_baselines=evaluated_baselines, mode_name=args.mode)
        
        plot_path = os.path.join(args.output_dir, "trajectory_comparison.png")
        aligned_ours = ours_report["aligned_poses"]
        
        baselines_aligned = {}
        for r in results[1:]:
            for name, b_info in BASELINES.items():
                if b_info["label"] == r["label"]:
                    baselines_aligned[name] = r["aligned_poses"]
                    
        plot_trajectory_comparison(ours=aligned_ours, baselines_dict=baselines_aligned, gt=gt_poses, output_path=plot_path)
        
        error_plot_path = os.path.join(args.output_dir, "error_comparison.png")
        plot_errors_comparison(results, error_plot_path)
        
        print("Full mode completed successfully.")


if __name__ == "__main__":
    main()
