"""Command-line interface for slam_dnn visual odometry."""
import argparse
import logging
import sys
import time
from pathlib import Path

from . import VisualOdometry, PinholeCamera
from .io import FrameLoader
from .export import export_kitti_format, export_tum_format, load_kitti_format, load_tum_format
from .eval import evaluate
from .visualization import plot_trajectory_comparison

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="slam_dnn",
        description="SuperPoint-based visual odometry pipeline",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Image directory or video file path",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for trajectory files",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=63.0,
        help="Horizontal FOV in degrees (default: 63.0, phone wide-angle)",
    )
    parser.add_argument(
        "--matcher",
        choices=["lightglue", "classic"],
        default="lightglue",
        help="Matcher to use (default: lightglue)",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=2048,
        help="Max keypoints per frame (default: 2048)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Trajectory scale factor (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip trajectory plot generation",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional ground truth trajectory for evaluation (KITTI or TUM format)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Verbose logging (-v for INFO, -vv for DEBUG)",
    )
    return parser


# ---------------------------------------------------------------------------
# Ground truth loading helpers
# ---------------------------------------------------------------------------

def load_ground_truth(filepath: Path) -> list:
    """Auto-detect and load ground truth trajectory (TUM or KITTI format).

    TUM format: lines have 8 values (timestamp + xyz + quat)
    KITTI format: lines have 12 values (3x4 row-major)
    """
    with open(filepath) as f:
        first_line = f.readline().strip()
    n_vals = len(first_line.split())
    if n_vals == 8:
        poses, _ = load_tum_format(str(filepath))
        return poses
    elif n_vals == 12:
        return load_kitti_format(str(filepath))
    else:
        raise ValueError(
            f"Cannot detect ground truth format: first line has {n_vals} values "
            f"(expected 8 for TUM or 12 for KITTI)"
        )


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(args_device: str) -> str:
    """Select compute device based on args and availability."""
    import torch
    if args_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return args_device


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> int:
    """Run the visual odometry pipeline. Returns 0 on success, 1 on failure."""
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device selection ---
    device = select_device(args.device)

    # --- Load image sequence ---
    try:
        loader = FrameLoader(str(args.input))
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load input: {e}")
        return 1

    n_frames = len(loader)
    logger.info(f"Found {n_frames} frames in {args.input}")
    logger.info(f"Device: {device}, Matcher: {args.matcher}, FOV: {args.fov}°")

    # --- Load first frame to get camera dimensions ---
    first_frame = next(iter(loader))
    h, w = first_frame.shape[:2]
    logger.info(f"Frame dimensions: {w}x{h}")

    # --- Initialize VisualOdometry ---
    cam = PinholeCamera(width=w, height=h, fov_deg=args.fov)
    vo = VisualOdometry(
        camera=cam,
        matcher=args.matcher,
        max_keypoints=args.max_keypoints,
        scale=args.scale,
        device=device,
    )

    # --- Pipeline ---
    frame_timestamps = [0.0]

    logger.info(f"\nProcessing {n_frames} frames...")
    t_start = time.time()

    # Re-iterate loader to process frames
    loader = FrameLoader(str(args.input))
    for i, img in enumerate(loader):
        pose = vo.process_frame(img)
        if pose is not None:
            frame_timestamps.append(float(i))
            if args.verbose >= 2:
                pos = vo.get_trajectory().get_positions()[-1]
                logger.debug(
                    f"  Frame {i}: OK "
                    f"| pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
                )
        else:
            if args.verbose >= 1:
                logger.info(f"  Frame {i}: initial frame, establishing baseline")

    elapsed = time.time() - t_start
    logger.info(f"\nDone in {elapsed:.1f}s")

    # --- Summary ---
    stats = vo.get_stats()
    total_processed = n_frames - 1
    logger.info("\nSummary:")
    logger.info(f"  Total frames:      {n_frames}")
    logger.info(f"  Processed pairs:   {total_processed}")
    logger.info(f"  Successful:        {stats['successful']}")
    logger.info(f"  Tracking lost:     {stats['tracking_lost']}")
    logger.info(f"  Pose failed:       {stats['pose_failed']}")

    # --- Save outputs ---
    poses = vo.get_trajectory().get_poses()

    kitti_path = output_dir / "trajectory_kitti.txt"
    tum_path = output_dir / "trajectory_tum.txt"

    export_kitti_format(poses, str(kitti_path))
    export_tum_format(poses, str(tum_path), timestamps=frame_timestamps)

    # --- Load optional ground truth ---
    gt_poses = None
    if args.ground_truth:
        gt_path = args.ground_truth
        if gt_path.exists():
            try:
                gt_poses = load_ground_truth(gt_path)
                logger.info(f"  Loaded ground truth: {len(gt_poses)} poses from {gt_path}")
            except Exception as e:
                logger.warning(f"Failed to load ground truth: {e}")
        else:
            logger.warning(f"Ground truth file not found: {gt_path}")

    # --- Plot ---
    if not args.no_plot:
        try:
            plot_path = output_dir / "trajectory_plot.png"
            plot_trajectory_comparison(
                estimated=poses,
                ground_truth=gt_poses,
                title="Camera Trajectory (Top-Down View)",
                save_path=str(plot_path),
                show=False,
            )
            logger.info(f"  Plot:  {plot_path}")
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")

    # --- Evaluation ---
    if gt_poses is not None:
        try:
            eval_result = evaluate(poses, gt_poses, with_scale=True)
            logger.info(f"Evaluation: APE RMSE = {eval_result['ape_rmse']:.4f}, "
                       f"RTE RMSE = {eval_result['rte_rmse']:.4f}, "
                       f"scale = {eval_result['scale']:.4f}")
            # Save evaluation report
            eval_path = output_dir / "evaluation_report.txt"
            with open(eval_path, "w") as f:
                f.write("Trajectory Evaluation Report\n")
                f.write(f"{'=' * 40}\n")
                f.write("APE (Absolute Pose Error):\n")
                f.write(f"  RMSE: {eval_result['ape_rmse']:.4f}\n")
                f.write(f"  Mean: {eval_result['ape_mean']:.4f}\n")
                f.write("RTE (Relative Trajectory Error):\n")
                f.write(f"  RMSE: {eval_result['rte_rmse']:.4f}\n")
                f.write(f"  Mean: {eval_result['rte_mean']:.4f}\n")
                f.write(f"Umeyama Scale: {eval_result['scale']:.4f}\n")
                f.write(f"Num Estimated Poses: {eval_result['num_frames']}\n")
                f.write(f"Num Ground Truth Poses: {len(gt_poses)}\n")
            logger.info(f"  Evaluation report: {eval_path}")
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

    logger.info("\nOutputs:")
    logger.info(f"  KITTI: {kitti_path} ({len(poses)} poses)")
    logger.info(f"  TUM:   {tum_path} ({len(poses)} poses)")

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, non-zero on failure."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging based on verbosity
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=log_level,
    )

    # Validate input
    if not args.input.exists():
        logger.error(f"Input path does not exist: {args.input}")
        return 1

    logger.info("Starting SLAM-DNN visual odometry")

    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
