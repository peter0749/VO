#!/usr/bin/env python3
"""End-to-end SuperPoint Visual Odometry pipeline.

Processes an image sequence and outputs camera trajectory in KITTI and TUM
formats, plus a trajectory plot.

Usage:
    python run_vo.py --input tests/fixtures/kitti_05_subset \\
                     --output /tmp/vo_test \\
                     --fov 63 --matcher lightglue --verbose
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path

from slam_dnn import (
    VisualOdometry,
    PinholeCamera,
    export_kitti_format,
    export_tum_format,
    load_kitti_format,
    load_tum_format,
)
from slam_dnn.visualization import plot_trajectory_comparison


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def load_ground_truth_tum(filepath: Path) -> list:
    """Load ground truth trajectory in TUM format (timestamp tx ty tz qx qy qz qw).

    Returns:
        List of 4x4 SE3 matrices.
    """
    poses, _ = load_tum_format(str(filepath))
    return poses


def load_ground_truth_kitti(filepath: Path) -> list:
    """Load ground truth trajectory in KITTI format (12 floats per line).

    Returns:
        List of 4x4 SE3 matrices.
    """
    return load_kitti_format(str(filepath))


def load_ground_truth(filepath: Path) -> list:
    """Auto-detect and load ground truth trajectory (TUM or KITTI format).

    TUM format: lines have 8 values (timestamp + xyz + quat)
    KITTI format: lines have 12 values (3x4 row-major)
    """
    with open(filepath) as f:
        first_line = f.readline().strip()
    n_vals = len(first_line.split())
    if n_vals == 8:
        return load_ground_truth_tum(filepath)
    elif n_vals == 12:
        return load_ground_truth_kitti(filepath)
    else:
        raise ValueError(
            f"Cannot detect ground truth format: first line has {n_vals} values "
            f"(expected 8 for TUM or 12 for KITTI)"
        )


def plot_trajectory(poses: list, output_path: Path, gt_poses: list | None = None) -> None:
    """Generate top-down trajectory plot with start/end markers.

    Uses slam_dnn.visualization.plot_trajectory_comparison for
    publication-quality output with equal aspect ratio, grid, and legend.

    Args:
        poses: List of 4x4 SE3 matrices (estimated trajectory).
        output_path: Path to save PNG plot.
        gt_poses: Optional list of 4x4 SE3 matrices (ground truth).
    """
    if not poses:
        return

    plot_trajectory_comparison(
        estimated=poses,
        ground_truth=gt_poses,
        title="Camera Trajectory (Top-Down View)",
        save_path=str(output_path),
        show=False,
    )


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def load_image_sequence(input_dir: Path) -> list:
    """Return sorted list of image file paths in directory."""
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args) -> None:
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device selection ---
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # --- Load image sequence ---
    image_paths = load_image_sequence(input_dir)
    if not image_paths:
        print(f"ERROR: No images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    n_frames = len(image_paths)
    print(f"Found {n_frames} frames in {input_dir}")
    print(f"Device: {device}, Matcher: {args.matcher}, FOV: {args.fov}°")

    # --- Load first image to get dimensions for camera ---
    img0 = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if img0 is None:
        print(f"ERROR: Could not read {image_paths[0]}", file=sys.stderr)
        sys.exit(1)
    h, w = img0.shape[:2]
    del img0

    # --- Initialize VisualOdometry facade ---
    cam = PinholeCamera(width=w, height=h, fov_deg=args.fov)
    vo = VisualOdometry(
        cam,
        matcher=args.matcher,
        max_keypoints=args.max_keypoints,
        scale=args.scale,
        device=device,
    )

    # --- Pipeline ---
    frame_timestamps = [0.0]

    print(f"\nProcessing {n_frames} frames...")
    t_start = time.time()

    for i in range(n_frames):
        img = cv2.imread(str(image_paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            if args.verbose:
                print(f"  Frame {i}: SKIP (could not read)")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pose = vo.process_frame(gray)

        if pose is not None:
            frame_timestamps.append(float(i))
            if args.verbose:
                pos = vo.get_trajectory().get_positions()[-1]
                print(
                    f"  Frame {i}: OK "
                    f"| pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
                )
        else:
            if args.verbose and i > 0:
                print(f"  Frame {i}: tracking failed")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")

    # --- Summary ---
    stats = vo.get_stats()
    total_processed = n_frames - 1
    print(f"\nSummary:")
    print(f"  Total frames:      {n_frames}")
    print(f"  Processed pairs:   {total_processed}")
    print(f"  Successful:        {stats['successful']}")
    print(f"  Tracking lost:     {stats['tracking_lost']}")
    print(f"  Pose failed:       {stats['pose_failed']}")

    # --- Save outputs ---
    poses = vo.get_trajectory().get_poses()

    kitti_path = output_dir / "trajectory_kitti.txt"
    tum_path = output_dir / "trajectory_tum.txt"
    plot_path = output_dir / "trajectory_plot.png"

    export_kitti_format(poses, kitti_path)
    export_tum_format(poses, tum_path, timestamps=frame_timestamps)

    # --- Load optional ground truth ---
    gt_poses = None
    if getattr(args, "ground_truth", None):
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            gt_poses = load_ground_truth(gt_path)
            print(f"  Loaded ground truth: {len(gt_poses)} poses from {gt_path}")
        else:
            print(f"WARNING: Ground truth file not found: {gt_path}", file=sys.stderr)

    if not getattr(args, "no_plot", False):
        plot_trajectory(poses, plot_path, gt_poses=gt_poses)

    print(f"\nOutputs:")
    print(f"  KITTI: {kitti_path} ({len(poses)} poses)")
    print(f"  TUM:   {tum_path} ({len(poses)} poses)")
    if not getattr(args, "no_plot", False):
        print(f"  Plot:  {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SuperPoint Visual Odometry Pipeline"
    )
    parser.add_argument(
        "--input", required=True, help="Directory containing PNG/JPG images"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for trajectory files"
    )
    parser.add_argument(
        "--fov", type=float, default=63.0, help="Camera FOV in degrees (default: 63)"
    )
    parser.add_argument(
        "--matcher",
        choices=["lightglue", "classic"],
        default="lightglue",
        help="Matching backend (default: lightglue)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device selection (default: auto)",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=1024,
        help="Max keypoints per frame (default: 1024)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Trajectory scale factor (default: 1.0)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed per-frame info"
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to ground truth trajectory file (TUM or KITTI format)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip trajectory plot generation (faster for batch runs)",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
