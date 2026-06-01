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
    SuperPointExtractor,
    LightGlueMatcher,
    ClassicMatcher,
    estimate_essential,
    TrajectoryAccumulator,
    K_from_fov,
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_trajectory_kitti(poses: list, filepath: Path) -> None:
    """Save trajectory in KITTI format (12 floats per line, 3x4 row-major)."""
    with open(filepath, "w") as f:
        for T in poses:
            T_3x4 = T[:3, :]
            f.write(" ".join(f"{x:.6f}" for x in T_3x4.flatten()) + "\n")


def save_trajectory_tum(poses: list, timestamps: list, filepath: Path) -> None:
    """Save trajectory in TUM format (timestamp tx ty tz qx qy qz qw)."""
    from scipy.spatial.transform import Rotation

    with open(filepath, "w") as f:
        for t, T in zip(timestamps, poses):
            tx, ty, tz = T[:3, 3]
            quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
            f.write(
                f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )


def plot_trajectory(poses: list, output_path: Path) -> None:
    """Generate top-down trajectory plot with start/end markers."""
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt

    if not poses:
        return

    # Use camera center positions (world frame) for the plot
    positions = []
    for T in poses:
        R = T[:3, :3]
        t = T[:3, 3]
        c_world = -R.T @ t
        positions.append(c_world)
    positions = np.array(positions)

    x = positions[:, 0]
    z = positions[:, 2]

    plt.figure(figsize=(10, 10))
    plt.plot(x, z, "k-", linewidth=1, label="trajectory")
    plt.plot(x[0], z[0], "go", markersize=12, label="start", zorder=10)
    plt.plot(x[-1], z[-1], "ro", markersize=12, label="end", zorder=10)

    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Camera Trajectory (Top-Down View)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


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

    # --- Load first image to get dimensions for K ---
    img0 = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if img0 is None:
        print(f"ERROR: Could not read {image_paths[0]}", file=sys.stderr)
        sys.exit(1)
    h, w = img0.shape[:2]
    K = K_from_fov(w, h, fov_deg=args.fov)
    del img0  # free memory

    # --- Initialize components ---
    extractor = SuperPointExtractor(
        max_keypoints=args.max_keypoints, device=device
    )
    if args.matcher == "lightglue":
        matcher = LightGlueMatcher(device=device)
    else:
        matcher = ClassicMatcher()

    trajectory = TrajectoryAccumulator(scale=args.scale)

    # --- Pipeline state ---
    tracking_lost = 0
    pose_failed = 0
    successful = 0
    frame_timestamps = [0.0]

    print(f"\nProcessing {n_frames} frames...")
    t_start = time.time()

    prev_feats = None

    for i in range(n_frames):
        img = cv2.imread(str(image_paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            if args.verbose:
                print(f"  Frame {i}: SKIP (could not read)")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = extractor.extract(gray)

        # LightGlueMatcher expects descriptors in (D, N) format
        # (SuperPointExtractor returns (N, D); the matcher transposes again)
        if args.matcher == "lightglue":
            feats["descriptors"] = feats["descriptors"].T

        if prev_feats is None:
            prev_feats = feats
            if args.verbose:
                print(f"  Frame {i}: first frame, {len(feats['keypoints'])} keypoints")
            continue

        match_result = matcher.match(prev_feats, feats)
        n_matches = len(match_result["points0"])

        if n_matches < 20:
            tracking_lost += 1
            if args.verbose:
                print(f"  Frame {i}: tracking_lost ({n_matches} matches < 20)")
            prev_feats = feats
            continue

        result = estimate_essential(
            match_result["points0"], match_result["points1"], K
        )

        if result is None:
            pose_failed += 1
            if args.verbose:
                print(f"  Frame {i}: pose_failed")
            prev_feats = feats
            continue

        R, t, inlier_mask = result
        trajectory.add_pose(R, t)
        successful += 1
        frame_timestamps.append(float(i))

        if args.verbose:
            pos = trajectory.get_positions()[-1]
            print(
                f"  Frame {i}: OK | matches={n_matches} inliers={inlier_mask.sum()} "
                f"| pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
            )

        prev_feats = feats

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")

    # --- Summary ---
    total_processed = n_frames - 1  # first frame is reference only
    print(f"\nSummary:")
    print(f"  Total frames:      {n_frames}")
    print(f"  Processed pairs:   {total_processed}")
    print(f"  Successful:        {successful}")
    print(f"  Tracking lost:     {tracking_lost}")
    print(f"  Pose failed:       {pose_failed}")

    # --- Save outputs ---
    poses = trajectory.get_poses()

    kitti_path = output_dir / "trajectory_kitti.txt"
    tum_path = output_dir / "trajectory_tum.txt"
    plot_path = output_dir / "trajectory_plot.png"

    save_trajectory_kitti(poses, kitti_path)
    save_trajectory_tum(poses, frame_timestamps, tum_path)
    plot_trajectory(poses, plot_path)

    print(f"\nOutputs:")
    print(f"  KITTI: {kitti_path} ({len(poses)} poses)")
    print(f"  TUM:   {tum_path} ({len(poses)} poses)")
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

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
