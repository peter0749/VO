#!/usr/bin/env python3
"""CLI wrapper for generating synthetic visual odometry datasets.

Generates a synthetic VO dataset with known ground truth and saves it
to disk in KITTI-compatible format (images directory + poses.txt + calib.txt).

Usage:
    python scripts/generate_synthetic.py --scenario mixed --n-frames 20
    python scripts/generate_synthetic.py --scenario translation --n-frames 50 --output data/synthetic/trans
    python scripts/generate_synthetic.py --scenario rotation --n-points 500 --noise 1.0

Output structure:
    <output_dir>/
        images/
            000000.png
            000001.png
            ...
        poses.txt     # KITTI format: 12 floats per line (3x4 row-major)
        calib.txt     # Camera intrinsics in KITTI calibration format

The output can be loaded directly by slam_dnn.io.FrameLoader (images)
and slam_dnn.export.load_kitti_format (poses).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from slam_dnn.testdata.synthetic import SyntheticVODataset


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic visual odometry datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        choices=SyntheticVODataset.VALID_SCENARIOS,
        default="mixed",
        help="Camera motion scenario (default: mixed)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=50,
        help="Number of frames to generate (default: 50)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=300,
        help="Number of 3D points in scene (default: 300)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic",
        help="Output directory path (default: data/synthetic)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height in pixels (default: 480)",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=63.0,
        help="Horizontal FOV in degrees (default: 63.0)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.5,
        help="Gaussian noise sigma in pixels (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--format",
        choices=["kitti"],
        default="kitti",
        help="Output format (default: kitti)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    Args:
        argv: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    print(f"Generating synthetic VO dataset:")
    print(f"  Scenario:  {args.scenario}")
    print(f"  Frames:    {args.n_frames}")
    print(f"  Points:    {args.n_points}")
    print(f"  Size:      {args.width}x{args.height}")
    print(f"  FOV:       {args.fov}°")
    print(f"  Noise:     {args.noise}px")
    print(f"  Seed:      {args.seed}")
    print(f"  Output:    {args.output}")
    print()

    t0 = time.time()
    try:
        ds = SyntheticVODataset(
            scenario=args.scenario,
            n_frames=args.n_frames,
            n_points=args.n_points,
            image_size=(args.width, args.height),
            fov_deg=args.fov,
            noise_px=args.noise,
            seed=args.seed,
        )
        ds.save(args.output, format=args.format)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    print(f"Done! Generated {args.n_frames} frames in {elapsed:.1f}s")
    print(f"Output saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
