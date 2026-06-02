"""Wrapper for the minislam baseline VO system.

This module provides an adapter to run minislam's monocular visual odometry
pipeline on KITTI-format datasets for baseline comparison with slam_dnn.

IMPORTANT:
- This file is part of slam_dnn, NOT minislam.
- No slam_dnn core module (slam_dnn.*) should import from this file.
- minislam imports happen at runtime to allow graceful failure when
  the package is not installed.

minislam uses ORB features + Essential matrix decomposition for pose
estimation, with loop closure detection via descriptor similarity.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MINISLAM_SUBMODULE = Path(__file__).parent / "minislam"
MINISLAM_SRC = MINISLAM_SUBMODULE / "src"
if MINISLAM_SRC.is_dir() and str(MINISLAM_SRC) not in sys.path:
    sys.path.insert(0, str(MINISLAM_SRC))

PINNED_COMMIT = "962096d5bb8919317cceef9c0f2f98f023d9fcf3"


def check_minislam_available() -> bool:
    """Returns True if minislam is installed and importable.

    Checks that the core minislam modules (camera, odometry) can be
    imported. Does not verify that all optional dependencies (pygame,
    pyopengl for 3D display) are present.
    """
    try:
        from minislam.camera import Camera  # noqa: F401
        from minislam.odometry import VisualOdometry  # noqa: F401
        from minislam.dataset import ImageLoader  # noqa: F401
        return True
    except ImportError:
        return False


def _parse_kitti_calib(calib_path: str) -> dict[str, float]:
    """Parse KITTI calibration file and extract left camera intrinsics.

    KITTI calib.txt contains 3x4 projection matrices P0, P1, etc.
    P0 is the left gray camera. The intrinsic matrix K is embedded in
    the first 3 columns of P0::

        P0 = [fx  0  cx  0 ]
             [ 0  fy cy  0 ]
             [ 0  0  1   0 ]

    Args:
        calib_path: Path to KITTI calib.txt file.

    Returns:
        Dict with keys: fx, fy, cx, cy

    Raises:
        ValueError: If P0 entry is not found in the calibration file.
        FileNotFoundError: If calib_path does not exist.
    """
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    calib_data: dict[str, list[float]] = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 13:
                key = parts[0].rstrip(":")
                values = [float(v) for v in parts[1:13]]
                calib_data[key] = values

    if "P0" not in calib_data:
        raise ValueError(f"P0 not found in {calib_path}")

    p = calib_data["P0"]
    # Row-major 3x4:
    #   [p[0]  p[1]  p[2]  p[3] ]    fx=p[0], cx=p[2]
    #   [p[4]  p[5]  p[6]  p[7] ]    fy=p[5], cy=p[6]
    #   [p[8]  p[9]  p[10] p[11]]
    return {
        "fx": p[0],
        "fy": p[5],
        "cx": p[2],
        "cy": p[6],
    }


def _get_image_dimensions(image_dir: str) -> tuple[int, int]:
    """Read (width, height) from the first PNG image in a directory.

    Args:
        image_dir: Directory containing PNG images.

    Returns:
        (width, height) tuple.

    Raises:
        ValueError: If no PNG images found or first image unreadable.
    """
    import cv2

    images = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".png"))
    if not images:
        raise ValueError(f"No PNG images found in {image_dir}")

    img = cv2.imread(os.path.join(image_dir, images[0]))
    if img is None:
        raise ValueError(f"Could not read image: {images[0]}")

    h, w = img.shape[:2]
    return w, h


def run_minislam_on_kitti(
    data_dir: str,
    output_dir: str,
    use_calib_intrinsics: bool = True,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Run minislam on a KITTI-format dataset and return estimated poses.

    Loads images from ``data_dir/image_0/``, reads camera intrinsics from
    ``data_dir/calib.txt`` (when *use_calib_intrinsics* is True), runs the
    minislam visual odometry pipeline, and writes a KITTI-format trajectory
    to ``output_dir/minislam_trajectory.txt``.

    Args:
        data_dir: Path to KITTI sequence directory.  Expected layout::

            data_dir/
                image_0/    # PNG images (000000.png, 000001.png, ...)
                calib.txt   # Camera calibration (P0, P1 matrices)
                poses.txt   # (optional, ground truth - not used here)

        output_dir: Directory where output files are written.
        use_calib_intrinsics: If True, parse intrinsics from calib.txt.
            If False, derive approximate intrinsics from image dimensions
            assuming a 90-degree horizontal FOV.
        max_frames: Process at most this many frames. None = all frames.

    Returns:
        List of 4x4 estimated pose matrices (np.ndarray, float64).
        Returns an empty list if minislam is not installed, the data
        directory is invalid, or the pipeline fails.
    """
    # Lazy imports -- graceful failure when minislam is missing
    try:
        import cv2
        from minislam.camera import Camera
        from minislam.dataset import ImageLoader
        from minislam.odometry import VisualOdometry
    except ImportError as e:
        logger.error("minislam not available: %s", e)
        return []

    try:
        return _run_pipeline(
            data_dir, output_dir, use_calib_intrinsics, max_frames,
            cv2, Camera, ImageLoader, VisualOdometry,
        )
    except Exception as e:
        logger.error("minislam pipeline failed: %s", e)
        return []


def _run_pipeline(
    data_dir: str,
    output_dir: str,
    use_calib_intrinsics: bool,
    max_frames: int | None,
    cv2,
    Camera,
    ImageLoader,
    VisualOdometry,
) -> list[np.ndarray]:
    """Internal pipeline runner, separated for clean exception handling."""
    image_dir = os.path.join(data_dir, "image_0")
    calib_path = os.path.join(data_dir, "calib.txt")

    if not os.path.isdir(image_dir):
        logger.error("Image directory not found: %s", image_dir)
        return []

    # Determine intrinsics
    if use_calib_intrinsics and os.path.isfile(calib_path):
        intrinsics = _parse_kitti_calib(calib_path)
        logger.info(
            "Using calib.txt intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            intrinsics["fx"], intrinsics["fy"],
            intrinsics["cx"], intrinsics["cy"],
        )
    else:
        if not use_calib_intrinsics:
            logger.info("Using FOV-based intrinsics (calib ignored)")
        else:
            logger.info("calib.txt not found, falling back to FOV-based intrinsics")
        w, h = _get_image_dimensions(image_dir)
        # Approximate: 90-degree horizontal FOV
        fx = fy = w / (2.0 * np.tan(np.radians(45.0)))
        cx = w / 2.0
        cy = h / 2.0
        intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    # Image dimensions
    width, height = _get_image_dimensions(image_dir)

    # Build minislam camera and odometry (without display/loop-closure)
    camera = Camera(
        width, height,
        intrinsics["fx"], intrinsics["fy"],
        intrinsics["cx"], intrinsics["cy"],
    )
    vo = VisualOdometry(camera, enable_loop_closure=False)

    # Load and process frames
    loader = ImageLoader(image_dir)
    poses: list[np.ndarray] = []
    frame_count = 0
    success_count = 0

    for i, img in enumerate(loader):
        if max_frames is not None and i >= max_frames:
            break

        img_resized = cv2.resize(img, (width, height))
        frame_count += 1

        try:
            vo.process_frame(img_resized, i)
            # vo.poses accumulates 4x4 matrices starting from frame 1
            if vo.poses:
                poses.append(vo.poses[-1].copy())
                success_count += 1
        except Exception as e:
            # minislam raises on < 8 matches (assert in get_matches)
            logger.debug("Frame %d skipped: %s", i, e)
            continue

    logger.info(
        "minislam: %d/%d frames successful",
        success_count, frame_count,
    )

    # Write KITTI format trajectory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "minislam_trajectory.txt")
    with open(output_path, "w") as f:
        for pose in poses:
            # KITTI format: 12 floats per line (row-major 3x4 from 4x4 matrix)
            row = pose[:3, :].ravel()
            f.write(" ".join(f"{v:.9e}" for v in row) + "\n")

    logger.info("Trajectory saved: %s (%d poses)", output_path, len(poses))
    return poses
