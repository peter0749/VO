"""KITTI odometry dataset loader with calibration and ground truth support."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator

from .camera import K_from_fov


class KITTIFrameLoader:
    """Load KITTI odometry sequences with calibration and ground truth poses.

    Supports two directory layouts:

    - **Flat format**: ``base_dir/image_0/``, ``base_dir/calib.txt``,
      ``base_dir/poses.txt``
    - **Nested format**: ``base_dir/sequences/XX/image_0/``,
      ``base_dir/sequences/XX/calib.txt``, ``base_dir/sequences/XX/poses.txt``
      where ``XX`` is the sequence number (e.g., "05").

    The loader auto-detects the format by checking for the flat layout first.
    If ``base_dir/image_0/`` exists, flat format is used. Otherwise, it falls
    back to the nested layout.

    Calibration is read from ``calib.txt`` which contains projection matrices
    in KITTI format (``P0: f1 f2 ... f12``). The intrinsic matrix K is
    extracted as the left 3x3 block of P0. If calibration is missing or
    ``use_calib_intrinsics`` is False, a fallback K matrix is computed from
    image dimensions and a default field of view (63 degrees).

    Ground truth poses are read from ``poses.txt`` (12 floats per line,
    row-major 3x4 matrix). If the file is missing, ``get_ground_truth()``
    returns None.

    Args:
        base_dir: Path to KITTI data root directory.
        sequence: Sequence identifier (e.g., "05"). Used for nested format
            to locate ``sequences/{sequence}/``. Kept for API compatibility.
        max_frames: If provided, limit iteration to this many frames.
        use_calib_intrinsics: If True (default), read intrinsics from
            ``calib.txt``. If False or calib.txt is missing, use
            ``K_from_fov(width, height, fov_deg=63)`` as fallback.

    Raises:
        FileNotFoundError: If base_dir does not exist or image directory
            is not found in either flat or nested format.

    Example:
        >>> loader = KITTIFrameLoader('path/to/kitti/', max_frames=10)
        >>> K = loader.get_intrinsics()
        >>> poses = loader.get_ground_truth()
        >>> for frame in loader:
        ...     img = frame['image']
        ...     gt = frame['gt_pose']  # 4x4 matrix or None
    """

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}

    def __init__(
        self,
        base_dir: str,
        sequence: str = "05",
        max_frames: int | None = None,
        use_calib_intrinsics: bool = True,
    ):
        base_path = Path(base_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")

        # Auto-detect format: flat vs nested
        flat_image_dir = base_path / "image_0"
        if flat_image_dir.is_dir():
            self._sequence_dir = base_path
        else:
            nested_dir = base_path / "sequences" / sequence
            if nested_dir.is_dir():
                self._sequence_dir = nested_dir
            else:
                raise FileNotFoundError(
                    f"Cannot find image_0 directory. Checked:\n"
                    f"  - Flat: {flat_image_dir}\n"
                    f"  - Nested: {nested_dir / 'image_0'}"
                )

        self._image_dir = self._sequence_dir / "image_0"
        if not self._image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self._image_dir}")

        self.base_dir = base_path
        self.sequence = sequence
        self.max_frames = max_frames
        self.use_calib_intrinsics = use_calib_intrinsics

        # Discover image files
        self._images = sorted(
            p for p in self._image_dir.iterdir()
            if p.suffix.lower() in self._IMAGE_EXTENSIONS
        )
        if not self._images:
            raise ValueError(f"No image files found in {self._image_dir}")

        # Load calibration and poses
        self._calib_path = self._sequence_dir / "calib.txt"
        self._poses_path = self._sequence_dir / "poses.txt"

        self._calib = None
        if self._calib_path.exists():
            self._calib = self.parse_calib(str(self._calib_path))

        self._gt_poses = None
        if self._poses_path.exists():
            self._gt_poses = self.parse_poses(str(self._poses_path))

        # Cache intrinsics
        self._K = None

    def __len__(self) -> int:
        """Return the number of frames (capped by max_frames)."""
        n = len(self._images)
        if self.max_frames is not None:
            n = min(n, self.max_frames)
        return n

    def __iter__(self) -> Iterator[dict]:
        """Yield frames as dicts with image, timestamp, and ground truth pose.

        Yields:
            dict with keys:
                - 'image': BGR uint8 ndarray
                - 'timestamp': float (frame index as timestamp)
                - 'gt_pose': 4x4 ndarray or None
        """
        for i, img_path in enumerate(self._images):
            if self.max_frames is not None and i >= self.max_frames:
                break

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            gt_pose = None
            if self._gt_poses is not None and i < len(self._gt_poses):
                gt_pose = self._gt_poses[i]

            yield {
                "image": frame,
                "timestamp": float(i),
                "gt_pose": gt_pose,
            }

    def get_intrinsics(self) -> np.ndarray:
        """Return the 3x3 camera intrinsic matrix K.

        If ``use_calib_intrinsics=True`` and ``calib.txt`` exists, extracts
        K from the P0 projection matrix: ``K = P0[:3, :3]``.

        Otherwise, computes K from image dimensions using
        ``K_from_fov(width, height, fov_deg=63)``.

        Returns:
            3x3 intrinsic matrix as float64 ndarray.
        """
        if self._K is not None:
            return self._K.copy()

        # Try to use calibration file
        if self.use_calib_intrinsics and self._calib is not None:
            P0 = self._calib["P0"]
            self._K = P0[:3, :3].copy()
        else:
            # Fallback: compute from image dimensions
            img = cv2.imread(str(self._images[0]))
            h, w = img.shape[:2]
            self._K = K_from_fov(w, h, fov_deg=63.0)

        return self._K.copy()

    def get_ground_truth(self) -> list[np.ndarray] | None:
        """Return list of 4x4 ground truth poses, or None if unavailable.

        Poses are parsed from ``poses.txt`` (12 floats per line, row-major
        3x4 matrix). Each pose is converted to a 4x4 homogeneous matrix.

        Returns:
            List of 4x4 float64 ndarrays, or None if poses.txt is missing.
        """
        if self._gt_poses is None:
            return None
        return [pose.copy() for pose in self._gt_poses]

    @staticmethod
    def parse_calib(calib_path: str) -> dict:
        """Parse KITTI calibration file.

        Reads projection matrices and transformation matrices from the
        calibration file. Expected format:
        - ``P0: f1 f2 ... f12`` (3x4 flattened projection matrix)
        - ``P1: f1 f2 ... f12``
        - ``P2: f1 f2 ... f12``
        - ``P3: f1 f2 ... f12``
        - ``R0_rect: f1 ... f9`` (3x3 rectification matrix)
        - ``Tr_velo_to_cam: f1 ... f12`` (3x4 transformation)
        - ``Tr_imu_to_velo: f1 ... f12``

        Args:
            calib_path: Path to calib.txt file.

        Returns:
            Dict with keys 'P0', 'P1', 'P2', 'P3', 'R0_rect',
            'Tr_velo_to_cam', 'Tr_imu_to_velo' (whichever are present).
            Projection matrices are stored as 3x4 float64 ndarrays.
            R0_rect is stored as 3x3. Transformation matrices are 3x4.

        Example:
            >>> calib = KITTIFrameLoader.parse_calib('calib.txt')
            >>> P0 = calib['P0']  # 3x4 projection matrix
            >>> K = P0[:3, :3]    # Extract intrinsic matrix
        """
        result = {}
        with open(calib_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(":")
                if len(parts) != 2:
                    continue

                key = parts[0].strip()
                values = np.array([float(x) for x in parts[1].strip().split()])

                if key in ("P0", "P1", "P2", "P3"):
                    result[key] = values.reshape(3, 4)
                elif key == "R0_rect":
                    result["R0_rect"] = values.reshape(3, 3)
                elif key in ("Tr_velo_to_cam", "Tr_imu_to_velo"):
                    result[key] = values.reshape(3, 4)

        return result

    @staticmethod
    def parse_poses(poses_path: str) -> list[np.ndarray]:
        """Parse KITTI ground truth poses file.

        Each line contains 12 floats representing a 3x4 matrix in row-major
        order. These are converted to 4x4 homogeneous transformation matrices
        by appending a ``[0, 0, 0, 1]`` row.

        Args:
            poses_path: Path to poses.txt file.

        Returns:
            List of 4x4 float64 ndarrays representing camera poses.

        Example:
            >>> poses = KITTIFrameLoader.parse_poses('poses.txt')
            >>> pose0 = poses[0]  # 4x4 matrix
            >>> R = pose0[:3, :3]  # Rotation
            >>> t = pose0[:3, 3]   # Translation
        """
        poses = []
        with open(poses_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                values = np.array([float(x) for x in line.split()])
                if len(values) != 12:
                    continue

                # Reshape to 3x4 and convert to 4x4
                pose_3x4 = values.reshape(3, 4)
                pose_4x4 = np.vstack([pose_3x4, [0.0, 0.0, 0.0, 1.0]])
                poses.append(pose_4x4)

        return poses
