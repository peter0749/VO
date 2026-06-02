"""Tests for KITTIFrameLoader — KITTI odometry dataset loader.

Covers:
- Flat and nested directory format auto-detection
- Calibration parsing (calib.txt → P0 projection matrix → K)
- Ground truth parsing (poses.txt → list of 4x4 matrices)
- Graceful fallback: missing calib.txt → FOV-based K, missing poses.txt → None
- max_frames limiting
- use_calib_intrinsics flag
- Static parse_calib / parse_poses methods
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2
from pathlib import Path

from slam_dnn.kitti_loader import KITTIFrameLoader
from slam_dnn.camera import K_from_fov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_image(path: Path, w: int = 320, h: int = 240) -> None:
    """Write a synthetic BGR uint8 image to disk."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_calib_txt(path: Path, fx: float = 500.0, fy: float = 500.0,
                    cx: float = 160.0, cy: float = 120.0) -> None:
    """Write a synthetic KITTI calib.txt with known intrinsics."""
    # P0 is a 3x4 projection matrix: [K | 0]
    p0_values = [fx, 0.0, cx, 0.0,
                 0.0, fy, cy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
    p0_str = " ".join(f"{v:.6f}" for v in p0_values)

    # Duplicate for P1, P2, P3 (same values for simplicity)
    with open(path, "w") as f:
        f.write(f"P0: {p0_str}\n")
        f.write(f"P1: {p0_str}\n")
        f.write(f"P2: {p0_str}\n")
        f.write(f"P3: {p0_str}\n")
        f.write("R0_rect: 1 0 0 0 1 0 0 0 1\n")
        f.write("Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        f.write("Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n")


def _make_poses_txt(path: Path, n_poses: int) -> list[np.ndarray]:
    """Write a synthetic KITTI poses.txt and return expected poses.

    Each pose is identity with a unique translation to verify correctness.
    Returns list of expected 4x4 matrices.
    """
    expected = []
    with open(path, "w") as f:
        for i in range(n_poses):
            # Identity rotation, translation = [i, i*0.5, i*0.1]
            values = [1.0, 0.0, 0.0, float(i),
                      0.0, 1.0, 0.0, float(i) * 0.5,
                      0.0, 0.0, 1.0, float(i) * 0.1]
            line = " ".join(f"{v:.6f}" for v in values)
            f.write(line + "\n")

            pose_4x4 = np.eye(4, dtype=np.float64)
            pose_4x4[0, 3] = float(i)
            pose_4x4[1, 3] = float(i) * 0.5
            pose_4x4[2, 3] = float(i) * 0.1
            expected.append(pose_4x4)

    return expected


def _create_flat_kitti_dir(base: Path, n_frames: int = 5,
                           with_calib: bool = True,
                           with_poses: bool = True) -> list[np.ndarray]:
    """Create a flat KITTI directory layout. Returns expected poses list."""
    image_dir = base / "image_0"
    image_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_frames):
        _create_synthetic_image(image_dir / f"{i:06d}.png")

    if with_calib:
        _make_calib_txt(base / "calib.txt")

    expected_poses = []
    if with_poses:
        expected_poses = _make_poses_txt(base / "poses.txt", n_frames)

    return expected_poses


def _create_nested_kitti_dir(base: Path, sequence: str = "05",
                              n_frames: int = 5,
                              with_calib: bool = True,
                              with_poses: bool = True) -> list[np.ndarray]:
    """Create a nested KITTI directory layout. Returns expected poses list."""
    seq_dir = base / "sequences" / sequence
    image_dir = seq_dir / "image_0"
    image_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_frames):
        _create_synthetic_image(image_dir / f"{i:06d}.png")

    if with_calib:
        _make_calib_txt(seq_dir / "calib.txt")

    expected_poses = []
    if with_poses:
        expected_poses = _make_poses_txt(seq_dir / "poses.txt", n_frames)

    return expected_poses


# ---------------------------------------------------------------------------
# Test: Flat format auto-detection
# ---------------------------------------------------------------------------

class TestFlatFormat:
    """Tests for flat directory format: base_dir/image_0/."""

    def test_loads_images_from_flat_dir(self, tmp_path):
        """Loader finds images in flat format and iterates correctly."""
        _create_flat_kitti_dir(tmp_path, n_frames=5)

        loader = KITTIFrameLoader(str(tmp_path))

        assert len(loader) == 5
        frames = list(loader)
        assert len(frames) == 5

        for frame in frames:
            assert "image" in frame
            assert "timestamp" in frame
            assert "gt_pose" in frame
            assert frame["image"].dtype == np.uint8
            assert frame["image"].ndim == 3

    def test_intrinsics_from_calib_flat(self, tmp_path):
        """get_intrinsics() returns 3x3 K from P0 in calib.txt."""
        _create_flat_kitti_dir(tmp_path, n_frames=3, with_calib=True)

        loader = KITTIFrameLoader(str(tmp_path))
        K = loader.get_intrinsics()

        assert K.shape == (3, 3)
        # Verify known values from _make_calib_txt
        assert abs(K[0, 0] - 500.0) < 1e-6  # fx
        assert abs(K[1, 1] - 500.0) < 1e-6  # fy
        assert abs(K[0, 2] - 160.0) < 1e-6  # cx
        assert abs(K[1, 2] - 120.0) < 1e-6  # cy
        assert abs(K[2, 2] - 1.0) < 1e-6

    def test_ground_truth_flat(self, tmp_path):
        """get_ground_truth() returns list of 4x4 poses."""
        expected_poses = _create_flat_kitti_dir(tmp_path, n_frames=5)

        loader = KITTIFrameLoader(str(tmp_path))
        gt = loader.get_ground_truth()

        assert gt is not None
        assert len(gt) == 5
        for i, pose in enumerate(gt):
            assert pose.shape == (4, 4)
            np.testing.assert_allclose(pose, expected_poses[i], atol=1e-5)

    def test_gt_pose_in_iter(self, tmp_path):
        """Each frame dict includes correct gt_pose."""
        expected = _create_flat_kitti_dir(tmp_path, n_frames=3)

        loader = KITTIFrameLoader(str(tmp_path))
        for i, frame in enumerate(loader):
            np.testing.assert_allclose(frame["gt_pose"], expected[i], atol=1e-5)


# ---------------------------------------------------------------------------
# Test: Nested format auto-detection
# ---------------------------------------------------------------------------

class TestNestedFormat:
    """Tests for nested directory format: base_dir/sequences/XX/image_0/."""

    def test_loads_images_from_nested_dir(self, tmp_path):
        """Loader auto-detects nested format and loads images."""
        _create_nested_kitti_dir(tmp_path, sequence="05", n_frames=4)

        loader = KITTIFrameLoader(str(tmp_path), sequence="05")

        assert len(loader) == 4
        frames = list(loader)
        assert len(frames) == 4

    def test_intrinsics_from_nested(self, tmp_path):
        """get_intrinsics() works with nested format."""
        _create_nested_kitti_dir(tmp_path, sequence="05", n_frames=3)

        loader = KITTIFrameLoader(str(tmp_path), sequence="05")
        K = loader.get_intrinsics()

        assert K.shape == (3, 3)
        assert abs(K[0, 0] - 500.0) < 1e-6

    def test_ground_truth_nested(self, tmp_path):
        """get_ground_truth() works with nested format."""
        expected = _create_nested_kitti_dir(tmp_path, sequence="07", n_frames=3)

        loader = KITTIFrameLoader(str(tmp_path), sequence="07")
        gt = loader.get_ground_truth()

        assert gt is not None
        assert len(gt) == 3


# ---------------------------------------------------------------------------
# Test: Missing files → graceful fallback
# ---------------------------------------------------------------------------

class TestGracefulFallback:
    """Tests for missing calibration and pose files."""

    def test_missing_poses_returns_none(self, tmp_path):
        """Missing poses.txt → get_ground_truth() returns None."""
        _create_flat_kitti_dir(tmp_path, n_frames=3, with_poses=False)

        loader = KITTIFrameLoader(str(tmp_path))
        assert loader.get_ground_truth() is None

    def test_missing_poses_gt_pose_in_iter_is_none(self, tmp_path):
        """Without poses.txt, each frame['gt_pose'] is None."""
        _create_flat_kitti_dir(tmp_path, n_frames=3, with_poses=False)

        loader = KITTIFrameLoader(str(tmp_path))
        for frame in loader:
            assert frame["gt_pose"] is None

    def test_missing_calib_falls_back_to_fov(self, tmp_path):
        """Missing calib.txt → K computed from K_from_fov(img_w, img_h, 63)."""
        _create_flat_kitti_dir(tmp_path, n_frames=2, with_calib=False)

        loader = KITTIFrameLoader(str(tmp_path))
        K = loader.get_intrinsics()

        # Verify it matches K_from_fov for 320x240 images
        expected_K = K_from_fov(320, 240, fov_deg=63.0)
        assert K.shape == (3, 3)
        np.testing.assert_allclose(K, expected_K, atol=1e-6)

    def test_use_calib_false_uses_fov(self, tmp_path):
        """use_calib_intrinsics=False forces FOV fallback even if calib exists."""
        _create_flat_kitti_dir(tmp_path, n_frames=2, with_calib=True)

        loader = KITTIFrameLoader(str(tmp_path), use_calib_intrinsics=False)
        K = loader.get_intrinsics()

        expected_K = K_from_fov(320, 240, fov_deg=63.0)
        np.testing.assert_allclose(K, expected_K, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: max_frames limiting
# ---------------------------------------------------------------------------

class TestMaxFrames:
    """Tests for max_frames parameter."""

    def test_max_frames_limits_len(self, tmp_path):
        """max_frames caps the return value of __len__."""
        _create_flat_kitti_dir(tmp_path, n_frames=10)

        loader = KITTIFrameLoader(str(tmp_path), max_frames=5)
        assert len(loader) == 5

    def test_max_frames_limits_iteration(self, tmp_path):
        """max_frames limits the number of yielded frames."""
        _create_flat_kitti_dir(tmp_path, n_frames=10)

        loader = KITTIFrameLoader(str(tmp_path), max_frames=5)
        frames = list(loader)
        assert len(frames) == 5

    def test_max_frames_exceeds_available(self, tmp_path):
        """max_frames larger than available frames returns all."""
        _create_flat_kitti_dir(tmp_path, n_frames=3)

        loader = KITTIFrameLoader(str(tmp_path), max_frames=100)
        assert len(loader) == 3
        frames = list(loader)
        assert len(frames) == 3

    def test_max_frames_limits_gt_poses_too(self, tmp_path):
        """max_frames should also limit ground truth poses in iteration."""
        _create_flat_kitti_dir(tmp_path, n_frames=10)

        loader = KITTIFrameLoader(str(tmp_path), max_frames=3)
        frames = list(loader)
        assert len(frames) == 3
        # Each should still have a valid gt_pose
        for frame in frames:
            assert frame["gt_pose"] is not None


# ---------------------------------------------------------------------------
# Test: parse_calib static method
# ---------------------------------------------------------------------------

class TestParseCalib:
    """Tests for parse_calib static method."""

    def test_parse_calib_returns_dict(self, tmp_path):
        """parse_calib returns dict with expected keys."""
        calib_path = tmp_path / "calib.txt"
        _make_calib_txt(calib_path)

        calib = KITTIFrameLoader.parse_calib(str(calib_path))

        assert "P0" in calib
        assert "P1" in calib
        assert "P2" in calib
        assert "P3" in calib
        assert "R0_rect" in calib
        assert "Tr_velo_to_cam" in calib
        assert "Tr_imu_to_velo" in calib

    def test_parse_calib_matrix_shapes(self, tmp_path):
        """Parsed matrices have correct shapes."""
        calib_path = tmp_path / "calib.txt"
        _make_calib_txt(calib_path)

        calib = KITTIFrameLoader.parse_calib(str(calib_path))

        assert calib["P0"].shape == (3, 4)
        assert calib["P1"].shape == (3, 4)
        assert calib["P2"].shape == (3, 4)
        assert calib["P3"].shape == (3, 4)
        assert calib["R0_rect"].shape == (3, 3)
        assert calib["Tr_velo_to_cam"].shape == (3, 4)


# ---------------------------------------------------------------------------
# Test: parse_poses static method
# ---------------------------------------------------------------------------

class TestParsePoses:
    """Tests for parse_poses static method."""

    def test_parse_poses_returns_list_of_4x4(self, tmp_path):
        """parse_poses returns list of 4x4 matrices."""
        poses_path = tmp_path / "poses.txt"
        _make_poses_txt(poses_path, n_poses=5)

        poses = KITTIFrameLoader.parse_poses(str(poses_path))

        assert len(poses) == 5
        for pose in poses:
            assert pose.shape == (4, 4)

    def test_parse_poses_last_row(self, tmp_path):
        """Last row of each pose is [0, 0, 0, 1]."""
        poses_path = tmp_path / "poses.txt"
        _make_poses_txt(poses_path, n_poses=3)

        poses = KITTIFrameLoader.parse_poses(str(poses_path))

        for pose in poses:
            np.testing.assert_array_equal(pose[3], [0.0, 0.0, 0.0, 1.0])

    def test_parse_poses_translation(self, tmp_path):
        """Translation values are parsed correctly."""
        poses_path = tmp_path / "poses.txt"
        _make_poses_txt(poses_path, n_poses=5)

        poses = KITTIFrameLoader.parse_poses(str(poses_path))

        for i, pose in enumerate(poses):
            assert abs(pose[0, 3] - float(i)) < 1e-5
            assert abs(pose[1, 3] - float(i) * 0.5) < 1e-5
            assert abs(pose[2, 3] - float(i) * 0.1) < 1e-5


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for error cases."""

    def test_missing_base_dir_raises(self, tmp_path):
        """Non-existent base_dir raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            KITTIFrameLoader(str(tmp_path / "nonexistent"))

    def test_no_image_dir_raises(self, tmp_path):
        """Base dir exists but no image_0/ or sequences/ raises FileNotFoundError."""
        # Create base dir with no subdirectories
        (tmp_path / "random_file.txt").write_text("hello\n")

        with pytest.raises(FileNotFoundError):
            KITTIFrameLoader(str(tmp_path))

    def test_empty_image_dir_raises(self, tmp_path):
        """image_0/ exists but has no images → ValueError."""
        image_dir = tmp_path / "image_0"
        image_dir.mkdir()

        with pytest.raises(ValueError, match="No image files"):
            KITTIFrameLoader(str(tmp_path))


# ---------------------------------------------------------------------------
# Test: Timestamp values
# ---------------------------------------------------------------------------

class TestTimestamps:
    """Tests for timestamp field in yielded frames."""

    def test_timestamps_are_frame_indices(self, tmp_path):
        """Timestamps are float frame indices (0.0, 1.0, 2.0, ...)."""
        _create_flat_kitti_dir(tmp_path, n_frames=5)

        loader = KITTIFrameLoader(str(tmp_path))
        for i, frame in enumerate(loader):
            assert frame["timestamp"] == float(i)
