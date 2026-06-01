"""Tests for camera.py and trajectory.py functions."""
import numpy as np
import numpy.testing as npt
import pytest

from slam_dnn.camera import K_from_fov, PinholeCamera
from slam_dnn.trajectory import (
    pose_Rt, compose_pose, normalize_translation, extract_translations
)


class TestKFromFov:
    """Tests for K_from_fov function."""

    def test_k_from_fov_90_degrees(self):
        """K_from_fov(640, 480, 90) gives fx=320 (320/tan(45°)=320)."""
        K = K_from_fov(640, 480, 90)
        npt.assert_allclose(K[0, 0], 320.0)
        npt.assert_allclose(K[1, 1], 320.0)
        npt.assert_allclose(K[0, 2], 320.0)
        npt.assert_allclose(K[1, 2], 240.0)
        npt.assert_allclose(K[2, 2], 1.0)
        # Off-diagonal elements in top-left 2x2 should be zero
        npt.assert_allclose(K[:2, :2] - np.diag(np.diag(K[:2, :2])), 0.0)

    def test_k_from_fov_63_degrees_default(self):
        """Default FOV=63° gives known focal length."""
        K = K_from_fov(640, 480)
        expected_fx = (640 / 2.0) / np.tan(np.deg2rad(63.0) / 2.0)
        npt.assert_allclose(K[0, 0], expected_fx)
        npt.assert_allclose(K[1, 1], expected_fx)  # square pixels

    def test_k_from_fov_square_pixels(self):
        """fx must equal fy for square pixel assumption."""
        K = K_from_fov(1280, 720, 75)
        npt.assert_allclose(K[0, 0], K[1, 1])

    def test_k_from_fov_principal_point_center(self):
        """Principal point must be at image center."""
        K = K_from_fov(800, 600, 70)
        npt.assert_allclose(K[0, 2], 400.0)
        npt.assert_allclose(K[1, 2], 300.0)


class TestPinholeCameraDataclass:
    """Tests for PinholeCamera dataclass."""

    def test_pinhole_camera_auto_computes_K(self):
        """PinholeCamera should auto-compute K and K_inv."""
        cam = PinholeCamera(width=640, height=480, fov_deg=63)
        assert cam.K.shape == (3, 3)
        assert cam.K_inv.shape == (3, 3)

    def test_pinhole_camera_K_inv_is_inverse(self):
        """K @ K_inv ≈ I."""
        cam = PinholeCamera(width=640, height=480, fov_deg=63)
        npt.assert_allclose(cam.K @ cam.K_inv, np.eye(3), atol=1e-10)


class TestPoseRt:
    """Tests for pose_Rt function."""

    def test_pose_Rt_identity(self):
        """pose_Rt(I, 0) is identity 4x4."""
        R = np.eye(3)
        t = np.zeros(3)
        T = pose_Rt(R, t)
        npt.assert_allclose(T, np.eye(4))

    def test_pose_Rt_translation_only(self):
        """pose_Rt(I, t) has correct translation column."""
        R = np.eye(3)
        t = np.array([1, 2, 3])
        T = pose_Rt(R, t)
        npt.assert_allclose(T[:3, 3], t)
        npt.assert_allclose(T[:3, :3], np.eye(3))

    def test_pose_Rt_handles_column_vector(self):
        """pose_Rt should accept (3,1) column vector."""
        R = np.eye(3)
        t = np.array([[1], [2], [3]])
        T = pose_Rt(R, t)
        npt.assert_allclose(T[:3, 3], [1, 2, 3])


class TestComposePose:
    """Tests for SE3 composition."""

    def test_compose_identity_left(self):
        """compose_pose(I, T) == T."""
        T = np.array([
            [0, -1, 0, 1],
            [1, 0, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        result = compose_pose(np.eye(4), T)
        npt.assert_allclose(result, T)

    def test_compose_identity_right(self):
        """compose_pose(T, I) == T."""
        T = np.array([
            [0, -1, 0, 1],
            [1, 0, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        result = compose_pose(T, np.eye(4))
        npt.assert_allclose(result, T)

    def test_compose_with_inverse(self):
        """compose_pose(T, T^-1) ≈ I."""
        R = np.array([
            [0.866, -0.5, 0],
            [0.5, 0.866, 0],
            [0, 0, 1],
        ])
        t = np.array([1, 2, 3])
        T = pose_Rt(R, t)
        T_inv = np.linalg.inv(T)
        result = compose_pose(T, T_inv)
        npt.assert_allclose(result, np.eye(4), atol=1e-10)


class TestNormalizeTranslation:
    """Tests for normalize_translation function."""

    def test_unit_already(self):
        """Unit vectors unchanged."""
        t = np.array([1, 0, 0])
        npt.assert_allclose(normalize_translation(t), t)

    def test_scale_to_unit(self):
        """Non-unit vectors scaled correctly."""
        t = np.array([3, 4, 0])
        result = normalize_translation(t)
        npt.assert_allclose(result, [0.6, 0.8, 0.0])
        npt.assert_allclose(np.linalg.norm(result), 1.0)

    def test_zero_raises(self):
        """Zero vector raises ValueError."""
        with pytest.raises(ValueError):
            normalize_translation(np.zeros(3))

    def test_preserves_direction(self):
        """Normalization preserves direction, only scales."""
        t = np.array([1, 1, 0])
        result = normalize_translation(t)
        expected_dir = t / np.linalg.norm(t)
        npt.assert_allclose(result, expected_dir)


class TestExtractTranslations:
    """Tests for extract_translations function."""

    def test_empty_list(self):
        """Empty list returns (0,3) array."""
        result = extract_translations([])
        assert result.shape == (0, 3)

    def test_three_poses(self):
        """Extracts translation from 3 poses."""
        poses = [
            pose_Rt(np.eye(3), [1, 0, 0]),
            pose_Rt(np.eye(3), [2, 0, 0]),
            pose_Rt(np.eye(3), [3, 0, 0]),
        ]
        result = extract_translations(poses)
        assert result.shape == (3, 3)
        npt.assert_allclose(result, [[1, 0, 0], [2, 0, 0], [3, 0, 0]])
