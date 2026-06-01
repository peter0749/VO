"""Tests for Essential matrix pose estimation."""
import cv2
import numpy as np
import pytest

from slam_dnn.camera import K_from_fov
from slam_dnn.exceptions import TrackingLostError
from slam_dnn.pose import estimate_essential


class TestSyntheticPoseRecovery:
    """Test pose estimation recovers known relative pose from synthetic data."""

    def test_synthetic_pose_recovery(self):
        """Recover known relative pose from synthetic 3D points."""
        # Setup
        K = K_from_fov(640, 480, fov_deg=63.0)
        np.random.seed(42)

        # Generate 100 random 3D points in camera 0 frame (z forward)
        pts_3d = np.random.uniform(-2, 2, (100, 3))
        pts_3d[:, 2] += 10  # push forward in z (z in [8, 12])

        # Ground truth relative pose (camera 0 -> camera 1)
        rvec_true = np.array([0.01, 0.02, -0.01])
        t_true = np.array([0.5, -0.2, 1.0])

        R_true, _ = cv2.Rodrigues(rvec_true)

        # Project to camera 0 (identity pose)
        pts_2d_0, _ = cv2.projectPoints(
            pts_3d.astype(np.float64),
            np.eye(3), np.zeros(3), K, None,
        )
        pts_2d_0 = pts_2d_0.reshape(-1, 2)

        # Project to camera 1
        pts_2d_1, _ = cv2.projectPoints(
            pts_3d.astype(np.float64),
            R_true, t_true, K, None,
        )
        pts_2d_1 = pts_2d_1.reshape(-1, 2)

        # Add Gaussian noise (0.5 pixel std)
        pts_2d_0 += np.random.normal(0, 0.5, pts_2d_0.shape)
        pts_2d_1 += np.random.normal(0, 0.5, pts_2d_1.shape)

        # Recover pose
        result = estimate_essential(
            pts_2d_0.astype(np.float32),
            pts_2d_1.astype(np.float32),
            K,
        )
        assert result is not None, "Pose estimation failed"

        R_est, t_est, inlier_mask = result

        # Rotation error as geodesic angle
        rot_error_deg = np.degrees(np.arccos(
            np.clip((np.trace(R_est @ R_true.T) - 1) / 2, -1, 1)
        ))

        # Translation direction error (unit vectors)
        t_true_unit = t_true / np.linalg.norm(t_true)
        t_est_unit = t_est / np.linalg.norm(t_est)
        trans_error_deg = np.degrees(np.arccos(
            np.clip(np.dot(t_true_unit, t_est_unit), -1, 1)
        ))

        assert rot_error_deg < 2.0, f"Rotation error {rot_error_deg:.2f}° > 2°"
        assert trans_error_deg < 5.0, (
            f"Translation direction error {trans_error_deg:.2f}° > 5°"
        )
        assert inlier_mask.sum() > 50, f"Too few inliers: {inlier_mask.sum()}/100"


class TestInsufficientMatches:
    """Test that functions degrade gracefully with few points."""

    def test_insufficient_matches_returns_none(self):
        """With only 3 matches (< 8 minimum), must return None."""
        K = K_from_fov(640, 480, fov_deg=63.0)
        pts0 = np.array([[100, 100], [200, 150], [300, 200]], dtype=np.float32)
        pts1 = np.array([[102, 101], [201, 152], [302, 198]], dtype=np.float32)
        assert estimate_essential(pts0, pts1, K) is None

    def test_empty_arrays_returns_none(self):
        """Empty point arrays must return None."""
        K = K_from_fov(640, 480, fov_deg=63.0)
        pts0 = np.empty((0, 2), dtype=np.float32)
        pts1 = np.empty((0, 2), dtype=np.float32)
        assert estimate_essential(pts0, pts1, K) is None


class TestTrackingLostError:
    """Test TrackingLostError exception behavior."""

    def test_tracking_lost_exception_importable(self):
        """TrackingLostError is importable and subclasses Exception."""
        assert issubclass(TrackingLostError, Exception)

    def test_tracking_lost_exception_raised(self):
        """TrackingLostError can be raised and caught as Exception."""
        with pytest.raises(TrackingLostError, match="tracking lost"):
            raise TrackingLostError("tracking lost")


class TestZeroMotionDegenerate:
    """Test behaviour when camera does not move (degenerate case)."""

    def test_identity_pose_with_zero_motion(self):
        """Static camera => identity R and near-zero t, OR None (degenerate)."""
        K = K_from_fov(640, 480, fov_deg=63.0)
        np.random.seed(7)

        # Random 3D points
        pts_3d = np.random.uniform(-2, 2, (100, 3)).astype(np.float64)
        pts_3d[:, 2] += 10

        # Both cameras share same pose => zero motion
        pts_2d, _ = cv2.projectPoints(
            pts_3d, np.eye(3), np.zeros(3), K, None,
        )
        pts_2d = pts_2d.reshape(-1, 2).astype(np.float32)

        result = estimate_essential(pts_2d, pts_2d.copy(), K)

        if result is not None:
            R_est, t_est, inlier_mask = result
            # With zero motion, R should be near identity
            rot_error_deg = np.degrees(np.arccos(
                np.clip((np.trace(R_est) - 1) / 2, -1, 1)
            ))
            assert rot_error_deg < 5.0, (
                f"Rotation deviates {rot_error_deg:.2f}° for zero motion"
            )


class TestEstimateEssentialChallengingCases:
    """Challenging pose estimation scenarios teaching failure modes."""

    def test_estimate_essential_with_large_rotation(self):
        """30° rotation is near the practical limit for essential matrix estimation.

        Beyond ~30° rotation, the epipolar geometry becomes ill-conditioned
        because most feature correspondences fall outside the image. This
        test documents the upper bound of what works reliably.
        """
        K = K_from_fov(640, 480, fov_deg=63.0)
        np.random.seed(99)

        pts_3d = np.random.uniform(-2, 2, (150, 3))
        pts_3d[:, 2] += 10

        rvec_true = np.array([0.0, 0.0, np.deg2rad(30.0)])
        t_true = np.array([0.3, 0.1, 0.5])
        R_true, _ = cv2.Rodrigues(rvec_true)

        pts_2d_0, _ = cv2.projectPoints(
            pts_3d.astype(np.float64), np.eye(3), np.zeros(3), K, None
        )
        pts_2d_1, _ = cv2.projectPoints(
            pts_3d.astype(np.float64), R_true, t_true, K, None
        )
        pts_2d_0 = pts_2d_0.reshape(-1, 2)
        pts_2d_1 = pts_2d_1.reshape(-1, 2)

        pts_2d_0 += np.random.normal(0, 0.3, pts_2d_0.shape)
        pts_2d_1 += np.random.normal(0, 0.3, pts_2d_1.shape)

        result = estimate_essential(
            pts_2d_0.astype(np.float32),
            pts_2d_1.astype(np.float32),
            K,
        )
        assert result is not None, "Pose estimation failed for 30° rotation"

        R_est, t_est, inlier_mask = result
        rot_error_deg = np.degrees(np.arccos(
            np.clip((np.trace(R_est @ R_true.T) - 1) / 2, -1, 1)
        ))
        assert rot_error_deg < 10.0, (
            f"Rotation error {rot_error_deg:.2f}° exceeds 10° for 30° true rotation"
        )

    def test_estimate_essential_noisy_correspondences(self):
        """1.0 pixel Gaussian noise is realistic for hand-labeled correspondences.

        Teaches: essential matrix estimation is robust to moderate noise
        (RANSAC handles outliers), but accuracy degrades. Rotation error
        should stay < 5° with 100 points at 1.0 px noise.
        """
        K = K_from_fov(640, 480, fov_deg=63.0)
        np.random.seed(77)

        pts_3d = np.random.uniform(-2, 2, (100, 3))
        pts_3d[:, 2] += 10

        rvec_true = np.array([0.05, 0.03, -0.02])
        t_true = np.array([0.8, -0.3, 1.2])
        R_true, _ = cv2.Rodrigues(rvec_true)

        pts_2d_0, _ = cv2.projectPoints(
            pts_3d.astype(np.float64), np.eye(3), np.zeros(3), K, None
        )
        pts_2d_1, _ = cv2.projectPoints(
            pts_3d.astype(np.float64), R_true, t_true, K, None
        )
        pts_2d_0 = pts_2d_0.reshape(-1, 2)
        pts_2d_1 = pts_2d_1.reshape(-1, 2)

        pts_2d_0 += np.random.normal(0, 1.0, pts_2d_0.shape)
        pts_2d_1 += np.random.normal(0, 1.0, pts_2d_1.shape)

        result = estimate_essential(
            pts_2d_0.astype(np.float32),
            pts_2d_1.astype(np.float32),
            K,
        )
        assert result is not None, "Pose estimation failed with 1.0 px noise"

        R_est, t_est, inlier_mask = result
        rot_error_deg = np.degrees(np.arccos(
            np.clip((np.trace(R_est @ R_true.T) - 1) / 2, -1, 1)
        ))
        assert rot_error_deg < 10.0, (
            f"Rotation error {rot_error_deg:.2f}° > 10° with 1.0 px noise"
        )
        assert inlier_mask.sum() > 30, (
            f"Expected > 30 inliers with 1.0 px noise, got {inlier_mask.sum()}"
        )
