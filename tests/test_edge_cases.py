"""Tests for edge case handling (pure rotation, tracking lost, config)."""
import numpy as np
import numpy.testing as npt
import pytest

from slam_dnn.pose import detect_pure_rotation, estimate_essential, estimate_essential_or_homography
from slam_dnn.config import VOConfig
from slam_dnn import PinholeCamera, VisualOdometry
from slam_dnn.exceptions import TrackingLostError


def make_pure_rotation_scene(n_points=100, fov_deg=63.0):
    """Returns (pts0, pts1, K) where camera only rotated."""
    K = np.array([
        [600, 0, 320],
        [0, 600, 240],
        [0, 0, 1],
    ], dtype=float)

    rng = np.random.default_rng(42)
    pts_3d = rng.normal(size=(n_points, 3))
    pts_3d = pts_3d / np.linalg.norm(pts_3d, axis=1, keepdims=True) * 10

    theta = np.radians(30)
    R2 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])

    pts1_h = (K @ pts_3d.T).T
    pts1 = pts1_h[:, :2] / pts1_h[:, 2:3]

    pts2_h = (K @ (R2 @ pts_3d.T)).T
    pts2 = pts2_h[:, :2] / pts2_h[:, 2:3]

    return pts1.astype(np.float32), pts2.astype(np.float32), K


def make_translation_scene(n_points=100):
    """Returns (pts0, pts1, K) where camera translated with strong parallax."""
    K = np.array([
        [600, 0, 320],
        [0, 600, 240],
        [0, 0, 1],
    ], dtype=float)

    rng = np.random.default_rng(123)
    pts_3d = rng.normal(size=(n_points, 3)) * 2.0
    pts_3d[:, 2] += 5.0

    R1 = np.eye(3)
    t1 = np.zeros(3)

    R2 = np.eye(3)
    t2 = np.array([2.0, 1.0, 2.0])

    pts1_3d = (R1 @ pts_3d.T + t1.reshape(3, 1)).T
    pts1_h = (K @ pts1_3d.T).T
    pts1 = pts1_h[:, :2] / pts1_h[:, 2:3]

    pts2_3d = (R2 @ pts_3d.T + t2.reshape(3, 1)).T
    pts2_h = (K @ pts2_3d.T).T
    pts2 = pts2_h[:, :2] / pts2_h[:, 2:3]

    return pts1.astype(np.float32), pts2.astype(np.float32), K


def test_detect_pure_rotation_true():
    """Synthesize pure rotation (zero translation)."""
    pts1, pts2, K = make_pure_rotation_scene(n_points=100)
    result = detect_pure_rotation(pts1, pts2, K)
    assert result is True, "Should detect pure rotation"


def test_detect_pure_rotation_false_with_translation():
    """Synthesize forward motion."""
    pts1, pts2, K = make_translation_scene(n_points=100)
    result = detect_pure_rotation(pts1, pts2, K)
    assert result is False, "Should NOT detect pure rotation with translation"


def test_detect_pure_rotation_with_noise():
    """Pure rotation + small noise on points."""
    pts1, pts2, K = make_pure_rotation_scene(n_points=100)
    rng = np.random.default_rng(999)
    noise = rng.normal(0, 0.5, pts2.shape)
    pts2_noisy = pts2 + noise.astype(np.float32)

    result = detect_pure_rotation(pts1, pts2_noisy, K, ransac_thresh=3.0)
    assert result is True, "Should still detect pure rotation with small noise"


def test_detect_pure_rotation_insufficient_points():
    """Only 3 points — should return False gracefully."""
    pts1, pts2, K = make_pure_rotation_scene(n_points=3)
    result = detect_pure_rotation(pts1, pts2, K)
    assert result is False, "Should return False for insufficient points"


def test_estimate_essential_or_homography_normal_case():
    """Regular motion with translation."""
    pts1, pts2, K = make_translation_scene(n_points=100)
    result = estimate_essential_or_homography(pts1, pts2, K)

    assert result is not None, "Should succeed for normal motion"
    R, t, mask = result

    assert R.shape == (3, 3), "R should be 3x3"
    assert t.shape == (3,), "t should be (3,)"
    assert mask.shape[0] == len(pts1), "mask should match input length"

    t_norm = np.linalg.norm(t)
    assert t_norm > 0.1, f"Translation should be non-zero for normal motion, got {t_norm}"


def test_estimate_essential_or_homography_pure_rotation_fallback():
    """When E fails, use homography fallback."""
    pts1, pts2, K = make_pure_rotation_scene(n_points=100)

    result = estimate_essential_or_homography(pts1, pts2, K, ransac_thresh=1.0)

    assert result is not None, "Should succeed with homography fallback"
    R, t, mask = result

    assert R.shape == (3, 3), "R should be 3x3"
    assert t.shape == (3,), "t should be (3,)"

    t_norm = np.linalg.norm(t)
    assert t_norm < 0.01, f"Translation should be ~0 for pure rotation, got {t_norm}"

    R_det = np.linalg.det(R)
    assert abs(R_det - 1.0) < 0.01, f"R should be valid rotation (det=+1), got det={R_det}"


def test_vo_config_defaults():
    """Verify all default values match plan."""
    config = VOConfig()

    assert config.max_keypoints == 2048
    assert config.detection_threshold == 0.0005
    assert config.matcher == 'lightglue'
    assert config.lightglue_threshold == 0.1
    assert config.classic_ratio == 0.75
    assert config.ransac_threshold == 1.0
    assert config.ransac_confidence == 0.999
    assert config.min_matches == 20
    assert config.scale == 1.0
    assert config.fov_deg == 63.0
    assert config.handle_pure_rotation is True
    assert config.device == 'auto'


def test_vo_config_can_override():
    """Create with custom values."""
    config = VOConfig(max_keypoints=500, matcher='classic', scale=2.0)

    assert config.max_keypoints == 500
    assert config.matcher == 'classic'
    assert config.scale == 2.0

    assert config.detection_threshold == 0.0005
    assert config.fov_deg == 63.0


def test_vo_per_frame_stats_tracked():
    """Process a few frames and check stats are tracked."""
    camera = PinholeCamera(width=640, height=480, fov_deg=63)
    vo = VisualOdometry(camera, matcher='classic', max_keypoints=512, device='cpu')

    rng = np.random.default_rng(42)
    for _ in range(3):
        img = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        vo.process_frame(img)

    stats = vo.get_per_frame_stats()
    assert len(stats) == 3, f"Should have 3 frame stats, got {len(stats)}"

    for stat in stats:
        assert "frame_idx" in stat
        assert "num_matches" in stat
        assert "num_inliers" in stat
        assert "tracking_lost" in stat
        assert "pose_failed" in stat


def test_vo_per_frame_stats_reset_clears():
    """reset() clears per-frame stats."""
    camera = PinholeCamera(width=640, height=480, fov_deg=63)
    vo = VisualOdometry(camera, matcher='classic', max_keypoints=512, device='cpu')

    rng = np.random.default_rng(42)
    for _ in range(2):
        img = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        vo.process_frame(img)

    assert len(vo.get_per_frame_stats()) > 0, "Should have stats before reset"

    vo.reset()
    assert len(vo.get_per_frame_stats()) == 0, "Stats should be cleared after reset"
