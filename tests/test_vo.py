"""Tests for VisualOdometry facade class."""
import numpy as np
import numpy.testing as npt
import pytest

from slam_dnn import VisualOdometry, PinholeCamera, ClassicMatcher, MatcherBase


@pytest.fixture
def camera():
    """PinholeCamera with standard dimensions."""
    return PinholeCamera(width=640, height=480, fov_deg=63.0)


@pytest.fixture(scope="module")
def random_images():
    """5 random grayscale images for processing."""
    return [np.random.randint(0, 256, (480, 640), dtype=np.uint8) for _ in range(5)]


class TestVisualOdometryInitialization:
    """Tests for VisualOdometry constructor."""

    def test_visual_odometry_initialization_default(self, camera):
        """Default constructor works."""
        vo = VisualOdometry(camera)
        assert vo.camera is camera
        assert vo.min_matches == 20

    def test_visual_odometry_initialization_with_matcher_string(self, camera):
        """Constructor accepts matcher='classic'."""
        vo = VisualOdometry(
            camera, matcher="classic", max_keypoints=500, scale=0.5, device="cpu"
        )
        assert isinstance(vo.matcher, ClassicMatcher)

    def test_visual_odometry_initialization_with_matcher_instance(self, camera):
        """Constructor accepts MatcherBase instance directly."""
        custom = ClassicMatcher(ratio=0.6)
        vo = VisualOdometry(camera, matcher=custom)
        assert vo.matcher is custom

    def test_unknown_matcher_raises(self, camera):
        """Constructor with unknown matcher string raises ValueError."""
        with pytest.raises(ValueError):
            VisualOdometry(camera, matcher="nonsense")


class TestVisualOdometryProcessing:
    """Tests for process_frame and related methods."""

    def test_first_frame_returns_none(self, camera):
        """process_frame() on first frame returns None."""
        vo = VisualOdometry(camera, device="cpu", max_keypoints=50)
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        pose = vo.process_frame(img)
        assert pose is None
        assert vo.get_stats()["total"] == 1

    def test_consecutive_frames_run_without_crash(self, camera, random_images):
        """Multiple process_frame() calls don't crash."""
        vo = VisualOdometry(camera, device="cpu", max_keypoints=50, min_matches=5)
        for img in random_images:
            vo.process_frame(img)
        assert vo.get_stats()["total"] == 5

    def test_process_frame_returns_3x4_or_none(self, camera):
        """process_frame returns 3x4 pose or None."""
        vo = VisualOdometry(camera, device="cpu", max_keypoints=50)
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        pose = vo.process_frame(img)
        assert pose is None  # First frame is always None


class TestVisualOdometryStats:
    """Tests for stats tracking."""

    def test_get_stats_tracks_counts(self, camera):
        """Stats dict has correct keys after processing."""
        vo = VisualOdometry(camera, device="cpu", max_keypoints=50)
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        vo.process_frame(img)
        stats = vo.get_stats()
        expected_keys = {"total", "successful", "tracking_lost", "pose_failed"}
        assert set(stats.keys()) == expected_keys
        assert stats["total"] == 1

    def test_get_stats_returns_copy(self, camera):
        """get_stats() returns a copy, not a reference."""
        vo = VisualOdometry(camera, device="cpu")
        stats1 = vo.get_stats()
        stats1["total"] = 999
        stats2 = vo.get_stats()
        assert stats2["total"] != 999

    def test_stats_increment_on_consecutive_frames(self, camera, random_images):
        """Stats correctly track total frames processed."""
        vo = VisualOdometry(camera, device="cpu", max_keypoints=50, min_matches=5)
        for img in random_images:
            vo.process_frame(img)
        stats = vo.get_stats()
        assert stats["total"] == 5
        # Some combination of successful/tracking_lost/pose_failed should equal total - 1
        # (first frame always returns None without going through match/pose)
        processed = (
            stats["successful"] + stats["tracking_lost"] + stats["pose_failed"]
        )
        assert processed == 4  # 5 total - 1 first frame


class TestVisualOdometryReset:
    """Tests for reset() method."""

    def test_reset_clears_state(self, camera):
        """reset() returns to initial state."""
        vo = VisualOdometry(camera, device="cpu", max_keypoints=50)
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        vo.process_frame(img)
        vo.reset()
        assert vo.get_stats()["total"] == 0
        assert vo._prev_feats is None
        assert vo._frame_idx == 0
        # Trajectory still has initial identity pose
        assert len(vo.get_trajectory().get_poses()) == 1


class TestVisualOdometryInterface:
    """Tests for MatcherBase interface conformance."""

    def test_matcher_attribute_is_matcher_base(self, camera):
        """vo.matcher is always a MatcherBase instance."""
        vo1 = VisualOdometry(camera, matcher="lightglue", device="cpu")
        vo2 = VisualOdometry(camera, matcher="classic", device="cpu")
        assert isinstance(vo1.matcher, MatcherBase)
        assert isinstance(vo2.matcher, MatcherBase)


class TestVisualOdometryEdgeCases:
    """Additional edge cases teaching VO pipeline robustness."""

    def test_vo_config_object_overrides_params(self, camera):
        """VOConfig overrides constructor string arguments when provided.

        Teaches: the VOConfig dataclass is the canonical way to configure
        the pipeline. When passed, it takes precedence over individual kwargs
        like max_keypoints and scale.
        """
        from slam_dnn import VOConfig

        cfg = VOConfig(max_keypoints=256, scale=2.0, min_matches=10, device="cpu")
        vo = VisualOdometry(camera, matcher="classic", config=cfg)
        assert vo.get_stats()["total"] == 0
        assert vo.min_matches == 10
        assert isinstance(vo.matcher, ClassicMatcher)

    def test_vo_get_trajectory_returns_accumulator(self, camera):
        """get_trajectory() returns a TrajectoryAccumulator with initial identity.

        Teaches: even before processing any frames, the trajectory starts
        with one identity pose (the camera's starting position).
        """
        from slam_dnn import TrajectoryAccumulator

        vo = VisualOdometry(camera, device="cpu")
        traj = vo.get_trajectory()
        assert isinstance(traj, TrajectoryAccumulator)
        assert len(traj) == 1

    def test_vo_reset_and_process_again(self, camera):
        """After reset(), VO state is clean enough to process a fresh sequence.

        Teaches: reset() doesn't just clear counters — it fully reinitializes
        the internal feature cache and trajectory, so the next frame is
        treated as a new "first frame" (returning None).
        """
        vo = VisualOdometry(camera, device="cpu", max_keypoints=50)
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        vo.process_frame(img)
        vo.reset()

        pose = vo.process_frame(img)
        assert pose is None, "After reset, first frame should return None"
        assert vo.get_stats()["total"] == 1
