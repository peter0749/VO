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
        expected_keys = {"total", "successful", "tracking_lost", "pose_failed", "keyframes", "motion_model_fallbacks"}
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


class TestKeyframeSelector:
    """Tests for KeyframeSelector class."""

    def test_keyframe_selector_initialization(self):
        """Constructor stores parameters correctly."""
        from slam_dnn.keyframe import KeyframeSelector
        selector = KeyframeSelector(min_parallax=5.0, max_overlap=0.9, max_interval=5)
        assert selector.min_parallax == 5.0
        assert selector.max_overlap == 0.9
        assert selector.max_interval == 5

    def test_keyframe_selector_interval_cap(self):
        """Always triggers keyframe when interval cap is reached."""
        from slam_dnn.keyframe import KeyframeSelector
        selector = KeyframeSelector(min_parallax=100.0, max_overlap=0.0, max_interval=2)
        pts = np.array([[10, 10]], dtype=np.float32)
        
        # First call increments counter but doesn't trigger since parallax/overlap are low
        assert not selector.should_insert(pts, pts, 10)
        # Second call reaches max_interval=2, forcing a keyframe
        assert selector.should_insert(pts, pts, 10)

    def test_keyframe_selector_parallax_trigger(self):
        """Triggers keyframe when median parallax is high."""
        from slam_dnn.keyframe import KeyframeSelector
        selector = KeyframeSelector(min_parallax=10.0, max_overlap=0.0, max_interval=10)
        pts0 = np.array([[10, 10]], dtype=np.float32)
        pts1 = np.array([[21, 10]], dtype=np.float32)  # displacement is 11 > 10
        assert selector.should_insert(pts0, pts1, 10)

    def test_keyframe_selector_overlap_trigger(self):
        """Triggers keyframe when overlap ratio is low."""
        from slam_dnn.keyframe import KeyframeSelector
        selector = KeyframeSelector(min_parallax=100.0, max_overlap=0.8, max_interval=10)
        pts = np.array([[10, 10], [20, 20]], dtype=np.float32)
        # 2 matches out of 10 total features = 0.2 overlap < 0.8 trigger threshold
        assert selector.should_insert(pts, pts, 10)


class TestMotionModel:
    """Tests for MotionModel class."""

    def test_motion_model_initialization(self):
        """Constructor stores parameters correctly."""
        from slam_dnn.motion_model import MotionModel
        model = MotionModel(ema_alpha=0.3)
        assert model.ema_alpha == 0.3
        assert not model._initialized

    def test_motion_model_update_and_predict(self):
        """MotionModel successfully updates velocity and predicts relative poses."""
        from slam_dnn.motion_model import MotionModel
        model = MotionModel(ema_alpha=1.0)  # Always use the newest pose
        
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        model.update(R, t)
        
        R_pred, t_pred = model.predict()
        assert model._initialized
        npt.assert_array_almost_equal(R_pred, R)
        npt.assert_array_almost_equal(t_pred, t)

    def test_motion_model_reset(self):
        """Reset clears state."""
        from slam_dnn.motion_model import MotionModel
        model = MotionModel()
        model.update(np.eye(3), np.array([1, 1, 1]))
        model.reset()
        assert not model._initialized
        npt.assert_array_almost_equal(model._t_vel, np.zeros(3))


class TestTrackManager:
    """Tests for TrackManager cross-frame tracking class."""

    def test_track_manager_propagation(self):
        """TrackManager correctly propagates unique feature track IDs."""
        from slam_dnn.local_ba import TrackManager
        manager = TrackManager()
        
        kps0 = np.array([[10, 10], [20, 20]], dtype=np.float32)
        kps1 = np.array([[15, 15], [25, 25]], dtype=np.float32)
        matches = np.array([[0, 0], [1, 1]], dtype=np.int32)
        
        manager.add_keyframe_matches(1, 0, kps0, kps1, matches)
        assert len(manager.tracks) == 2
        assert 0 in manager.frame_to_tracks[0]
        assert 1 in manager.frame_to_tracks[1]


class TestLocalBundleAdjuster:
    """Tests for LocalBundleAdjuster optimization class."""

    def test_local_ba_initialization(self):
        """Constructor stores parameters correctly."""
        from slam_dnn.local_ba import LocalBundleAdjuster
        ba = LocalBundleAdjuster(window_size=5)
        assert ba.window_size == 5


class TestVisualOdometryTimings:
    """Tests for pipeline timing instrumentation."""

    def test_get_timings_keys(self, camera):
        """get_timings returns dictionary with expected timing stage keys."""
        vo = VisualOdometry(camera, device="cpu")
        timings = vo.get_timings()
        expected = {"extraction", "matching", "pose_estimation", "total"}
        assert set(timings.keys()) == expected
        assert all(isinstance(v, float) for v in timings.values())



