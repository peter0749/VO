"""High-level Visual Odometry orchestrator."""
import numpy as np
import logging
import time
from typing import Iterator

from .camera import PinholeCamera
from .config import VOConfig
from .features import SuperPointExtractor
from .matching import MatcherBase, create_matcher
from .pose import estimate_essential, estimate_essential_or_homography
from .trajectory import TrajectoryAccumulator
from .keyframe import KeyframeSelector
from .motion_model import MotionModel


logger = logging.getLogger(__name__)


class VisualOdometry:
    """Facade that orchestrates the full VO pipeline.

    Example:
        >>> from slam_dnn import VisualOdometry, PinholeCamera
        >>> camera = PinholeCamera(width=640, height=480, fov_deg=63)
        >>> vo = VisualOdometry(camera, matcher='lightglue', device='cpu')
        >>> for frame in video_frames:
        ...     pose = vo.process_frame(frame)
    """

    def __init__(
        self,
        camera: PinholeCamera,
        matcher: str | MatcherBase = "lightglue",
        max_keypoints: int = 1024,
        scale: float = 1.0,
        device: str = "auto",
        min_matches: int = 20,
        config: VOConfig | None = None,
    ):
        """
        Args:
            camera: PinholeCamera with intrinsics.
            matcher: "lightglue", "classic", or MatcherBase instance.
            max_keypoints: SuperPoint max keypoints per frame.
            scale: Trajectory scale factor.
            device: "auto" | "cuda" | "mps" | "cpu".
            min_matches: Minimum matches required for pose estimation.
            config: Optional VOConfig; overrides other kwargs when provided.
        """
        if config is None:
            config = VOConfig()
            # Preserve overrides if custom values were explicitly passed as kwargs
            if max_keypoints != 1024:
                config.max_keypoints = max_keypoints
            if scale != 1.0:
                config.scale = scale
            if device != "auto":
                config.device = device
            if min_matches != 20:
                config.min_matches = min_matches
            if isinstance(matcher, str):
                if matcher == "lightglue":
                    matcher = config.matcher
                else:
                    config.matcher = matcher
        else:
            # Respect overrides when config is provided but explicitly overridden in kwargs
            if max_keypoints != 1024:
                config.max_keypoints = max_keypoints
            if scale != 1.0:
                config.scale = scale
            if device != "auto":
                config.device = device
            if min_matches != 20:
                config.min_matches = min_matches
            if isinstance(matcher, str) and matcher == "lightglue":
                matcher = config.matcher

        self.camera = camera
        self.min_matches = config.min_matches
        self._handle_pure_rotation = config.handle_pure_rotation
        self._ransac_threshold = config.ransac_threshold
        self._ransac_confidence = config.ransac_confidence

        self.extractor = SuperPointExtractor(
            max_keypoints=config.max_keypoints,
            conf_thresh=config.detection_threshold,
            device=config.device,
            target_resolution=config.target_resolution,
        )

        if isinstance(matcher, MatcherBase):
            self.matcher = matcher
        else:
            if matcher == "lightglue":
                self.matcher = create_matcher(
                    matcher,
                    filter_threshold=config.lightglue_threshold,
                    device=config.device,
                )
            elif matcher == "classic":
                self.matcher = create_matcher(
                    matcher,
                    ratio=config.classic_ratio,
                    device=config.device,
                )
            else:
                self.matcher = create_matcher(matcher, device=config.device)

        self.trajectory = TrajectoryAccumulator(scale=scale)
        self._prev_feats = None
        self._frame_idx = 0
        
        # Keyframe & Motion Model Heuristics
        self._use_keyframe_selection = config.use_keyframe_selection
        self.keyframe_selector = KeyframeSelector(
            min_parallax=config.min_parallax,
            max_overlap=config.max_overlap,
            max_interval=config.max_keyframe_interval,
        )
        
        self._use_motion_model = config.use_motion_model
        self.motion_model = MotionModel(ema_alpha=config.motion_model_alpha)

        self._per_frame_stats: list[dict] = []
        self._stats = {
            "total": 0,
            "successful": 0,
            "tracking_lost": 0,
            "pose_failed": 0,
            "keyframes": 0,
            "motion_model_fallbacks": 0,
        }
        self._timings = {
            "extraction": 0.0,
            "matching": 0.0,
            "pose_estimation": 0.0,
            "total": 0.0,
        }

    def process_frame(self, image: np.ndarray) -> np.ndarray | None:
        """Process one frame, return relative pose (3x4) if successful, else None.

        The first frame always returns None (no previous frame to compare against).
        """
        t_start = time.perf_counter()
        self._stats["total"] += 1
        
        t_extract_start = time.perf_counter()
        feats = self.extractor.extract(image)
        self._timings["extraction"] += time.perf_counter() - t_extract_start

        if self._prev_feats is None:
            self._prev_feats = feats
            self._frame_idx += 1
            self._stats["keyframes"] += 1
            self._per_frame_stats.append({
                "frame_idx": self._frame_idx - 1,
                "num_matches": 0,
                "num_inliers": 0,
                "tracking_lost": False,
                "pose_failed": False,
                "is_keyframe": True,
                "motion_model_fallback": False,
            })
            self._timings["total"] += time.perf_counter() - t_start
            return None

        t_match_start = time.perf_counter()
        match_result = self.matcher.match(
            self._prev_feats, feats, image_size=image.shape[:2]
        )
        self._timings["matching"] += time.perf_counter() - t_match_start
        n_matches = len(match_result["points0"])

        # 1. Tracking lost handling
        if n_matches < self.min_matches:
            self._stats["tracking_lost"] += 1
            logger.warning(
                f"Frame {self._frame_idx}: tracking lost "
                f"({n_matches} < {self.min_matches})"
            )
            
            if self._use_motion_model:
                R, t = self.motion_model.predict()
                self.trajectory.add_pose_relative_to_keyframe(R, t, is_keyframe=False)
                self._stats["motion_model_fallbacks"] += 1
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": True,
                    "pose_failed": False,
                    "is_keyframe": False,
                    "motion_model_fallback": True,
                })
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                pose_3x4 = np.hstack([R, t.reshape(3, 1)])
                return pose_3x4
            else:
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": True,
                    "pose_failed": False,
                    "is_keyframe": False,
                    "motion_model_fallback": False,
                })
                self._prev_feats = feats
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                return None

        # 2. Estimate relative pose
        t_pose_start = time.perf_counter()
        pose_fn = estimate_essential_or_homography if self._handle_pure_rotation else estimate_essential
        result = pose_fn(
            match_result["points0"],
            match_result["points1"],
            self.camera.K,
            ransac_thresh=self._ransac_threshold,
            conf=self._ransac_confidence,
        )
        self._timings["pose_estimation"] += time.perf_counter() - t_pose_start

        # 3. Pose estimation failed handling
        if result is None:
            self._stats["pose_failed"] += 1
            logger.warning(f"Frame {self._frame_idx}: pose estimation failed")
            
            if self._use_motion_model:
                R, t = self.motion_model.predict()
                self.trajectory.add_pose_relative_to_keyframe(R, t, is_keyframe=False)
                self._stats["motion_model_fallbacks"] += 1
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": False,
                    "pose_failed": True,
                    "is_keyframe": False,
                    "motion_model_fallback": True,
                })
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                pose_3x4 = np.hstack([R, t.reshape(3, 1)])
                return pose_3x4
            else:
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": False,
                    "pose_failed": True,
                    "is_keyframe": False,
                    "motion_model_fallback": False,
                })
                self._prev_feats = feats
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                return None

        # 4. Successful pose estimation: decide if we insert a keyframe
        R, t, inlier_mask = result
        
        is_kf = True
        if self._use_keyframe_selection:
            is_kf = self.keyframe_selector.should_insert(
                match_result["points0"],
                match_result["points1"],
                len(feats["keypoints"]),
            )

        self.trajectory.add_pose_relative_to_keyframe(R, t, is_keyframe=is_kf)
        self.motion_model.update(R, t)
        
        if is_kf:
            self._prev_feats = feats
            self._stats["keyframes"] += 1

        self._frame_idx += 1
        self._stats["successful"] += 1
        self._per_frame_stats.append({
            "frame_idx": self._frame_idx - 1,
            "num_matches": n_matches,
            "num_inliers": int(inlier_mask.sum()),
            "tracking_lost": False,
            "pose_failed": False,
            "is_keyframe": is_kf,
            "motion_model_fallback": False,
        })

        self._timings["total"] += time.perf_counter() - t_start
        pose_3x4 = np.hstack([R, t.reshape(3, 1)])
        return pose_3x4

    def process_sequence(self, images: Iterator[np.ndarray]) -> list[np.ndarray]:
        """Process a sequence of images, return list of 4x4 poses."""
        poses = []
        for img in images:
            pose = self.process_frame(img)
            if pose is not None:
                T = np.eye(4)
                T[:3, :] = pose
                poses.append(T)
        return poses

    def get_trajectory(self) -> TrajectoryAccumulator:
        """Return the trajectory accumulator."""
        return self.trajectory

    def get_stats(self) -> dict:
        """Return statistics dict."""
        return self._stats.copy()

    def get_timings(self) -> dict:
        """Return accumulated pipeline timings (in seconds) per stage."""
        return self._timings.copy()

    def get_per_frame_stats(self) -> list[dict]:
        """Return copy of per-frame statistics list."""
        return [s.copy() for s in self._per_frame_stats]

    def reset(self) -> None:
        """Reset internal state (trajectory + frame counter)."""
        self.trajectory.reset()
        self._prev_feats = None
        self._frame_idx = 0
        self.keyframe_selector.reset()
        self.motion_model.reset()
        self._per_frame_stats = []
        self._stats = {
            "total": 0,
            "successful": 0,
            "tracking_lost": 0,
            "pose_failed": 0,
            "keyframes": 0,
            "motion_model_fallbacks": 0,
        }
        self._timings = {
            "extraction": 0.0,
            "matching": 0.0,
            "pose_estimation": 0.0,
            "total": 0.0,
        }
