"""High-level Visual Odometry orchestrator."""
import numpy as np
import logging
from typing import Iterator

from .camera import PinholeCamera
from .config import VOConfig
from .features import SuperPointExtractor
from .matching import MatcherBase, create_matcher
from .pose import estimate_essential, estimate_essential_or_homography
from .trajectory import TrajectoryAccumulator


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
        if config is not None:
            max_keypoints = config.max_keypoints
            scale = config.scale
            device = config.device
            min_matches = config.min_matches
            if isinstance(matcher, str) and matcher == "lightglue":
                matcher = config.matcher

        self.camera = camera
        self.min_matches = min_matches
        self._handle_pure_rotation = config.handle_pure_rotation if config else True

        self.extractor = SuperPointExtractor(max_keypoints=max_keypoints, device=device)

        if isinstance(matcher, MatcherBase):
            self.matcher = matcher
        else:
            self.matcher = create_matcher(matcher, device=device)

        self.trajectory = TrajectoryAccumulator(scale=scale)
        self._prev_feats = None
        self._frame_idx = 0
        self._per_frame_stats: list[dict] = []
        self._stats = {
            "total": 0,
            "successful": 0,
            "tracking_lost": 0,
            "pose_failed": 0,
        }

    def process_frame(self, image: np.ndarray) -> np.ndarray | None:
        """Process one frame, return relative pose (3x4) if successful, else None.

        The first frame always returns None (no previous frame to compare against).
        """
        self._stats["total"] += 1
        feats = self.extractor.extract(image)

        if self._prev_feats is None:
            self._prev_feats = feats
            self._frame_idx += 1
            self._per_frame_stats.append({
                "frame_idx": self._frame_idx - 1,
                "num_matches": 0,
                "num_inliers": 0,
                "tracking_lost": False,
                "pose_failed": False,
            })
            return None

        match_result = self.matcher.match(
            self._prev_feats, feats, image_size=image.shape[:2]
        )
        n_matches = len(match_result["points0"])

        if n_matches < self.min_matches:
            self._stats["tracking_lost"] += 1
            logger.warning(
                f"Frame {self._frame_idx}: tracking lost "
                f"({n_matches} < {self.min_matches})"
            )
            self._per_frame_stats.append({
                "frame_idx": self._frame_idx,
                "num_matches": n_matches,
                "num_inliers": 0,
                "tracking_lost": True,
                "pose_failed": False,
            })
            self._prev_feats = feats
            self._frame_idx += 1
            return None

        pose_fn = estimate_essential_or_homography if self._handle_pure_rotation else estimate_essential
        result = pose_fn(
            match_result["points0"],
            match_result["points1"],
            self.camera.K,
        )

        if result is None:
            self._stats["pose_failed"] += 1
            logger.warning(f"Frame {self._frame_idx}: pose estimation failed")
            self._per_frame_stats.append({
                "frame_idx": self._frame_idx,
                "num_matches": n_matches,
                "num_inliers": 0,
                "tracking_lost": False,
                "pose_failed": True,
            })
            self._prev_feats = feats
            self._frame_idx += 1
            return None

        R, t, inlier_mask = result
        self.trajectory.add_pose(R, t)
        self._prev_feats = feats
        self._frame_idx += 1
        self._stats["successful"] += 1
        self._per_frame_stats.append({
            "frame_idx": self._frame_idx - 1,
            "num_matches": n_matches,
            "num_inliers": int(inlier_mask.sum()),
            "tracking_lost": False,
            "pose_failed": False,
        })

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

    def get_per_frame_stats(self) -> list[dict]:
        """Return copy of per-frame statistics list."""
        return [s.copy() for s in self._per_frame_stats]

    def reset(self) -> None:
        """Reset internal state (trajectory + frame counter)."""
        self.trajectory.reset()
        self._prev_feats = None
        self._frame_idx = 0
        self._per_frame_stats = []
        self._stats = {
            "total": 0,
            "successful": 0,
            "tracking_lost": 0,
            "pose_failed": 0,
        }
