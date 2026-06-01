"""High-level Visual Odometry orchestrator."""
import numpy as np
import logging
from typing import Iterator

from .camera import PinholeCamera
from .features import SuperPointExtractor
from .matching import MatcherBase, create_matcher
from .pose import estimate_essential
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
    ):
        """
        Args:
            camera: PinholeCamera with intrinsics.
            matcher: "lightglue", "classic", or MatcherBase instance.
            max_keypoints: SuperPoint max keypoints per frame.
            scale: Trajectory scale factor.
            device: "auto" | "cuda" | "mps" | "cpu".
            min_matches: Minimum matches required for pose estimation.
        """
        self.camera = camera
        self.min_matches = min_matches

        self.extractor = SuperPointExtractor(max_keypoints=max_keypoints, device=device)

        if isinstance(matcher, MatcherBase):
            self.matcher = matcher
        else:
            self.matcher = create_matcher(matcher, device=device)

        self.trajectory = TrajectoryAccumulator(scale=scale)
        self._prev_feats = None
        self._frame_idx = 0
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
            self._prev_feats = feats
            self._frame_idx += 1
            return None

        result = estimate_essential(
            match_result["points0"],
            match_result["points1"],
            self.camera.K,
        )

        if result is None:
            self._stats["pose_failed"] += 1
            logger.warning(f"Frame {self._frame_idx}: pose estimation failed")
            self._prev_feats = feats
            self._frame_idx += 1
            return None

        R, t, inlier_mask = result
        self.trajectory.add_pose(R, t)
        self._prev_feats = feats
        self._frame_idx += 1
        self._stats["successful"] += 1

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

    def reset(self) -> None:
        """Reset internal state (trajectory + frame counter)."""
        self.trajectory.reset()
        self._prev_feats = None
        self._frame_idx = 0
        self._stats = {
            "total": 0,
            "successful": 0,
            "tracking_lost": 0,
            "pose_failed": 0,
        }
