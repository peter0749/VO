"""Keyframe selection based on parallax and overlap heuristics."""
import numpy as np


class KeyframeSelector:
    """Selects whether to insert a new keyframe based on geometric sufficient change."""

    def __init__(
        self,
        min_parallax: float = 8.0,
        max_overlap: float = 0.85,
        max_interval: int = 10,
    ):
        """
        Args:
            min_parallax: Minimum median parallax in pixels to trigger a keyframe.
            max_overlap: Maximum feature overlap ratio to keep tracking the current frame
                         without creating a keyframe.
            max_interval: Maximum consecutive frames allowed before forcing a keyframe.
        """
        self.min_parallax = min_parallax
        self.max_overlap = max_overlap
        self.max_interval = max_interval
        self._frames_since_last_kf = 0

    def should_insert(
        self,
        points0: np.ndarray,
        points1: np.ndarray,
        num_total_features: int,
    ) -> bool:
        """Decide if a new keyframe should be inserted.

        Args:
            points0: Keypoints in the last keyframe, shape (K, 2).
            points1: Corresponding keypoints in the current frame, shape (K, 2).
            num_total_features: Total features in the current frame.

        Returns:
            True if the current frame should be selected as a new keyframe,
            False otherwise.
        """
        self._frames_since_last_kf += 1

        # Rule 1: Always insert if interval cap is reached
        if self._frames_since_last_kf >= self.max_interval:
            self._frames_since_last_kf = 0
            return True

        if len(points0) == 0 or len(points1) == 0:
            # Degenerate case, force keyframe
            self._frames_since_last_kf = 0
            return True

        # Rule 2: Parallax check (median displacement of matches)
        displacements = np.linalg.norm(points1 - points0, axis=1)
        median_parallax = np.median(displacements)
        if median_parallax > self.min_parallax:
            self._frames_since_last_kf = 0
            return True

        # Rule 3: Overlap check (fraction of matched features)
        overlap_ratio = len(points0) / max(num_total_features, 1)
        if overlap_ratio < self.max_overlap:
            self._frames_since_last_kf = 0
            return True

        return False

    def reset(self) -> None:
        """Reset internal frame counter."""
        self._frames_since_last_kf = 0
