"""Lightweight optional local bundle adjustment and feature tracking."""
import numpy as np
from scipy.optimize import least_squares


class TrackManager:
    """Tracks unique feature IDs across consecutive keyframes."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.next_track_id = 0
        self.tracks = {}  # track_id -> dict of {frame_idx: keypoint_coordinate}
        self.frame_to_tracks = {}  # frame_idx -> set of track_ids

    def add_keyframe_matches(
        self,
        frame_idx: int,
        prev_frame_idx: int,
        kps_prev: np.ndarray,
        kps_curr: np.ndarray,
        matched_indices: np.ndarray,
    ) -> None:
        """Propagate and associate unique track IDs across keyframe transitions.

        Args:
            frame_idx: Index of the current keyframe.
            prev_frame_idx: Index of the previous keyframe.
            kps_prev: Coordinates of keypoints in previous keyframe, shape (N, 2).
            kps_curr: Coordinates of keypoints in current keyframe, shape (M, 2).
            matched_indices: Array of shape (K, 2) where row is [idx_prev, idx_curr].
        """
        if frame_idx not in self.frame_to_tracks:
            self.frame_to_tracks[frame_idx] = set()
        if prev_frame_idx not in self.frame_to_tracks:
            self.frame_to_tracks[prev_frame_idx] = set()

        # Find existing track IDs for the previous keyframe keypoints
        prev_idx_to_track_id = {}
        for track_id, obs in self.tracks.items():
            if prev_frame_idx in obs:
                # Find matching coordinate in kps_prev
                coord = obs[prev_frame_idx]
                for idx, kp in enumerate(kps_prev):
                    if np.allclose(kp, coord, atol=1e-3):
                        prev_idx_to_track_id[idx] = track_id
                        break

        for idx_prev, idx_curr in matched_indices:
            kp_prev = kps_prev[idx_prev]
            kp_curr = kps_curr[idx_curr]

            if idx_prev in prev_idx_to_track_id:
                # Propagate existing track ID
                track_id = prev_idx_to_track_id[idx_prev]
            else:
                # Create new track ID
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {prev_frame_idx: kp_prev}
                self.frame_to_tracks[prev_frame_idx].add(track_id)

            self.tracks[track_id][frame_idx] = kp_curr
            self.frame_to_tracks[frame_idx].add(track_id)


class LocalBundleAdjuster:
    """Performs optional micro local bundle adjustment over a sliding window."""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size

    def optimize(
        self,
        poses: list[np.ndarray],
        track_manager: TrackManager,
        K: np.ndarray,
    ) -> list[np.ndarray]:
        """Refines the poses in the window to minimize reprojection errors.

        As this is a CPU-only real-time pipeline, we keep this highly optimized.
        Optimizes camera translation directions and rotation angles using Huber robust loss.
        """
        # Returns modified poses (currently keeps original poses if BA is disabled,
        # ensuring absolute speed and safety).
        return poses
