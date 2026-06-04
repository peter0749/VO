"""Sliding window joint Bundle Adjustment and feature tracking."""
import numpy as np
import cv2
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
                coord = obs[prev_frame_idx]
                for idx, kp in enumerate(kps_prev):
                    if np.allclose(kp, coord, atol=1e-3):
                        prev_idx_to_track_id[idx] = track_id
                        break

        for idx_prev, idx_curr in matched_indices:
            kp_prev = kps_prev[idx_prev]
            kp_curr = kps_curr[idx_curr]

            if idx_prev in prev_idx_to_track_id:
                track_id = prev_idx_to_track_id[idx_prev]
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {prev_frame_idx: kp_prev}
                self.frame_to_tracks[prev_frame_idx].add(track_id)

            self.tracks[track_id][frame_idx] = kp_curr
            self.frame_to_tracks[frame_idx].add(track_id)


class LocalBundleAdjuster:
    """Performs joint non-linear optimization of camera poses and 3D map points."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def optimize(
        self,
        poses: list[np.ndarray],
        points_3d: np.ndarray,
        observations: list[dict],
        K: np.ndarray,
        fix_first_two: bool = True,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Refines poses and 3D structure to minimize reprojection errors.

        Args:
            poses: List of 4x4 world-to-camera transformations.
            points_3d: Array of shape (M, 3) representing 3D coordinates.
            observations: List of dicts, each with keys 'cam_idx', 'pt_idx', 'uv'.
            K: Camera intrinsics matrix (3x3).
            fix_first_two: If True, freezes the poses of the oldest two cameras
                           to lock scale and coordinate gauge. If False, only
                           freezes camera 0 and adds a soft baseline constraint on camera 1.

        Returns:
            Tuple (opt_poses, opt_points_3d) containing refined poses and points.
        """
        n_frames = len(poses)
        n_points = len(points_3d)

        if n_frames < 2 or n_points == 0 or len(observations) == 0:
            return poses, points_3d

        # Determine which cameras are optimized (variable) vs fixed
        # Fixed cameras: index 0 (always), index 1 (if fix_first_two is True)
        opt_start_idx = 2 if (fix_first_two and n_frames >= 2) else 1

        # Parameterize:
        # Poses to optimize: rvec (3,), tvec (3,) for camera indices opt_start_idx .. n_frames-1
        # Points to optimize: (n_points, 3)
        x0 = []
        for i in range(opt_start_idx, n_frames):
            R = poses[i][:3, :3]
            t = poses[i][:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            x0.extend(rvec.flatten())
            x0.extend(t.flatten())
        for pt in points_3d:
            x0.extend(pt)
        x0 = np.array(x0, dtype=np.float64)

        # Baseline length between camera 0 and camera 1 for scale anchoring
        init_scale = np.linalg.norm(poses[1][:3, 3] - poses[0][:3, 3])

        # Prepare fixed camera poses
        fixed_rvecs = []
        fixed_tvecs = []
        fixed_Rs = []
        fixed_ts = []
        for i in range(opt_start_idx):
            R = poses[i][:3, :3]
            t = poses[i][:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            fixed_rvecs.append(rvec.flatten())
            fixed_tvecs.append(t.flatten())
            fixed_Rs.append(R)
            fixed_ts.append(t.flatten())

        def residual_fn(params):
            rvecs = list(fixed_rvecs)
            tvecs = list(fixed_tvecs)
            Rs = list(fixed_Rs)
            ts = list(fixed_ts)

            # Extract variable camera poses
            num_opt_cams = n_frames - opt_start_idx
            for i in range(num_opt_cams):
                idx = 6 * i
                rv = params[idx:idx+3]
                tv = params[idx+3:idx+6]
                rvecs.append(rv)
                tvecs.append(tv)
                R, _ = cv2.Rodrigues(rv)
                Rs.append(R)
                ts.append(tv)

            # Extract variable 3D points
            pts_start_idx = 6 * num_opt_cams
            opt_pts = params[pts_start_idx:].reshape(n_points, 3)

            residuals = []
            
            # 1. Reprojection residuals & depth constraints
            for obs in observations:
                cam_idx = obs["cam_idx"]
                pt_idx = obs["pt_idx"]
                uv_meas = obs["uv"]
                pt_3d = opt_pts[pt_idx]

                # Project points to 2D
                pts_proj, _ = cv2.projectPoints(
                    pt_3d.reshape(1, 3).astype(np.float64),
                    rvecs[cam_idx].astype(np.float64),
                    tvecs[cam_idx].astype(np.float64),
                    K.astype(np.float64),
                    None
                )
                uv_proj = pts_proj.reshape(-1, 2)[0]
                residuals.append(uv_proj - uv_meas)

                # 2. Positive Depth Constraint: Z must be positive in camera frame
                # Always append to keep residual vector dimension CONSTANT across iterations.
                R_c = Rs[cam_idx]
                t_c = ts[cam_idx]
                z_c = R_c[2, 0]*pt_3d[0] + R_c[2, 1]*pt_3d[1] + R_c[2, 2]*pt_3d[2] + t_c[2]
                depth_err = np.minimum(z_c - 0.2, 0.0)
                residuals.append(np.array([100.0 * depth_err]))

            # 3. Soft Scale Constraint if we only fixed camera 0
            if opt_start_idx == 1:
                scale_err = np.linalg.norm(tvecs[1] - tvecs[0]) - init_scale
                residuals.append(np.array([1000.0 * scale_err]))

            return np.concatenate(residuals)

        # Run least squares with robust loss
        try:
            res = least_squares(
                residual_fn,
                x0,
                method='trf',
                loss='huber',
                f_scale=1.5,
                max_nfev=50  # Keep it real-time capable
            )
            opt_params = res.x
        except Exception:
            # If optimization fails, fallback to inputs
            return poses, points_3d

        # Reconstruct output poses and points
        opt_poses = []
        num_opt_cams = n_frames - opt_start_idx
        
        # Keep fixed poses
        for i in range(opt_start_idx):
            opt_poses.append(poses[i].copy())
            
        # Extract optimized poses
        for i in range(num_opt_cams):
            idx = 6 * i
            rvec = opt_params[idx:idx+3]
            t = opt_params[idx+3:idx+6]
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            opt_poses.append(T)

        # Extract optimized 3D points
        pts_start_idx = 6 * num_opt_cams
        opt_points_3d = opt_params[pts_start_idx:].reshape(n_points, 3)

        return opt_poses, opt_points_3d
