"""Trajectory accumulation (SE3)."""
import numpy as np


def pose_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from a 3x3 rotation R and a 3-vector translation t.

    The resulting matrix represents the homogeneous transformation:
        T = [[R, t],
             [0, 0, 0, 1]]

    Args:
        R: 3x3 rotation matrix.
        t: 3-element translation vector, or shape (3, 1) column vector.

    Returns:
        4x4 rigid-body transformation matrix as a float64 ndarray.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def compose_pose(T_prev: np.ndarray, T_rel: np.ndarray) -> np.ndarray:
    """Compose two 4x4 SE(3) transformations.

    The result describes the transformation from a fixed world frame to
    the current camera frame, given the previous world-to-camera matrix
    and the relative camera-to-camera motion.

    T_global_new = T_prev @ T_rel

    Args:
        T_prev: Previous global 4x4 pose (world to previous camera frame).
        T_rel: Relative 4x4 motion (previous frame to current frame).

    Returns:
        New global 4x4 pose (world to current camera frame).
    """
    return T_prev @ T_rel


def normalize_translation(t: np.ndarray) -> np.ndarray:
    """Normalize a translation vector to unit length.

    Useful for monocular VO where only the direction of translation is
    meaningful. The resulting vector preserves direction but has norm 1.

    Args:
        t: 3-element translation vector.

    Returns:
        Unit-norm float64 vector in the same direction.

    Raises:
        ValueError: If t is a zero vector (norm < 1e-10).
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    norm = np.linalg.norm(t)
    if norm < 1e-10:
        raise ValueError("Cannot normalize zero vector")
    return t / norm


def extract_translations(poses: list) -> np.ndarray:
    """Extract translation vectors from a list of 4x4 SE(3) poses.

    Translation vectors represent camera positions in the world frame
    (up to the scale ambiguity of monocular VO).

    Args:
        poses: List of 4x4 SE(3) matrices.

    Returns:
        (N, 3) float64 array where row i is the translation of pose i.
        Returns a (0, 3) array when poses is empty.
    """
    if len(poses) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array([p[:3, 3] for p in poses], dtype=np.float64)


class TrajectoryAccumulator:
    """Accumulates relative SE(3) poses into a global camera trajectory.

    Maintains a running camera pose (position and orientation) by composing
    relative motion estimates. The first pose is always the identity matrix
    (camera at the origin of the world frame).

    For each new relative pose (R_rel, t_rel), the internal state is updated:
        t_global = t_global + scale * (R_global @ t_rel_normalized)
        R_global = R_global @ R_rel

    where t_rel_normalized is the unit-direction of t_rel (since monocular VO
    cannot determine translation magnitude).

    Attributes:
        scale: Multiplicative factor applied to each translation step. Use this
            to compensate for the scale ambiguity of monocular VO. Default 1.0.

    Example:
        >>> import numpy as np
        >>> traj = TrajectoryAccumulator(scale=1.0)
        >>> # Simulate a camera moving forward with a slight rotation
        >>> R = np.eye(3)
        >>> t = np.array([0.0, 0.0, 1.0])  # forward one unit
        >>> traj.add_pose(R, t)
        >>> positions = traj.get_positions()
        >>> positions.shape
        (2, 3)
        >>> traj.save("trajectory.txt", format="kitti")
    """

    def __init__(self, scale: float = 1.0):
        """Initialize the trajectory with a single identity pose.

        Args:
            scale: Translation scaling factor applied to each incremental
                motion. Set above 1.0 to amplify, below 1.0 to attenuate.
                Default 1.0.
        """
        self.scale = scale
        self.poses: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
        self._current_R: np.ndarray = np.eye(3, dtype=np.float64)
        self._current_t: np.ndarray = np.zeros(3, dtype=np.float64)
        self._kf_R: np.ndarray = np.eye(3, dtype=np.float64)
        self._kf_t: np.ndarray = np.zeros(3, dtype=np.float64)

    def add_pose(self, R: np.ndarray, t: np.ndarray) -> None:
        """Add a relative pose (frame i-1 to frame i) to the trajectory.

        The relative motion is composed into the running global pose:
            t_global_new = t_global_old + scale * (R_global_old @ t_rel_unit)
            R_global_new = R_global_old @ R_rel

        where t_rel_unit is t_rel normalized to unit length. If the
        translation is near-zero (pure rotation), it is added as-is
        without normalization.

        Args:
            R: 3x3 rotation matrix from the previous to the current frame.
            t: (3,) or (3, 1) translation vector in the previous camera frame.
        """
        t = np.asarray(t, dtype=np.float64).flatten()
        R = np.asarray(R, dtype=np.float64)

        # Normalize translation (monocular VO only recovers direction)
        t_norm = np.linalg.norm(t)
        if t_norm > 1e-8:
            t_unit = t / t_norm
        else:
            t_unit = t  # degenerate: pure rotation or zero motion

        # Update global translation: transform relative t by current R, apply scale
        # t_global_new = t_global_old + scale * (R_global_old @ t_rel_normalized)
        self._current_t = self._current_t + self.scale * (self._current_R @ t_unit)

        # Update global rotation: compose
        # R_global_new = R_global_old @ R_rel
        self._current_R = self._current_R @ R

        # Periodically SVD re-orthogonalize to prevent numerical drift (Phase 2.4!)
        U, _, Vt = np.linalg.svd(self._current_R)
        self._current_R = U @ Vt

        # Build 4x4 SE3 and append to trajectory list
        T_current = pose_Rt(self._current_R, self._current_t)
        self.poses.append(T_current)

    def add_pose_relative_to_keyframe(self, R: np.ndarray, t: np.ndarray, is_keyframe: bool = True) -> None:
        """Add a relative pose where the pose is computed relative to the last keyframe.

        If is_keyframe is True, the keyframe base pose is updated to this new pose.
        If is_keyframe is False, the pose is appended, but subsequent frames will
        still calculate their pose relative to the previous keyframe.
        """
        t = np.asarray(t, dtype=np.float64).flatten()
        R = np.asarray(R, dtype=np.float64)

        t_norm = np.linalg.norm(t)
        if t_norm > 1e-8:
            t_unit = t / t_norm
        else:
            t_unit = t

        # Compute new global pose relative to the last keyframe base
        self._current_t = self._kf_t + self.scale * (self._kf_R @ t_unit)
        self._current_R = self._kf_R @ R

        # SVD re-orthogonalize to prevent numerical drift
        U, _, Vt = np.linalg.svd(self._current_R)
        self._current_R = U @ Vt

        if is_keyframe:
            # Update keyframe base to current global pose
            self._kf_R = self._current_R.copy()
            self._kf_t = self._current_t.copy()

        # Build 4x4 SE3 and append to trajectory list
        T_current = pose_Rt(self._current_R, self._current_t)
        self.poses.append(T_current)

    def get_poses(self) -> list[np.ndarray]:
        """Return the list of all accumulated global 4x4 SE(3) poses.

        Returns:
            List of 4x4 matrices representing the world-to-camera
            transformation at each frame, starting with the identity.
        """
        return self.poses

    def get_positions(self) -> np.ndarray:
        """Return an (N, 3) array of camera center positions in world frame.

        The camera center for pose T = [[R, t], [0, 1]] is:
            C_world = -R^T @ t

        This is the physical position of the camera origin in the world
        coordinate frame.

        Returns:
            Array where row i is the 3D camera center at frame i.
        """
        positions = []
        for T in self.poses:
            R = T[:3, :3]
            t = T[:3, 3]
            # Camera center in world frame: C_world = -R^T @ t
            c_world = -R.T @ t
            positions.append(c_world)
        return np.array(positions, dtype=np.float64)

    def reset(self) -> None:
        """Reset the trajectory to its initial state.

        Clears all accumulated poses and resets the internal rotation
        and translation to identity and zero. The trajectory length
        becomes 1 (a single identity pose).
        """
        self.poses = [np.eye(4, dtype=np.float64)]
        self._current_R = np.eye(3, dtype=np.float64)
        self._current_t = np.zeros(3, dtype=np.float64)
        self._kf_R = np.eye(3, dtype=np.float64)
        self._kf_t = np.zeros(3, dtype=np.float64)

    def __len__(self) -> int:
        """Return the number of accumulated poses, including the initial identity.

        This is always one more than the number of add_pose calls.
        """
        return len(self.poses)

    def save(self, filepath: str, format: str = "kitti", **kwargs) -> None:
        """Save the trajectory to a file in KITTI or TUM format.

        Args:
            filepath: Output file path.
            format: 'kitti' for KITTI format (12 floats/line) or 'tum' for
                TUM RGB-D format (timestamp + xyz + quaternion).
            **kwargs: Forwarded to the export function, such as 'timestamps'
                for TUM format.

        Raises:
            ValueError: If format is not 'kitti' or 'tum'.
        """
        from .export import export_kitti_format, export_tum_format

        if format == "kitti":
            export_kitti_format(self.get_poses(), filepath)
        elif format == "tum":
            export_tum_format(self.get_poses(), filepath, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format!r}. Use 'kitti' or 'tum'.")
