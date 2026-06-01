"""Trajectory accumulation (SE3)."""
import numpy as np


def pose_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 SE3 matrix from 3x3 rotation R and 3-vector translation t.

    Args:
        R: 3x3 rotation matrix
        t: 3-element translation vector (or 3x1 column)

    Returns:
        4x4 transformation matrix [R|t; 0,0,0,1]
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def compose_pose(T_prev: np.ndarray, T_rel: np.ndarray) -> np.ndarray:
    """Compose two SE3 transformations: T_global = T_prev @ T_rel.

    Args:
        T_prev: Previous global 4x4 pose
        T_rel: Relative 4x4 motion

    Returns:
        New global 4x4 pose
    """
    return T_prev @ T_rel


def normalize_translation(t: np.ndarray) -> np.ndarray:
    """Normalize translation vector to unit length.

    Args:
        t: 3-element translation vector

    Returns:
        Unit-norm vector in same direction

    Raises:
        ValueError: If t is zero vector
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    norm = np.linalg.norm(t)
    if norm < 1e-10:
        raise ValueError("Cannot normalize zero vector")
    return t / norm


def extract_translations(poses: list) -> np.ndarray:
    """Extract translation vectors from list of 4x4 poses.

    Args:
        poses: List of 4x4 SE3 matrices

    Returns:
        (N, 3) array of translation vectors (positions in world frame)
    """
    if len(poses) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array([p[:3, 3] for p in poses], dtype=np.float64)


class TrajectoryAccumulator:
    """Accumulates SE3 poses frame-by-frame into a trajectory.

    Maintains the running camera pose (position + orientation) as relative
    poses are added. The first pose is identity (camera at origin).

    Attributes:
        scale: Translation scaling factor (default 1.0). Used to compensate
            for scale ambiguity in monocular VO.
    """

    def __init__(self, scale: float = 1.0):
        """Initialize trajectory with identity pose.

        Args:
            scale: Translation scaling factor (default 1.0).
        """
        self.scale = scale
        self.poses: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
        self._current_R: np.ndarray = np.eye(3, dtype=np.float64)
        self._current_t: np.ndarray = np.zeros(3, dtype=np.float64)

    def add_pose(self, R: np.ndarray, t: np.ndarray) -> None:
        """Add relative pose (camera frame i-1 → frame i) to trajectory.

        Updates the running SE3 pose via composition:
            t_global = t_global + scale * (R_global @ t_rel_normalized)
            R_global = R_global @ R_rel

        Args:
            R: (3, 3) rotation matrix (frame i-1 → frame i)
            t: (3,) or (3, 1) translation vector (frame i-1 → frame i)
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

        # Build 4x4 SE3 and append to trajectory list
        T_current = pose_Rt(self._current_R, self._current_t)
        self.poses.append(T_current)

    def get_poses(self) -> list[np.ndarray]:
        """Return list of all accumulated SE3 poses (4x4 matrices).

        Returns:
            List of T_world_to_camera_i, starting with identity at i=0.
        """
        return self.poses

    def get_positions(self) -> np.ndarray:
        """Return (N, 3) array of camera center positions in world frame.

        Returns:
            Array where row i is the camera center at frame i.
            Camera center = -R^T @ t (world coords of camera origin).
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
        """Reset trajectory to initial state (single identity pose)."""
        self.poses = [np.eye(4, dtype=np.float64)]
        self._current_R = np.eye(3, dtype=np.float64)
        self._current_t = np.zeros(3, dtype=np.float64)

    def __len__(self) -> int:
        """Return number of accumulated poses (including initial identity)."""
        return len(self.poses)
