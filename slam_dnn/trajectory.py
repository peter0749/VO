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
