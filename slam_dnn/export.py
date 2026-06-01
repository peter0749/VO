"""Trajectory export/load in KITTI and TUM formats.

KITTI format: 12 floats per line (row-major 3x4 matrix)
TUM format: timestamp tx ty tz qx qy qz qw (8 values per line)
"""

import numpy as np


def export_kitti_format(poses: list[np.ndarray], filepath: str) -> None:
    """Save list of 4x4 SE3 matrices as KITTI format.

    Args:
        poses: List of 4x4 transformation matrices.
        filepath: Output file path.
    """
    with open(filepath, "w") as f:
        for T in poses:
            T_3x4 = T[:3, :]
            f.write(" ".join(f"{x:.6f}" for x in T_3x4.flatten()) + "\n")


def export_tum_format(
    poses: list[np.ndarray], filepath: str, timestamps: list[float] | None = None
) -> None:
    """Save list of 4x4 SE3 matrices as TUM RGB-D format.

    Args:
        poses: List of 4x4 transformation matrices.
        filepath: Output file path.
        timestamps: Optional list of timestamps. Defaults to frame indices (0, 1, 2, ...).
    """
    from scipy.spatial.transform import Rotation

    if timestamps is None:
        timestamps = list(range(len(poses)))

    with open(filepath, "w") as f:
        for t, T in zip(timestamps, poses):
            tx, ty, tz = T[:3, 3]
            quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
            f.write(
                f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )


def load_kitti_format(filepath: str) -> list[np.ndarray]:
    """Load KITTI trajectory from file (reverse of export_kitti_format).

    Args:
        filepath: Path to KITTI format file.

    Returns:
        List of 4x4 SE3 matrices.

    Raises:
        ValueError: If a line does not contain exactly 12 floats.
    """
    poses = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(x) for x in line.split()]
            assert len(values) == 12, f"Expected 12 floats, got {len(values)}"
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = np.array(values).reshape(3, 4)
            poses.append(T)
    return poses


def load_tum_format(filepath: str) -> tuple[list[np.ndarray], list[float]]:
    """Load TUM trajectory from file (reverse of export_tum_format).

    Args:
        filepath: Path to TUM format file.

    Returns:
        Tuple of (poses, timestamps) where poses is list of 4x4 SE3 matrices
        and timestamps is list of float timestamps.

    Raises:
        ValueError: If a line does not contain exactly 8 values.
    """
    from scipy.spatial.transform import Rotation

    poses, timestamps = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            assert len(values) == 8, f"Expected 8 values, got {len(values)}"
            t, x, y, z, qx, qy, qz, qw = values
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            poses.append(T)
            timestamps.append(t)
    return poses, timestamps
