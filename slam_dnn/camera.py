"""Camera intrinsics and models."""
from dataclasses import dataclass
import numpy as np


def K_from_fov(width: int, height: int, fov_deg: float = 63.0) -> np.ndarray:
    """Build 3x3 intrinsic matrix K from horizontal FOV.

    Assumes square pixels and principal point at image center.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_deg: Horizontal FOV in degrees (default: 63.0, phone wide-angle)

    Returns:
        3x3 intrinsic matrix K
    """
    fx = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    fy = fx  # square pixels
    cx, cy = width / 2.0, height / 2.0
    return np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@dataclass
class PinholeCamera:
    """Pinhole camera model with FOV-derived intrinsics."""
    width: int
    height: int
    fov_deg: float = 63.0

    def __post_init__(self):
        self.K = K_from_fov(self.width, self.height, self.fov_deg)
        self.K_inv = np.linalg.inv(self.K)
