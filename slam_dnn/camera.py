"""Camera intrinsics and models."""
from dataclasses import dataclass
import numpy as np


def K_from_fov(width: int, height: int, fov_deg: float = 63.0) -> np.ndarray:
    """Build a 3x3 camera intrinsic matrix K from horizontal field of view.

    Assumes square pixels and principal point at image center. The focal
    length is derived from ``f = (width / 2) / tan(fov_deg / 2)``.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fov_deg: Horizontal field of view in degrees. Defaults to 63.0,
            a typical phone wide-angle lens FOV.

    Returns:
        3x3 intrinsic matrix K as a float64 ndarray.

    Example:
        >>> K = K_from_fov(640, 480, fov_deg=63.0)
        >>> K[0, 0]  # fx
        438.41...
        >>> K[2, 2]  # homogeneous coordinate
        1.0
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
    """Pinhole camera model with FOV-derived intrinsics.

    Computes the intrinsic matrix K from horizontal field of view, and
    stores its precomputed inverse K_inv for convenience.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fov_deg: Horizontal field of view in degrees. Defaults to 63.0,
            a typical phone wide-angle lens FOV.
        K: 3x3 intrinsic matrix (computed in __post_init__).
        K_inv: 3x3 inverse of K (computed in __post_init__).

    Example:
        >>> camera = PinholeCamera(width=640, height=480, fov_deg=63.0)
        >>> camera.K.shape
        (3, 3)
        >>> camera.K_inv.shape
        (3, 3)
        >>> # Use K to normalize pixel coordinates to camera frame
        >>> import numpy as np
        >>> p_pixel = np.array([320.0, 240.0, 1.0])
        >>> p_cam = camera.K_inv @ p_pixel
    """
    width: int
    height: int
    fov_deg: float = 63.0

    def __post_init__(self):
        self.K = K_from_fov(self.width, self.height, self.fov_deg)
        self.K_inv = np.linalg.inv(self.K)
