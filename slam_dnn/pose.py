"""Essential matrix pose estimation."""
import cv2
import numpy as np

from .exceptions import TrackingLostError


def estimate_essential(
    points0: np.ndarray,
    points1: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float = 1.0,
    conf: float = 0.999,
) -> tuple | None:
    """Estimate relative pose (R, t) from matched keypoints via Essential matrix.

    Args:
        points0: (N, 2) float32, matched keypoints in image 0 (pixel coords)
        points1: (N, 2) float32, matched keypoints in image 1 (pixel coords)
        K: (3, 3) intrinsic matrix
        ransac_thresh: RANSAC inlier threshold in pixels (default 1.0)
        conf: RANSAC confidence (default 0.999)

    Returns:
        Tuple (R, t, inlier_mask) where:
            - R: (3, 3) rotation matrix (camera 0 -> camera 1)
            - t: (3,) translation vector (unit norm, up to scale)
            - inlier_mask: (N,) boolean array of inlier matches
        Returns None if insufficient matches or estimation fails.

    Note:
        Translation is only recovered up to scale. Use with unit baseline.
    """
    # Check minimum matches (need at least 8 for 5-point algorithm + RANSAC)
    if len(points0) < 8 or len(points1) < 8:
        return None

    # Normalize points to camera coordinates (divide by intrinsics)
    # Formula: p_norm = (p_pixel - [cx, cy]) / [fx, fy]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    points0_norm = (points0 - np.array([cx, cy])) / np.array([fx, fy])
    points1_norm = (points1 - np.array([cx, cy])) / np.array([fx, fy])

    # Estimate Essential matrix using 5-point RANSAC
    # cv2.findEssentialMat expects normalized coordinates when cameraMatrix = identity
    f_mean = (fx + fy) / 2.0  # average focal length for threshold scaling
    E, mask = cv2.findEssentialMat(
        points0_norm,
        points1_norm,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=conf,
        threshold=ransac_thresh / f_mean,
    )

    if E is None:
        return None

    # Extract first 3x3 block if E is stacked [3k x 3] (multi-pose case)
    if E.shape != (3, 3):
        E = E[:3, :]

    # Recover pose from Essential matrix (4-way chirality check)
    # Returns: (num_inliers, R, t, mask)
    # R: 3x3 rotation (camera 0 -> camera 1)
    # t: 3x1 translation (unit norm)
    try:
        num_inliers, R, t, mask = cv2.recoverPose(
            E, points0_norm, points1_norm, cameraMatrix=np.eye(3)
        )
    except cv2.error:
        return None

    if num_inliers == 0:
        return None

    # Verify R is a valid rotation (det = +1)
    if abs(np.linalg.det(R) - 1.0) > 0.01:
        return None

    # Convert t to (3,) array
    t = t.ravel()

    # Convert mask to boolean (N,)
    if mask is not None:
        inlier_mask = mask.ravel().astype(bool)
    else:
        inlier_mask = np.ones(len(points0), dtype=bool)

    return R, t, inlier_mask
