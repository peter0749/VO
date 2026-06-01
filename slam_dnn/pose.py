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


def detect_pure_rotation(
    points0: np.ndarray,
    points1: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float = 3.0,
    ratio_threshold: float = 0.6,
) -> bool:
    """Detect if camera motion is predominantly pure rotation.

    Uses Homography vs Essential matrix inlier comparison:
    - Estimate both H and E via RANSAC
    - If H inliers >> E inliers (ratio > threshold), return True
    - If both succeed with similar inlier counts, return False
    - If E has more inliers, definitely not pure rotation, return False

    Note: In pure rotation, the fundamental matrix F degenerates —
    Homography H correctly describes the entire motion.

    Args:
        points0, points1: Matched pixel coordinate pairs
        K: Camera intrinsic matrix
        ransac_thresh: Pixel threshold for RANSAC inlier detection
        ratio_threshold: H-inlier / E-inlier ratio above which pure rotation is declared

    Returns:
        True if motion is predominantly pure rotation
    """
    if len(points0) < 8 or len(points1) < 8:
        return False

    H, mask_h = cv2.findHomography(
        points0, points1, cv2.RANSAC, ransac_thresh
    )
    if H is None or mask_h is None:
        return False
    n_h = int(mask_h.sum())

    result = estimate_essential(
        points0, points1, K,
        ransac_thresh=ransac_thresh,
    )
    if result is None:
        return True
    _, _, mask_e = result
    n_e = int(mask_e.sum())

    if n_e == 0:
        return n_h > 0

    ratio = n_h / n_e
    return ratio > ratio_threshold


def estimate_essential_or_homography(
    points0: np.ndarray,
    points1: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float = 1.0,
    conf: float = 0.999,
) -> tuple | None:
    """Estimate pose with pure-rotation fallback.

    Algorithm:
        1. Try normal estimate_essential()
        2. If None (E failed), try detect_pure_rotation()
        3. If pure rotation: decompose H to get R-only, set t=0
        4. Return (R, t=zeros, inlier_mask) or None

    Returns same format as estimate_essential(): (R, t, inlier_mask) or None
    """
    result = estimate_essential(
        points0, points1, K,
        ransac_thresh=ransac_thresh,
        conf=conf,
    )
    if result is not None:
        return result

    if not detect_pure_rotation(points0, points1, K, ransac_thresh=ransac_thresh):
        return None

    H, mask = cv2.findHomography(
        points0, points1, cv2.RANSAC, ransac_thresh
    )
    if H is None:
        return None

    # H = K @ R @ K^-1 for pure rotation (no translation)
    # R = K^-1 @ H @ K
    K_inv = np.linalg.inv(K)
    R_unnormalized = K_inv @ H @ K

    # Orthogonalize R (polar decomposition via SVD)
    U, _, Vt = np.linalg.svd(R_unnormalized)
    R = U @ Vt

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        R = -R

    t = np.zeros(3)
    inlier_mask = mask.ravel().astype(bool)
    return R, t, inlier_mask
