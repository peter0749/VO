"""Essential matrix pose estimation."""
import cv2
import numpy as np


def estimate_essential(
    points0: np.ndarray,
    points1: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float = 1.0,
    conf: float = 0.999,
) -> tuple | None:
    """Estimate relative camera pose via the Essential matrix and 5-point RANSAC.

    Uses OpenCV's 5-point algorithm to find the Essential matrix E from
    normalized matched point pairs, then decomposes E with a chirality
    check to get the unique physical rotation and translation.

    The recovered translation has unit norm (only direction, not scale).

    Args:
        points0: Matched keypoints in the first frame, shape (N, 2), pixel coords.
        points1: Matched keypoints in the second frame, same shape.
        K: 3x3 camera intrinsic matrix.
        ransac_thresh: RANSAC inlier threshold in pixels. Default 1.0.
        conf: RANSAC confidence level. Default 0.999.

    Returns:
        Tuple (R, t, inlier_mask) where R is a 3x3 rotation matrix,
        t is a unit-norm (3,) translation vector, and inlier_mask is a (N,)
        boolean array. Returns None if too few matches or estimation fails.

    Note:
        Monocular VO can only recover translation direction, not magnitude.
        The returned t is unit-normalized and must be scaled externally
        (e.g., via TrajectoryAccumulator's scale factor).
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
    """Detect whether the camera motion between two frames is predominantly pure rotation.

    Pure rotation causes the Essential matrix to degenerate. This function
    compares Homography inlier count against Essential matrix inlier count.
    When the Homography explains substantially more matches than the
    Essential matrix (ratio above threshold), the motion is classified as
    pure rotation.

    Args:
        points0: Matched keypoints in the first frame, shape (N, 2), pixel coords.
        points1: Matched keypoints in the second frame, same shape.
        K: 3x3 camera intrinsic matrix.
        ransac_thresh: Pixel threshold for RANSAC inlier detection. Default 3.0.
        ratio_threshold: The Homography-to-Essential inlier count ratio above
            which pure rotation is declared. Default 0.6.

    Returns:
        True if the motion is predominantly rotation with negligible translation.
        False otherwise.
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
    """Estimate relative pose with a pure-rotation fallback.

    Tries standard Essential matrix estimation first. If that fails (e.g.,
    in a pure-rotation scenario where the Essential matrix degenerates),
    falls back to Homography decomposition to recover the rotation alone.

    Steps:
        1. Call ``estimate_essential``. If it succeeds, return its result.
        2. Call ``detect_pure_rotation``. If it returns False, return None.
        3. Estimate the Homography and decompose it as ``R = K^{-1} H K``.
           The translation is set to zero since there is no translational
           component in pure rotation.

    Args:
        points0: Matched keypoints in the first frame, shape (N, 2), pixel coords.
        points1: Matched keypoints in the second frame, same shape.
        K: 3x3 camera intrinsic matrix.
        ransac_thresh: RANSAC inlier threshold in pixels. Default 1.0.
        conf: RANSAC confidence level. Default 0.999.

    Returns:
        Tuple (R, t, inlier_mask) with t = zeros(3) in the pure-rotation case.
        Returns None if neither method produces a valid pose.
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
