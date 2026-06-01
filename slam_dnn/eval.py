"""Trajectory evaluation: Umeyama alignment, APE, RTE metrics.

Provides pure-NumPy implementations of:
  - Sim(3) / SE(3) trajectory alignment (Umeyama 1991, IEEE PAMI)
  - Absolute Pose Error (APE)
  - Relative Trajectory Error (RTE) with distance-based sliding window
  - A unified evaluate() entry-point returning a full metrics report.

No third-party trajectory evaluation libraries (e.g. evo) are required.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_camera_center(T: np.ndarray) -> np.ndarray:
    """Camera center in world frame from a world-to-camera extrinsic matrix.

    For T = [[R, t], [0, 1]] (world → camera), the camera centre C in world
    coordinates satisfies R C + t = 0 ⟹ C = −Rᵀ t.

    Parameters
    ----------
    T : array of shape (4, 4)
        World-to-camera SE(3) matrix.

    Returns
    -------
    C : array of shape (3,)
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return -R.T @ t


# ---------------------------------------------------------------------------
# Umeyama alignment
# ---------------------------------------------------------------------------

def align_umeyama(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Align *estimated* to *ground_truth* via Sim(3) (or SE(3) when ``with_scale=False``).

    Implements the closed-form least-squares alignment of Umeyama (1991).

    Parameters
    ----------
    estimated : array of shape (N, 3)
        Source point cloud (e.g. estimated camera centres).
    ground_truth : array of shape (N, 3)
        Target point cloud.
    with_scale : bool, default True
        If *True*, recover a full Sim(3) (scale + rotation + translation).
        If *False*, force ``scale = 1`` and only recover SE(3).

    Returns
    -------
    scale : float
        Uniform scale factor (1.0 when ``with_scale=False``).
    R : array of shape (3, 3)
        Rotation matrix.
    t : array of shape (3,)
        Translation vector.

    The aligned points are recovered as::

        aligned = scale * (R @ estimated.T) + t[:, None]   # (3, N)

    Raises
    ------
    ValueError
        If shapes do not match, points are not 3-D, or *N* < 3.
    """
    if estimated.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: estimated {estimated.shape} vs "
            f"ground_truth {ground_truth.shape}"
        )
    n, d = estimated.shape
    if d != 3:
        raise ValueError(f"Expected 3-D points, got {d}-D")
    if n < 3:
        raise ValueError(
            f"At least 3 point correspondences required, got {n}"
        )

    # 1. Centroids
    mu_e = np.mean(estimated, axis=0)
    mu_g = np.mean(ground_truth, axis=0)

    # 2. Centred coordinates
    e_c = estimated - mu_e
    g_c = ground_truth - mu_g

    # 3. Cross-covariance matrix
    C = g_c.T @ e_c  # (3, 3)

    # 4. SVD
    U, S, Vt = np.linalg.svd(C)

    # 5. Reflection fix: ensure det(R) = +1
    det_sign = np.linalg.det(U @ Vt)
    if det_sign < 0.0:
        # Flip last column of U (and corresponding singular value)
        U[:, -1] *= -1.0
        S[-1] *= -1.0

    R = U @ Vt

    # 6. Scale
    if with_scale:
        var_e = np.sum(e_c ** 2)  # total scatter of source points
        # S may contain a negative entry after reflection fix; use sum directly
        # (Umeyama: scale = trace(D S) / var_e, already encoded in S after fix)
        scale = float(np.sum(S) / var_e) if var_e > 0.0 else 1.0
    else:
        scale = 1.0

    # 7. Translation
    t = mu_g - scale * R @ mu_e

    return scale, R, t


def align_trajectories(
    est_poses: list[np.ndarray],
    gt_poses: list[np.ndarray],
    with_scale: bool = True,
) -> list[np.ndarray]:
    """Align a list of 4×4 SE(3) poses and return the aligned poses.

    Camera centres are extracted from each pose, aligned via Umeyama,
    and the resulting Sim(3) is applied to reconstruct the aligned 4×4
    pose list.

    Parameters
    ----------
    est_poses : list of (4, 4) arrays
        Estimated camera-to-world or world-to-camera poses (consistent
        convention required within the list).
    gt_poses : list of (4, 4) arrays
        Ground-truth poses (same convention as *est_poses*).
    with_scale : bool, default True
        Passed through to :func:`align_umeyama`.

    Returns
    -------
    aligned : list of (4, 4) arrays
        Aligned poses (same length as ``min(len(est_poses), len(gt_poses))``).

    Raises
    ------
    ValueError
        If fewer than 3 pose pairs are available.
    """
    n = min(len(est_poses), len(gt_poses))
    if n < 3:
        raise ValueError(f"At least 3 pose pairs required, got {n}")

    est_poses = est_poses[:n]
    gt_poses = gt_poses[:n]

    est_centers = np.array([_get_camera_center(T) for T in est_poses])
    gt_centers = np.array([_get_camera_center(T) for T in gt_poses])

    scale, R, t = align_umeyama(est_centers, gt_centers, with_scale=with_scale)

    aligned: list[np.ndarray] = []
    for T, C_old in zip(est_poses, est_centers):
        R_wc = T[:3, :3]
        # camera-to-world rotation
        R_cw = R_wc.T
        # Aligned camera-to-world rotation and centre
        R_cw_new = R @ R_cw
        C_new = scale * R @ C_old + t
        # Build aligned camera-to-world, then invert to world-to-camera
        T_cw_new = np.eye(4, dtype=np.float64)
        T_cw_new[:3, :3] = R_cw_new
        T_cw_new[:3, 3] = C_new
        aligned.append(np.linalg.inv(T_cw_new))

    return aligned


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def compute_ape(
    estimated_pos: np.ndarray,
    ground_truth_pos: np.ndarray,
) -> np.ndarray:
    """Absolute Pose Error: per-frame Euclidean distance.

    Parameters
    ----------
    estimated_pos : array of shape (N, 3)
    ground_truth_pos : array of shape (N, 3)

    Returns
    -------
    ape : array of shape (N,)
        ``ape[i] = ‖estimated[i] − gt[i]‖``

    Raises
    ------
    ValueError
        If input shapes do not match.
    """
    if estimated_pos.shape != ground_truth_pos.shape:
        raise ValueError(
            f"Shape mismatch: {estimated_pos.shape} vs {ground_truth_pos.shape}"
        )
    return np.linalg.norm(estimated_pos - ground_truth_pos, axis=1)


def compute_rte(
    estimated_pos: np.ndarray,
    ground_truth_pos: np.ndarray,
    window: float = 5.0,
) -> np.ndarray:
    """Relative Trajectory Error with a distance-based sliding window.

    For every frame *i*, the earliest frame *j < i* for which the
    ground-truth path distance ``cumdist(j, i) ≈ window`` is found, and::

        RTE[i] = ‖(est[i] − est[j]) − (gt[i] − gt[j])‖

    Frames for which the available path is shorter than ``0.5 × window``
    are assigned an RTE of zero.

    Parameters
    ----------
    estimated_pos : array of shape (N, 3)
    ground_truth_pos : array of shape (N, 3)
    window : float, default 5.0
        Target path length (metres) for the relative displacement.

    Returns
    -------
    rte : array of shape (N,)
        RTE values (leading frames may be zero where the window is too short).

    Raises
    ------
    ValueError
        If input shapes do not match.
    """
    if estimated_pos.shape != ground_truth_pos.shape:
        raise ValueError(
            f"Shape mismatch: {estimated_pos.shape} vs {ground_truth_pos.shape}"
        )

    n = len(estimated_pos)
    rte = np.zeros(n, dtype=np.float64)

    if n < 2:
        return rte

    # Cumulative arc-length along the ground-truth path
    step_dists = np.linalg.norm(np.diff(ground_truth_pos, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(step_dists)])

    j = 0  # monotonic left pointer
    for i in range(1, n):
        # Advance j as far as possible while keeping the sub-path ≥ window
        while j + 1 < i and (cum_dist[i] - cum_dist[j + 1]) >= window:
            j += 1

        actual_window = cum_dist[i] - cum_dist[j]
        if actual_window < window * 0.5:
            # Sub-path too short — leave rte[i] = 0
            continue

        rel_est = estimated_pos[i] - estimated_pos[j]
        rel_gt = ground_truth_pos[i] - ground_truth_pos[j]
        rte[i] = np.linalg.norm(rel_est - rel_gt)

    return rte


# ---------------------------------------------------------------------------
# Unified evaluation
# ---------------------------------------------------------------------------

def evaluate(
    estimated: list[np.ndarray],
    ground_truth: list[np.ndarray],
    with_scale: bool = True,
) -> dict:
    """Compute a full alignment + error report for a pose trajectory.

    The estimated trajectory is first aligned to ground-truth via Umeyama
    (Sim(3) or SE(3)), then APE and RTE are computed on the aligned
    camera centres.

    Parameters
    ----------
    estimated : list of (4, 4) arrays
        Estimated poses (world-to-camera convention).
    ground_truth : list of (4, 4) arrays
        Ground-truth poses.
    with_scale : bool, default True
        Whether to allow scale correction during alignment.

    Returns
    -------
    report : dict with keys
        ``ape_rmse``  – RMSE of Absolute Pose Error (float)
        ``ape_mean``  – Mean APE (float)
        ``rte_rmse``  – RMSE of RTE over valid (non-zero) entries (float)
        ``rte_mean``  – Mean RTE over valid entries (float)
        ``scale``     – Umeyama scale factor recovered (float)
        ``num_frames`` – Number of frames evaluated (int)
    """
    n = min(len(estimated), len(ground_truth))
    est = list(estimated[:n])
    gt = list(ground_truth[:n])

    # Align and get scale
    est_centers_raw = np.array([_get_camera_center(T) for T in est])
    gt_centers = np.array([_get_camera_center(T) for T in gt])
    scale, _, _ = align_umeyama(est_centers_raw, gt_centers, with_scale=with_scale)

    aligned_poses = align_trajectories(est, gt, with_scale=with_scale)
    est_pos_aligned = np.array([_get_camera_center(T) for T in aligned_poses])

    ape = compute_ape(est_pos_aligned, gt_centers)
    rte = compute_rte(est_pos_aligned, gt_centers)

    rte_valid = rte[rte > 0.0]
    rte_mean = float(np.mean(rte_valid)) if len(rte_valid) > 0 else 0.0
    rte_rmse = float(np.sqrt(np.mean(rte_valid ** 2))) if len(rte_valid) > 0 else 0.0

    return {
        "ape_rmse": float(np.sqrt(np.mean(ape ** 2))),
        "ape_mean": float(np.mean(ape)),
        "rte_rmse": rte_rmse,
        "rte_mean": rte_mean,
        "scale": float(scale),
        "num_frames": n,
    }
