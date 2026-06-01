"""Tests for slam_dnn.eval — alignment + APE/RTE metrics."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from slam_dnn.eval import (
    align_umeyama,
    align_trajectories,
    compute_ape,
    compute_rte,
    evaluate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_z(angle_deg: float) -> np.ndarray:
    """3×3 rotation matrix about Z axis."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _straight_trajectory(n: int = 50, step: float = 1.0) -> list[np.ndarray]:
    """List of 4×4 SE(3) poses moving along +X with the given step."""
    poses = []
    for i in range(n):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = i * step
        poses.append(T)
    return poses


# ---------------------------------------------------------------------------
# Umeyama alignment
# ---------------------------------------------------------------------------


class TestUmeyama:
    def test_umeyama_identity_alignment(self):
        """Same trajectory as est and gt → scale=1, R=I, t=0."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-10.0, 10.0, (25, 3))

        scale, R, t = align_umeyama(pts, pts)

        assert abs(scale - 1.0) < 1e-6
        npt.assert_allclose(R, np.eye(3), atol=1e-8)
        npt.assert_allclose(t, np.zeros(3), atol=1e-8)

        recovered = (scale * R @ pts.T).T + t
        npt.assert_allclose(recovered, pts, atol=1e-8)

    def test_umeyama_known_scale_and_rotation(self):
        """Recover scale=2.0, Rz(30°), t=[1,2,3] from a random point set."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-10.0, 10.0, (40, 3))

        s_true = 2.0
        R_true = _rotation_z(30.0)
        t_true = np.array([1.0, 2.0, 3.0])
        gt = (s_true * (R_true @ pts.T)).T + t_true

        scale, R, t = align_umeyama(pts, gt)

        # Scale within 1 %
        assert 1.99 <= scale <= 2.01, f"scale={scale}"

        # Rotation error < 0.5°
        R_err = R @ R_true.T
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        assert angle_deg < 0.5, f"rotation error = {angle_deg:.3f}°"

        # Translation (allow numerical tolerance)
        npt.assert_allclose(t, t_true, atol=0.02)

    def test_umeyama_no_scale(self):
        """with_scale=False → scale is exactly 1.0 even when true scale ≠ 1."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-5.0, 5.0, (30, 3))

        R_true = _rotation_z(45.0)
        t_true = np.array([5.0, -3.0, 2.0])
        # Apply only rotation + translation (no scale)
        gt = (R_true @ pts.T).T + t_true

        scale, R, t = align_umeyama(pts, gt, with_scale=False)

        assert scale == 1.0
        R_err = R @ R_true.T
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        assert angle_deg < 0.5, f"rotation error = {angle_deg:.3f}°"
        npt.assert_allclose(t, t_true, atol=0.02)

    def test_umeyama_raises_on_shape_mismatch(self):
        a = np.zeros((5, 3))
        b = np.zeros((6, 3))
        with pytest.raises(ValueError, match="Shape mismatch"):
            align_umeyama(a, b)

    def test_umeyama_raises_on_few_points(self):
        a = np.zeros((2, 3))
        with pytest.raises(ValueError, match="At least 3"):
            align_umeyama(a, a)


# ---------------------------------------------------------------------------
# APE
# ---------------------------------------------------------------------------


class TestAPE:
    def test_ape_zero_error(self):
        """Identical inputs → all APE values ≈ 0."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-10.0, 10.0, (30, 3))
        ape = compute_ape(pts, pts)
        assert np.all(ape < 1e-10)

    def test_ape_constant_offset(self):
        """Offset of [5, 0, 0] on every frame → all APE = 5.0."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-10.0, 10.0, (20, 3))
        shifted = pts + np.array([5.0, 0.0, 0.0])
        ape = compute_ape(shifted, pts)
        npt.assert_allclose(ape, 5.0, atol=1e-10)

    def test_ape_raises_on_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_ape(np.zeros((5, 3)), np.zeros((5, 2)))


# ---------------------------------------------------------------------------
# RTE
# ---------------------------------------------------------------------------


class TestRTE:
    def test_rte_zero_for_identical(self):
        """Identical trajectories → all RTE = 0."""
        rng = np.random.default_rng(42)
        pts = np.cumsum(rng.uniform(0.5, 1.5, (40, 3)), axis=0)
        rte = compute_rte(pts, pts, window=5.0)
        npt.assert_allclose(rte, 0.0, atol=1e-10)

    def test_rte_constant_drift_grows(self):
        """Position bias growing per frame → RTE increases over time.

        estimated[i, 0] = gt[i, 0] × (1 + 0.01 × i) gives quadratic
        cumulative position error, which manifests as linearly growing
        RTE within a fixed-distance window.
        """
        n = 80
        step = 1.0
        gt_pos = np.zeros((n, 3), dtype=np.float64)
        gt_pos[:, 0] = np.arange(n, dtype=np.float64) * step

        # Multiplicative scale drift growing per frame
        est_pos = gt_pos.copy()
        for i in range(n):
            est_pos[i, 0] = gt_pos[i, 0] * (1.0 + 0.01 * i)

        rte = compute_rte(est_pos, gt_pos, window=5.0)

        valid = rte[rte > 0.0]
        assert len(valid) > 10, "Expected many valid RTE entries"

        # Later values must be larger on average than earlier values
        mid = len(valid) // 2
        early_mean = float(np.mean(valid[:max(1, mid)]))
        late_mean = float(np.mean(valid[mid:]))
        assert late_mean > early_mean, (
            f"Expected growing RTE: early={early_mean:.4f}, late={late_mean:.4f}"
        )


# ---------------------------------------------------------------------------
# align_trajectories
# ---------------------------------------------------------------------------


class TestAlignTrajectories:
    def test_align_identity(self):
        """Identical trajectories returned unchanged (up to numerical precision)."""
        poses = _straight_trajectory(n=20, step=1.0)
        aligned = align_trajectories(poses, poses, with_scale=True)
        assert len(aligned) == 20
        for orig, al in zip(poses, aligned):
            npt.assert_allclose(al, orig, atol=1e-8)


# ---------------------------------------------------------------------------
# evaluate() full pipeline
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_full_pipeline(self):
        """Identical pose lists → near-zero errors, scale ≈ 1, correct keys."""
        poses = _straight_trajectory(n=30, step=1.0)
        result = evaluate(poses, poses, with_scale=True)

        required_keys = {"ape_rmse", "ape_mean", "rte_rmse", "rte_mean", "scale", "num_frames", "aligned_poses"}
        assert required_keys == set(result.keys()), f"keys={set(result.keys())}"

        assert result["num_frames"] == 30
        assert result["ape_rmse"] < 1e-6
        assert result["ape_mean"] < 1e-6
        assert abs(result["scale"] - 1.0) < 1e-6
        # RTE is zero for identical trajectories (all entries zero)
        assert result["rte_rmse"] == 0.0
        assert result["rte_mean"] == 0.0

    def test_evaluate_with_known_transform(self):
        """Evaluate recovers scale and low APE for a known Sim(3) offset.

        Constructs est poses whose camera centres satisfy:
            gt_center = s_true * R_true @ est_center + t_true
        so Umeyama alignment should recover s_true, R_true, t_true exactly.
        """
        n = 40
        # Non-collinear 3-D trajectory for well-conditioned SVD
        gt_poses: list[np.ndarray] = []
        for i in range(n):
            T = np.eye(4)
            T[0, 3] = i * 0.5
            T[1, 3] = 2.0 * np.sin(i * 0.15)
            T[2, 3] = 0.5 * np.cos(i * 0.2)
            gt_poses.append(T)

        s_true = 2.0
        R_true = _rotation_z(30.0)
        t_true = np.array([1.0, 2.0, 0.0])

        # est_center = (1/s) * R^T @ (gt_center − t)
        est_poses: list[np.ndarray] = []
        for T_gt in gt_poses:
            C_gt = -T_gt[:3, :3].T @ T_gt[:3, 3]
            C_est = (1.0 / s_true) * R_true.T @ (C_gt - t_true)
            T_est = np.eye(4)
            T_est[:3, 3] = -C_est  # center = −I^T @ t_est ⟹ t_est = −center
            est_poses.append(T_est)

        result = evaluate(est_poses, gt_poses, with_scale=True)

        assert abs(result["scale"] - s_true) < 0.1, f"scale={result['scale']}"
        assert result["ape_rmse"] < 0.5, f"ape_rmse={result['ape_rmse']}"
        assert result["num_frames"] == n

    def test_evaluate_with_ground_truth_shorter(self):
        """gt shorter than est → truncate to min length gracefully."""
        est_poses = _straight_trajectory(n=40, step=1.0)
        gt_poses = _straight_trajectory(n=20, step=1.0)

        result = evaluate(est_poses, gt_poses)

        assert result["num_frames"] == 20
        assert "ape_rmse" in result
        assert result["ape_rmse"] < 1e-6


class TestEvaluateEdgeCases:
    """Additional evaluation edge cases."""

    def test_evaluate_with_scale_only(self):
        """evaluate() recovers pure scale difference (no rotation/translation offset).

        Teaches: when est trajectory camera centers are exactly 2x the
        ground truth centers, Umeyama alignment finds the scale that maps
        est→gt, so scale ≈ 0.5 (shrinking est to match gt).
        Uses a non-collinear trajectory (sinusoidal y-offset) for SVD stability.
        """
        n = 30
        gt_poses = []
        est_poses = []
        for i in range(n):
            T_gt = np.eye(4)
            T_gt[0, 3] = i * 1.0
            T_gt[1, 3] = np.sin(i * 0.3)
            gt_poses.append(T_gt)

            T_est = np.eye(4)
            T_est[0, 3] = T_gt[0, 3] * 2.0
            T_est[1, 3] = T_gt[1, 3] * 2.0
            est_poses.append(T_est)

        result = evaluate(est_poses, gt_poses, with_scale=True)
        assert abs(result["scale"] - 0.5) < 0.1, f"Expected scale≈0.5, got {result['scale']}"
        assert result["ape_rmse"] < 1.0, f"Aligned APE should be low, got {result['ape_rmse']}"

    def test_evaluate_no_scale_recovers_identity(self):
        """with_scale=False on identical trajectories gives scale=1, APE=0.

        Teaches: with_scale=False disables Sim(3) scale recovery, forcing
        SE(3)-only alignment. For identical poses this changes nothing.
        """
        poses = _straight_trajectory(n=20, step=1.0)
        result = evaluate(poses, poses, with_scale=False)
        assert result["scale"] == 1.0
        assert result["ape_rmse"] < 1e-6
        assert result["num_frames"] == 20

    def test_rte_returns_zeros_for_insufficient_path_length(self):
        """RTE assigns zero to frames where the accumulated path < 0.5*window.

        Teaches: the first few frames of a short trajectory won't have enough
        path length for the sliding window. These are marked RTE=0 and excluded
        from the valid RTE statistics.
        """
        n = 10
        gt_pos = np.zeros((n, 3))
        gt_pos[:, 0] = np.arange(n) * 0.5
        est_pos = gt_pos + np.array([0.1, 0.0, 0.0])

        rte = compute_rte(est_pos, gt_pos, window=5.0)
        assert rte.shape == (n,)
        assert rte[0] == 0.0
        assert rte[1] == 0.0
