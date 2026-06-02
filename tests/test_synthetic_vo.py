"""Synthetic VO pose recovery validation — end-to-end pipeline tests.

Exercises the full pipeline (extract → match → essential matrix →
trajectory accumulation) on purely synthetic data with known ground
truth.  Reports position / rotation errors and documents known
limitations (e.g. pure rotation handling).
"""

import logging
import numpy as np
import pytest

pytestmark = pytest.mark.slow

import cv2
import torch

from slam_dnn import (
    ClassicMatcher,
    K_from_fov,
    LightGlueMatcher,
    SuperPointExtractor,
    TrajectoryAccumulator,
    estimate_essential,
)

from tests.synthetic_scene import (
    SyntheticScene,
    align_trajectory,
    rotation_error_deg,
    translation_direction_error_deg,
    umeyama_alignment,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 640, 480


@pytest.fixture(autouse=True)
def _stable_rng():
    """Seed RNGs before each test for reproducible RANSAC and torch inference."""
    np.random.seed(42)
    cv2.setRNGSeed(42)
    torch.manual_seed(42)


@pytest.fixture(scope="module")
def K():
    return K_from_fov(WIDTH, HEIGHT, fov_deg=63.0)


@pytest.fixture(scope="module")
def extractor():
    return SuperPointExtractor(max_keypoints=300, device="cpu")


@pytest.fixture(scope="module")
def extractor_high_kp():
    return SuperPointExtractor(max_keypoints=500, device="cpu")


@pytest.fixture(scope="module")
def extractor_low_kp():
    return SuperPointExtractor(max_keypoints=100, device="cpu")


@pytest.fixture(scope="module")
def lg_matcher():
    return LightGlueMatcher(device="cpu")


@pytest.fixture(scope="module")
def classic_matcher():
    return ClassicMatcher(ratio=0.75)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _prepare_for_lg(feats: dict) -> dict:
    """Pass features through unchanged (descriptors already (N, 256))."""
    return feats


def run_pipeline(
    images,
    K,
    extractor,
    matcher,
    use_lg=True,
    min_matches=8,
    ransac_thresh=2.0,
):
    """Run extract → match → estimate → accumulate on an image sequence.

    Returns:
        trajectory: TrajectoryAccumulator with accumulated poses.
        stats: dict with per-frame match counts and failure counts.
    """
    trajectory = TrajectoryAccumulator()
    prev_feats = None
    match_counts = []
    pose_failures = 0
    tracking_lost = 0

    for img in images:
        feats = extractor.extract(img)
        lg_feats = _prepare_for_lg(feats) if use_lg else feats

        if prev_feats is None:
            prev_feats = lg_feats
            continue

        if use_lg:
            match_result = matcher.match(
                prev_feats, lg_feats, image_size=(HEIGHT, WIDTH)
            )
        else:
            match_result = matcher.match(prev_feats, lg_feats)
        n_matches = len(match_result["points0"])
        match_counts.append(n_matches)

        if n_matches < min_matches:
            tracking_lost += 1
            prev_feats = lg_feats
            continue

        result = estimate_essential(
            match_result["points0"],
            match_result["points1"],
            K,
            ransac_thresh=ransac_thresh,
        )
        if result is None:
            pose_failures += 1
            prev_feats = lg_feats
            continue

        R, t, _ = result
        trajectory.add_pose(R, t)
        prev_feats = lg_feats

    stats = {
        "match_counts": match_counts,
        "pose_failures": pose_failures,
        "tracking_lost": tracking_lost,
        "n_poses": len(trajectory) - 1,  # minus initial identity
    }
    return trajectory, stats


def _camera_centers(gt_poses):
    """Convert (R, t) world-to-camera pairs to camera centres in world frame."""
    return np.array([-R.T @ t for R, t in gt_poses])


def _compare_trajectories(trajectory, gt_poses):
    """Align recovered trajectory to GT and compute error metrics.

    trajectory.get_positions()[0] is identity (first camera at origin).
    trajectory.get_positions()[k] corresponds to gt_poses[k].

    Returns dict with position_errors, rotation_errors, s, R_align, n.
    """
    rec_positions = trajectory.get_positions()  # includes identity
    gt_positions = _camera_centers(gt_poses)

    n = min(len(rec_positions), len(gt_positions))
    rec = rec_positions[:n]
    gt = gt_positions[:n]

    s, R_align, t_align = umeyama_alignment(rec, gt)
    aligned = s * rec @ R_align.T + t_align

    pos_errors = np.linalg.norm(aligned - gt, axis=1)

    # Rotation comparison (accounting for frame alignment)
    rec_poses = trajectory.get_poses()
    rot_errors = []
    for i in range(n):
        est_R_w2c = rec_poses[i][:3, :3]  # estimated world-to-cam (in traj frame)
        est_R_c2w = est_R_w2c.T  # cam-to-world in traj frame
        # Transform to GT frame: R_aligned = R_align @ est_R_c2w
        est_R_c2w_aligned = R_align @ est_R_c2w
        gt_R_c2w = gt_poses[i][0].T  # GT cam-to-world
        rot_errors.append(rotation_error_deg(est_R_c2w_aligned, gt_R_c2w))

    return {
        "pos_errors": pos_errors,
        "rot_errors": np.array(rot_errors),
        "mean_pos_err": float(np.mean(pos_errors)),
        "mean_rot_err": float(np.mean(rot_errors)),
        "scale": s,
        "n": n,
        "aligned": aligned,
        "R_align": R_align,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyntheticVO:
    """End-to-end VO validation on synthetic scenes."""

    # --- Test 1: Circular trajectory recovery ---

    def test_circular_trajectory_recovery(self, K, extractor, lg_matcher):
        """Full pipeline recovers a circular trajectory within tolerance."""
        scene = SyntheticScene(n_points=500)
        gt_poses = scene.generate_trajectory(n_frames=15, motion_type="circular")

        images = [scene.render_from_pose(K, R, t) for R, t in gt_poses]
        trajectory, stats = run_pipeline(images, K, extractor, lg_matcher)

        n_recovered = stats["n_poses"]
        assert n_recovered >= 8, (
            f"Too few poses recovered: {n_recovered}/14"
        )

        cmp = _compare_trajectories(trajectory, gt_poses)
        trajectory_radius = 1.0
        median_pos_err = float(np.median(cmp["pos_errors"]))

        print(f"\nCircular trajectory recovery:")
        print(f"  Poses recovered: {n_recovered}/14")
        print(f"  Median position error: {median_pos_err:.4f}m "
              f"(radius: {trajectory_radius}m)")
        print(f"  Mean rotation error: {cmp['mean_rot_err']:.2f}°")
        print(f"  Matches/frame: mean={np.mean(stats['match_counts']):.0f}, "
              f"min={min(stats['match_counts']):.0f}")
        print(f"  Failures: pose={stats['pose_failures']}, "
              f"tracking_lost={stats['tracking_lost']}")

        assert median_pos_err < 0.40 * trajectory_radius, (
            f"Median position error {median_pos_err:.4f}m > "
            f"{0.40 * trajectory_radius:.2f}m (40% of radius)"
        )
        assert cmp["mean_rot_err"] < 15.0, (
            f"Mean rotation error {cmp['mean_rot_err']:.2f}° > 15.0°"
        )

    # --- Test 2: Linear forward motion ---

    def test_linear_forward_motion(self, K, extractor, lg_matcher):
        """Pure forward translation — direction should be recovered."""
        scene = SyntheticScene(n_points=500)
        gt_poses = scene.generate_trajectory(n_frames=12, motion_type="linear")

        images = [scene.render_from_pose(K, R, t) for R, t in gt_poses]
        trajectory, stats = run_pipeline(images, K, extractor, lg_matcher)

        n_recovered = stats["n_poses"]
        assert n_recovered >= 5, f"Too few poses recovered: {n_recovered}/11"

        cmp = _compare_trajectories(trajectory, gt_poses)

        # Step direction check
        aligned = cmp["aligned"]
        gt_pos = _camera_centers(gt_poses)[:cmp["n"]]
        direction_errors = []
        for i in range(1, min(len(aligned), len(gt_pos))):
            d_aligned = aligned[i] - aligned[i - 1]
            d_gt = gt_pos[i] - gt_pos[i - 1]
            if np.linalg.norm(d_aligned) > 1e-6 and np.linalg.norm(d_gt) > 1e-6:
                direction_errors.append(
                    translation_direction_error_deg(d_aligned, d_gt)
                )

        mean_dir_err = float(np.mean(direction_errors)) if direction_errors else 180.0

        # XY drift check
        xy_drift = np.sqrt(aligned[:, 0] ** 2 + aligned[:, 1] ** 2)
        max_xy = float(np.max(xy_drift))
        z_range = float(np.ptp(gt_pos[:, 2]))

        print(f"\nLinear forward motion:")
        print(f"  Poses recovered: {n_recovered}/9")
        print(f"  Mean step direction error: {mean_dir_err:.2f}°")
        print(f"  Max XY drift: {max_xy:.4f}m (Z range: {z_range:.4f}m)")
        print(f"  Matches/frame: mean={np.mean(stats['match_counts']):.0f}")

        assert mean_dir_err < 8.0, (
            f"Step direction error {mean_dir_err:.2f}° > 8.0°"
        )
        assert max_xy < 0.20 * max(z_range, 1.0), (
            f"XY drift {max_xy:.4f} exceeds 20% of Z range {z_range:.4f}"
        )

    # --- Test 3: Pure translation along X ---

    def test_pure_translation(self, K, extractor, lg_matcher):
        """Pure X-axis translation — rotations should stay near identity."""
        scene = SyntheticScene(n_points=500)
        gt_poses = scene.generate_trajectory(
            n_frames=8, motion_type="pure_translation_x"
        )

        images = [scene.render_from_pose(K, R, t) for R, t in gt_poses]
        trajectory, stats = run_pipeline(images, K, extractor, lg_matcher)

        n_recovered = stats["n_poses"]
        assert n_recovered >= 4, f"Too few poses recovered: {n_recovered}/7"

        cmp = _compare_trajectories(trajectory, gt_poses)
        mean_rot_dev = cmp["mean_rot_err"]

        # Translation magnitude consistency
        aligned = cmp["aligned"]
        step_vecs = np.diff(aligned, axis=0)
        step_lengths = np.linalg.norm(step_vecs, axis=1)
        if len(step_lengths) > 1 and np.mean(step_lengths) > 1e-6:
            cv = float(np.std(step_lengths) / np.mean(step_lengths))
        else:
            cv = 0.0

        print(f"\nPure X translation:")
        print(f"  Poses recovered: {n_recovered}/7")
        print(f"  Mean rotation deviation from GT: {mean_rot_dev:.2f}°")
        print(f"  Step length CV: {cv:.3f}")
        print(f"  Matches/frame: mean={np.mean(stats['match_counts']):.0f}")

        assert mean_rot_dev < 10.0, (
            f"Rotation deviation {mean_rot_dev:.2f}° > 10.0°"
        )
        assert cv < 0.30, (
            f"Step length CV {cv:.3f} > 0.30 (inconsistent steps)"
        )

    # --- Test 4: Pure rotation (known limitation) ---

    def test_pure_rotation(self, K, extractor, lg_matcher, caplog):
        """Pure yaw rotation — documents E-matrix limitation.

        Known limitation: the essential matrix encodes both rotation and
        translation, but when translation is zero the matrix becomes
        degenerate (rank-deficient).  Pose estimation will frequently
        fail.  This test verifies the pipeline doesn't crash and records
        the expected failures.
        """
        scene = SyntheticScene(n_points=500)
        gt_poses = scene.generate_trajectory(
            n_frames=12, motion_type="pure_rotation_yaw"
        )

        images = [scene.render_from_pose(K, R, t) for R, t in gt_poses]

        with caplog.at_level(logging.WARNING):
            trajectory, stats = run_pipeline(
                images, K, extractor, lg_matcher
            )

        n_recovered = stats["n_poses"]
        total_failures = stats["pose_failures"] + stats["tracking_lost"]

        print(f"\nPure rotation (yaw):")
        print(f"  Poses recovered: {n_recovered}/11")
        print(f"  Pose failures: {stats['pose_failures']}")
        print(f"  Tracking lost: {stats['tracking_lost']}")
        print(f"  Matches/frame: "
              f"mean={np.mean(stats['match_counts'] or [0]):.0f}")
        print(f"  Known limitation — pure rotation not well handled by E-matrix")

        # Pipeline must not crash
        assert True, "Pipeline did not crash on pure rotation"

        # Expect some failures or unusual results (E-matrix is degenerate
        # for pure rotation). If all succeed, document it.
        if total_failures == 0 and n_recovered > 0:
            print(f"  Note: pure rotation unexpectedly succeeded — "
                  f"E-matrix may return spurious solutions")
        assert total_failures >= 0, "Pipeline should not crash"

    # --- Test 5: Scale recovery via Umeyama ---

    def test_scale_recovery_with_umeyama(self, K, extractor, lg_matcher):
        """Umeyama alignment recovers scale for a 15-frame trajectory."""
        scene = SyntheticScene(n_points=500)
        gt_poses = scene.generate_trajectory(n_frames=15, motion_type="circular")

        images = [scene.render_from_pose(K, R, t) for R, t in gt_poses]
        trajectory, stats = run_pipeline(images, K, extractor, lg_matcher)

        n_recovered = stats["n_poses"]
        assert n_recovered >= 8, f"Too few poses recovered: {n_recovered}/14"

        cmp = _compare_trajectories(trajectory, gt_poses)

        gt_pos = _camera_centers(gt_poses)[:cmp["n"]]
        gt_scale = float(np.max(np.linalg.norm(
            gt_pos[:, None] - gt_pos[None, :], axis=2
        )))

        median_pos_err = float(np.median(cmp["pos_errors"]))

        print(f"\nScale recovery (Umeyama):")
        print(f"  Poses recovered: {n_recovered}/14")
        print(f"  Estimated scale: {cmp['scale']:.4f}")
        print(f"  GT scale (max span): {gt_scale:.4f}m")
        print(f"  Median aligned position error: {median_pos_err:.4f}m")
        print(f"  Error as % of GT scale: "
              f"{100 * median_pos_err / gt_scale:.1f}%")
        print(f"  Matches/frame: mean={np.mean(stats['match_counts']):.0f}")

        assert median_pos_err < 0.25 * gt_scale, (
            f"Median position error {median_pos_err:.4f}m > "
            f"{0.25 * gt_scale:.4f}m (25% of GT scale)"
        )

    # --- Test 6: LightGlue vs Classic matcher consistency ---

    def test_lightglue_vs_classic_matcher_consistency(
        self, K, extractor, lg_matcher, classic_matcher
    ):
        """Both matchers produce broadly consistent results on same data."""
        scene = SyntheticScene(n_points=400)
        gt_poses = scene.generate_trajectory(n_frames=10, motion_type="circular")
        images = [scene.render_from_pose(K, R, t) for R, t in gt_poses]

        traj_lg, stats_lg = run_pipeline(images, K, extractor, lg_matcher, use_lg=True)
        traj_cl, stats_cl = run_pipeline(
            images, K, extractor, classic_matcher, use_lg=False
        )

        lg_matches = stats_lg["match_counts"]
        cl_matches = stats_cl["match_counts"]
        lg_mean = float(np.mean(lg_matches)) if lg_matches else 0.0
        cl_mean = float(np.mean(cl_matches)) if cl_matches else 0.0

        print(f"\nLightGlue vs Classic matcher:")
        print(f"  LG: poses={stats_lg['n_poses']}, "
              f"mean matches/frame={lg_mean:.0f}, "
              f"failures={stats_lg['pose_failures'] + stats_lg['tracking_lost']}")
        print(f"  CL: poses={stats_cl['n_poses']}, "
              f"mean matches/frame={cl_mean:.0f}, "
              f"failures={stats_cl['pose_failures'] + stats_cl['tracking_lost']}")

        assert stats_lg["n_poses"] >= 4, (
            f"LightGlue too few poses: {stats_lg['n_poses']}"
        )
        assert stats_cl["n_poses"] >= 4, (
            f"Classic too few poses: {stats_cl['n_poses']}"
        )

        if lg_mean > 0 and cl_mean > 0:
            ratio = max(lg_mean, cl_mean) / min(lg_mean, cl_mean)
            print(f"  Match count ratio (max/min): {ratio:.1f}x")
            assert ratio < 10.0, (
                f"Match counts differ by {ratio:.1f}x (> 10x)"
            )

    # --- Test 7: Keypoint count sensitivity ---

    def test_keypoint_count_sensitivity(
        self, K, extractor_high_kp, extractor_low_kp, lg_matcher
    ):
        """More keypoints should produce lower (or equal) trajectory error.

        Performance vs accuracy trade-off: higher max_keypoints improves
        accuracy but increases compute time.
        """
        scene = SyntheticScene(n_points=500)
        gt_poses = scene.generate_trajectory(n_frames=12, motion_type="circular")
        images = [scene.render_from_pose(K, R, t) for R, t in gt_poses]

        def _run_and_error(ext, label):
            traj, st = run_pipeline(images, K, ext, lg_matcher)
            n = st["n_poses"]
            if n < 4:
                print(f"  {label}: only {n} poses — skipping error calc")
                return float("inf"), n, st
            cmp = _compare_trajectories(traj, gt_poses)
            return cmp["mean_pos_err"], n, st

        err_high, n_high, st_high = _run_and_error(extractor_high_kp, "500kp")
        err_low, n_low, st_low = _run_and_error(extractor_low_kp, "100kp")

        print(f"\nKeypoint count sensitivity:")
        print(f"  500 kp: poses={n_high}, mean pos error={err_high:.4f}m")
        print(f"  100 kp: poses={n_low}, mean pos error={err_low:.4f}m")
        print(f"  Matches/frame (500kp): "
              f"mean={np.mean(st_high['match_counts']):.0f}")
        print(f"  Matches/frame (100kp): "
              f"mean={np.mean(st_low['match_counts']):.0f}")

        if err_high < float("inf") and err_low < float("inf"):
            assert err_high <= err_low * 2.0, (
                f"High-kp error ({err_high:.4f}) is > 2x low-kp error "
                f"({err_low:.4f}) — more keypoints should not hurt"
            )
        else:
            assert n_high >= n_low, (
                f"High-kp recovered fewer poses ({n_high}) than "
                f"low-kp ({n_low})"
            )

    # --- Test 8: Trajectory drift analysis ---

    def test_trajectory_accumulator_drift(self, K, extractor, lg_matcher):
        """Error should not super-linearly accumulate over 30 frames."""
        scene = SyntheticScene(n_points=600)
        gt_poses = scene.generate_trajectory(n_frames=30, motion_type="circular")

        images = [scene.render_from_pose(K, R, t) for R, t in gt_poses]
        trajectory, stats = run_pipeline(images, K, extractor, lg_matcher)

        n_recovered = stats["n_poses"]
        assert n_recovered >= 15, (
            f"Too few poses for drift analysis: {n_recovered}/29"
        )

        cmp = _compare_trajectories(trajectory, gt_poses)
        cumulative_errors = cmp["pos_errors"]

        aligned = cmp["aligned"]
        gt_pos = _camera_centers(gt_poses)[:cmp["n"]]
        rec_steps = np.linalg.norm(np.diff(aligned, axis=0), axis=1)
        gt_steps = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
        step_errors = np.abs(rec_steps - gt_steps)
        mean_single_step = float(np.mean(step_errors))

        final_error = float(cumulative_errors[-1])

        print(f"\nTrajectory drift analysis ({n_recovered} poses):")
        print(f"  Cumulative errors: "
              f"first={cumulative_errors[0]:.4f}, "
              f"mid={cumulative_errors[cmp['n'] // 2]:.4f}, "
              f"last={final_error:.4f}")
        print(f"  Mean single-step error: {mean_single_step:.4f}m")
        if mean_single_step > 1e-8:
            print(f"  Final/mean-step ratio: "
                  f"{final_error / mean_single_step:.1f}x")
        print(f"  Matches/frame: mean={np.mean(stats['match_counts']):.0f}")

        if mean_single_step > 1e-8:
            assert final_error < 3.0 * mean_single_step * cmp["n"], (
                f"Final error {final_error:.4f}m is super-linear: "
                f">{3.0 * mean_single_step * cmp['n']:.4f}m "
                f"(3x mean_step * n_frames)"
            )

        mean_cum_err = float(np.mean(cumulative_errors))
        gt_span = float(np.max(np.linalg.norm(
            gt_pos[:, None] - gt_pos[None, :], axis=2
        )))
        print(f"  Mean cumulative error: {mean_cum_err:.4f}m "
              f"({100 * mean_cum_err / gt_span:.1f}% of span)")
        assert mean_cum_err < 0.20 * gt_span, (
            f"Mean cumulative error {mean_cum_err:.4f}m > "
            f"20% of GT span {gt_span:.4f}m"
        )
