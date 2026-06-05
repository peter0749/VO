"""Tests for TrajectoryAccumulator in trajectory.py."""
import numpy as np
import numpy.testing as npt
import pytest

from slam_dnn.trajectory import TrajectoryAccumulator


class TestTrajectoryInit:
    """Test initialization and basic properties."""

    def test_init_single_identity_pose(self):
        """Initial trajectory has exactly 1 pose (identity)."""
        traj = TrajectoryAccumulator(scale=1.0)
        assert len(traj.poses) == 1
        npt.assert_allclose(traj.poses[0], np.eye(4))

    def test_init_custom_scale(self):
        """Scale parameter is stored."""
        traj = TrajectoryAccumulator(scale=2.5)
        assert traj.scale == 2.5

    def test_get_positions_returns_shape(self):
        """get_positions returns (1, 3) for initial identity."""
        traj = TrajectoryAccumulator(scale=1.0)
        positions = traj.get_positions()
        assert positions.shape == (1, 3)
        npt.assert_allclose(positions[0], np.zeros(3))


class TestTrajectoryAccumulation:
    """Test pose accumulation behavior."""

    def test_accumulate_identity_poses(self):
        """10 identity poses → camera stays at origin."""
        traj = TrajectoryAccumulator(scale=1.0)
        R_id = np.eye(3)
        t_id = np.zeros(3)
        for _ in range(10):
            traj.add_pose(R_id, t_id)

        assert len(traj) == 11  # 1 identity + 10 added
        positions = traj.get_positions()
        for i in range(11):
            npt.assert_allclose(positions[i], np.zeros(3), atol=1e-10)

    def test_accumulate_constant_translation_x(self):
        """5 identical (I, [1,0,0]) poses → positions [0,0,0], [1,0,0], [2,0,0]..."""
        traj = TrajectoryAccumulator(scale=1.0)
        R_id = np.eye(3)
        t_right = np.array([1.0, 0.0, 0.0])
        for _ in range(5):
            traj.add_pose(R_id, t_right)

        positions = traj.get_positions()
        assert positions.shape == (6, 3)
        npt.assert_allclose(positions[0], [0, 0, 0])
        npt.assert_allclose(positions[1], [1, 0, 0])
        npt.assert_allclose(positions[2], [2, 0, 0])
        npt.assert_allclose(positions[3], [3, 0, 0])
        npt.assert_allclose(positions[4], [4, 0, 0])
        npt.assert_allclose(positions[5], [5, 0, 0])

    def test_accumulate_pure_rotation_in_place(self):
        """Pure rotation (t=0) → camera center stays at origin."""
        traj = TrajectoryAccumulator(scale=1.0)
        angle = np.radians(10)
        R_rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1],
        ])
        t_zero = np.zeros(3)

        for _ in range(36):  # 360 degrees total
            traj.add_pose(R_rot, t_zero)

        positions = traj.get_positions()
        # All positions should be at origin (pure rotation in place)
        for i in range(len(positions)):
            npt.assert_allclose(positions[i], np.zeros(3), atol=1e-10)

    def test_accumulate_with_scale_2(self):
        """scale=2.0 doubles the translation magnitude."""
        traj = TrajectoryAccumulator(scale=2.0)
        R_id = np.eye(3)
        t_right = np.array([1.0, 0.0, 0.0])
        for _ in range(5):
            traj.add_pose(R_id, t_right)

        positions = traj.get_positions()
        # With scale=2.0, each step should move 2 units (not 1)
        npt.assert_allclose(positions[5], [10, 0, 0], atol=1e-10)


class TestTrajectoryReset:
    """Test reset behavior."""

    def test_reset_clears_trajectory(self):
        """Reset returns trajectory to initial state."""
        traj = TrajectoryAccumulator(scale=1.0)
        for _ in range(5):
            traj.add_pose(np.eye(3), np.array([1, 0, 0]))

        assert len(traj) == 6  # before reset

        traj.reset()
        assert len(traj) == 1  # after reset
        npt.assert_allclose(traj.poses[0], np.eye(4))
        npt.assert_allclose(traj.get_positions()[0], np.zeros(3))


class TestTranslationNormalization:
    """Test that translation is normalized correctly."""

    def test_unit_translation_unchanged(self):
        """Translation of norm 1 is applied as-is."""
        traj = TrajectoryAccumulator(scale=1.0)
        R_id = np.eye(3)
        t_unit = np.array([1.0, 0.0, 0.0])  # norm = 1
        traj.add_pose(R_id, t_unit)

        positions = traj.get_positions()
        # Camera center = -R^T @ t = -t for identity rotation
        npt.assert_allclose(positions[1], [1, 0, 0], atol=1e-10)

    def test_large_translation_normalized_to_unit(self):
        """Large translation (norm 5) is normalized to unit before scaling."""
        traj = TrajectoryAccumulator(scale=1.0)
        R_id = np.eye(3)
        t_large = np.array([5.0, 0.0, 0.0])  # norm = 5
        traj.add_pose(R_id, t_large)

        positions = traj.get_positions()
        # Despite input norm=5, output should be 1 (normalized)
        npt.assert_allclose(positions[1], [1, 0, 0], atol=1e-10)

    def test_zero_translation_handled(self):
        """Zero translation (pure rotation) doesn't crash."""
        traj = TrajectoryAccumulator(scale=1.0)
        R_id = np.eye(3)
        t_zero = np.zeros(3)
        traj.add_pose(R_id, t_zero)

        positions = traj.get_positions()
        npt.assert_allclose(positions[1], np.zeros(3), atol=1e-10)


class TestTrajectoryEdgeCases:
    """Additional edge cases teaching trajectory accumulation limits."""

    def test_trajectory_circular_motion(self):
        """360° rotation in 36 steps (10° each) returns camera to origin.

        A circular path is the canonical test for rotation accumulation:
        after a full revolution, the camera center must be back at the start
        (within numerical precision). This catches sign errors in rotation
        composition.
        """
        traj = TrajectoryAccumulator(scale=1.0)
        angle = np.radians(10.0)
        R_rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1],
        ])
        t_zero = np.zeros(3)

        for _ in range(36):
            traj.add_pose(R_rot, t_zero)

        positions = traj.get_positions()
        npt.assert_allclose(positions[-1], positions[0], atol=1e-8,
                            err_msg="Circular motion should return to origin")

    def test_trajectory_accumulator_reset_idempotent(self):
        """Calling reset() twice should behave identically to calling it once.

        Teaches: reset() is safe to call multiple times (e.g., in cleanup
        code or error handlers). It's idempotent — already-reset state
        stays reset.
        """
        traj = TrajectoryAccumulator(scale=1.0)
        for _ in range(5):
            traj.add_pose(np.eye(3), np.array([1, 0, 0]))

        traj.reset()
        traj.reset()

        assert len(traj) == 1
        npt.assert_allclose(traj.poses[0], np.eye(4))
        npt.assert_allclose(traj.get_positions()[0], np.zeros(3))

    def test_trajectory_large_number_of_poses(self):
        """1000 identity poses accumulate without overflow or slowdown.

        Teaches: trajectory accumulation is O(n) and uses float64, so
        thousands of poses won't cause numerical drift or performance issues
        in typical SLAM scenarios (< 10k frames). Note: translation is
        normalized to unit length, so t=[0.1,0,0] becomes [1,0,0] per step.
        """
        traj = TrajectoryAccumulator(scale=1.0)
        R_id = np.eye(3)
        t_right = np.array([0.1, 0.0, 0.0])

        for _ in range(1000):
            traj.add_pose(R_id, t_right)

        assert len(traj) == 1001
        positions = traj.get_positions()
        npt.assert_allclose(positions[-1], [1000.0, 0, 0], atol=1e-6,
                            err_msg="1000 steps (normalized t=[1,0,0]) → 1000 units")

    def test_trajectory_scale_zero(self):
        """scale=0 freezes the camera at the origin (no translation applied).

        While not a typical use case, scale=0 is valid and should not crash.
        It effectively disables translation accumulation while still tracking
        rotation.
        """
        traj = TrajectoryAccumulator(scale=0.0)
        for _ in range(5):
            traj.add_pose(np.eye(3), np.array([1.0, 0.0, 0.0]))

        assert len(traj) == 6
        positions = traj.get_positions()
        for i in range(6):
            npt.assert_allclose(positions[i], np.zeros(3), atol=1e-10,
                                err_msg="scale=0 should keep camera at origin")
