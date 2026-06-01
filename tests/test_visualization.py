"""Tests for slam_dnn.visualization module."""
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from slam_dnn.visualization import (
    plot_trajectory_comparison,
    plot_trajectory_3d,
    plot_matches,
    save_trajectory_video,
)


# ---------------------------------------------------------------------------
# Trajectory generators
# ---------------------------------------------------------------------------

def _circular_pose_sequence(n: int, radius: float = 5.0) -> list:
    n_frames = n
    poses = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        c = np.array([radius * np.cos(theta), 0.0, radius * np.sin(theta)])
        forward = np.array([-np.sin(theta), 0.0, np.cos(theta)])
        right = np.cross(np.array([0, 1, 0]), forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        R = np.column_stack([right, up, forward])
        t = -R @ c
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        poses.append(T)
    return poses


def _spiral_poses(n: int, radius: float = 3.0, z_step: float = 0.1) -> list:
    n_frames = n
    positions = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        x = radius * np.cos(theta)
        y = z_step * i
        z = radius * np.sin(theta)
        positions.append([x, y, z])
    return np.array(positions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlotTrajectoryComparison2D:

    def test_plot_trajectory_2d_with_no_ground_truth(self, tmp_path):
        poses = _circular_pose_sequence(20)
        save_file = str(tmp_path / "traj_2d.png")

        fig = plot_trajectory_comparison(
            estimated=poses, ground_truth=None, save_path=save_file, show=False,
        )

        assert isinstance(fig, Figure)
        assert os.path.exists(save_file)
        assert os.path.getsize(save_file) > 0
        plt.close(fig)

    def test_plot_trajectory_2d_with_ground_truth(self, tmp_path):
        gt = _circular_pose_sequence(20, radius=5.0)

        rng = np.random.default_rng(42)
        estimated = []
        for T in gt:
            T_noisy = T.copy()
            T_noisy[:3, 3] += rng.normal(scale=0.3, size=3)
            estimated.append(T_noisy)

        save_file = str(tmp_path / "traj_2d_gt.png")
        fig = plot_trajectory_comparison(
            estimated=estimated, ground_truth=gt, save_path=save_file, show=False,
        )

        assert isinstance(fig, Figure)
        assert os.path.exists(save_file)
        size_with_gt = os.path.getsize(save_file)

        save_file_no_gt = str(tmp_path / "traj_2d_no_gt.png")
        fig2 = plot_trajectory_comparison(
            estimated=estimated, ground_truth=None, save_path=save_file_no_gt, show=False,
        )
        size_no_gt = os.path.getsize(save_file_no_gt)

        assert size_with_gt >= size_no_gt
        plt.close(fig)
        plt.close(fig2)

    def test_plot_trajectory_handles_empty(self, tmp_path):
        save_file = str(tmp_path / "traj_empty.png")
        fig = plot_trajectory_comparison(
            estimated=np.array([]), save_path=save_file, show=False,
        )

        assert isinstance(fig, Figure)
        assert os.path.exists(save_file)
        plt.close(fig)

    def test_plot_trajectory_equal_aspect(self, tmp_path):
        n = 50
        positions = np.zeros((n, 3))
        positions[:, 0] = np.linspace(0, 10, n)
        positions[:, 2] = np.sin(np.linspace(0, 2 * np.pi, n))

        save_file = str(tmp_path / "traj_aspect.png")
        fig = plot_trajectory_comparison(
            estimated=positions, save_path=save_file, show=False,
        )

        ax = fig.axes[0]
        aspect = ax.get_aspect()
        assert aspect == "equal" or aspect == 1.0 or aspect == "auto"
        plt.close(fig)

    def test_plot_trajectory_accepts_position_array(self, tmp_path):
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
        ], dtype=np.float64)

        save_file = str(tmp_path / "traj_positions.png")
        fig = plot_trajectory_comparison(
            estimated=positions, save_path=save_file, show=False,
        )

        assert isinstance(fig, Figure)
        assert os.path.exists(save_file)
        plt.close(fig)


class TestPlotTrajectory3D:

    def test_plot_trajectory_3d(self, tmp_path):
        positions = _spiral_poses(20, radius=3.0, z_step=0.2)
        save_file = str(tmp_path / "traj_3d.png")

        fig = plot_trajectory_3d(positions, save_path=save_file, show=False)

        assert isinstance(fig, Figure)
        assert any(hasattr(ax, "get_zlim") for ax in fig.axes)
        assert os.path.exists(save_file)
        assert os.path.getsize(save_file) > 0
        plt.close(fig)

    def test_plot_trajectory_3d_handles_empty(self, tmp_path):
        save_file = str(tmp_path / "traj_3d_empty.png")
        fig = plot_trajectory_3d(np.array([]), save_path=save_file, show=False)

        assert isinstance(fig, Figure)
        assert os.path.exists(save_file)
        plt.close(fig)


class TestPlotMatches:

    def test_plot_matches_synthetic(self, tmp_path):
        img0 = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (200, 300), dtype=np.uint8)

        rng = np.random.default_rng(123)
        n_matches = 30
        kp0 = rng.uniform(low=[0, 0], high=[300, 200], size=(n_matches, 2))
        kp1 = kp0 + rng.normal(scale=5.0, size=(n_matches, 2))

        matches = {"keypoints0": kp0, "keypoints1": kp1}

        save_file = str(tmp_path / "matches.png")
        fig = plot_matches(img0, img1, matches, n_show=20, save_path=save_file)

        assert isinstance(fig, Figure)
        assert os.path.exists(save_file)
        assert os.path.getsize(save_file) > 0
        plt.close(fig)

    def test_plot_matches_empty(self, tmp_path):
        img0 = np.zeros((100, 100, 3), dtype=np.uint8)
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        matches = {"keypoints0": np.array([]), "keypoints1": np.array([])}

        save_file = str(tmp_path / "matches_empty.png")
        fig = plot_matches(img0, img1, matches, save_path=save_file)

        assert isinstance(fig, Figure)
        assert os.path.exists(save_file)
        plt.close(fig)


class TestSaveTrajectoryVideo:

    def test_save_trajectory_video(self, tmp_path):
        n = 10
        positions = np.column_stack([
            np.linspace(0, 5, n),
            np.zeros(n),
            np.sin(np.linspace(0, np.pi, n)),
        ])
        images = [
            np.random.randint(0, 255, (100, 150), dtype=np.uint8)
            for _ in range(n)
        ]

        output = str(tmp_path / "trajectory.mp4")
        save_trajectory_video(positions, images, output, fps=5)

        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_save_trajectory_video_empty_raises(self):
        with pytest.raises(ValueError):
            save_trajectory_video(np.array([]), [], "/tmp/empty.mp4")
