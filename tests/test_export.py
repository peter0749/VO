"""Tests for slam_dnn.export — KITTI and TUM format round-trips + format validation."""

import os
import tempfile

import numpy as np
import pytest

from slam_dnn import (
    TrajectoryAccumulator,
    export_kitti_format,
    export_tum_format,
    load_kitti_format,
    load_tum_format,
)


@pytest.fixture
def sample_poses():
    """Return a list of 5 known 4x4 SE3 matrices."""
    poses = [np.eye(4, dtype=np.float64)]  # identity
    # Translation along x
    T1 = np.eye(4, dtype=np.float64)
    T1[:3, 3] = [1.0, 0.0, 0.0]
    poses.append(T1)
    # Translation along y
    T2 = np.eye(4, dtype=np.float64)
    T2[:3, 3] = [0.0, 2.0, 0.0]
    poses.append(T2)
    # Translation along z
    T3 = np.eye(4, dtype=np.float64)
    T3[:3, 3] = [0.0, 0.0, 3.0]
    poses.append(T3)
    # Combined translation + rotation
    T4 = np.eye(4, dtype=np.float64)
    T4[:3, 3] = [1.0, 2.0, 3.0]
    # 90-degree rotation about z-axis
    R_z = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    T4[:3, :3] = R_z
    poses.append(T4)
    return poses


@pytest.fixture
def sample_timestamps():
    """Return timestamps matching sample_poses."""
    return [0.0, 1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# KITTI round-trip
# ---------------------------------------------------------------------------

def test_kitti_roundtrip(sample_poses, tmp_path):
    """Save + load KITTI, verify poses match within 1e-6."""
    filepath = tmp_path / "kitti.txt"
    export_kitti_format(sample_poses, str(filepath))
    loaded = load_kitti_format(str(filepath))
    assert len(loaded) == len(sample_poses)
    for orig, rec in zip(sample_poses, loaded):
        np.testing.assert_allclose(orig, rec, atol=1e-6)


# ---------------------------------------------------------------------------
# TUM round-trip
# ---------------------------------------------------------------------------

def test_tum_roundtrip(sample_poses, sample_timestamps, tmp_path):
    """Save + load TUM with timestamps, verify matches."""
    filepath = tmp_path / "tum.txt"
    export_tum_format(sample_poses, str(filepath), timestamps=sample_timestamps)
    loaded_poses, loaded_ts = load_tum_format(str(filepath))
    assert len(loaded_poses) == len(sample_poses)
    assert len(loaded_ts) == len(sample_timestamps)
    for orig, rec in zip(sample_poses, loaded_poses):
        np.testing.assert_allclose(orig, rec, atol=1e-6)
    np.testing.assert_allclose(sample_timestamps, loaded_ts, atol=1e-6)


def test_tum_roundtrip_no_timestamps(sample_poses, tmp_path):
    """TUM export without timestamps uses frame indices."""
    filepath = tmp_path / "tum_no_ts.txt"
    export_tum_format(sample_poses, str(filepath))
    loaded_poses, loaded_ts = load_tum_format(str(filepath))
    expected_ts = list(range(len(sample_poses)))
    np.testing.assert_allclose(expected_ts, loaded_ts, atol=1e-6)
    for orig, rec in zip(sample_poses, loaded_poses):
        np.testing.assert_allclose(orig, rec, atol=1e-6)


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

def test_kitti_format_12_floats_per_line(sample_poses, tmp_path):
    """Every line in KITTI output has exactly 12 floats."""
    filepath = tmp_path / "kitti.txt"
    export_kitti_format(sample_poses, str(filepath))
    with open(filepath) as f:
        for i, line in enumerate(f):
            values = line.strip().split()
            assert len(values) == 12, f"Line {i}: expected 12 floats, got {len(values)}"


def test_tum_format_8_values_per_line(sample_poses, sample_timestamps, tmp_path):
    """Every line in TUM output has exactly 8 values."""
    filepath = tmp_path / "tum.txt"
    export_tum_format(sample_poses, str(filepath), timestamps=sample_timestamps)
    with open(filepath) as f:
        for i, line in enumerate(f):
            values = line.strip().split()
            assert len(values) == 8, f"Line {i}: expected 8 values, got {len(values)}"


# ---------------------------------------------------------------------------
# Identity pose round-trip
# ---------------------------------------------------------------------------

def test_identity_pose_roundtrip(tmp_path):
    """Single identity pose survives save+load unchanged."""
    identity = np.eye(4, dtype=np.float64)
    filepath = tmp_path / "identity.txt"
    export_kitti_format([identity], str(filepath))
    loaded = load_kitti_format(str(filepath))
    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded[0], identity)

    # Also test TUM
    filepath_tum = tmp_path / "identity_tum.txt"
    export_tum_format([identity], str(filepath_tum), timestamps=[0.0])
    loaded_poses, loaded_ts = load_tum_format(str(filepath_tum))
    assert len(loaded_poses) == 1
    np.testing.assert_array_equal(loaded_poses[0], identity)
    np.testing.assert_array_equal(loaded_ts, [0.0])


# ---------------------------------------------------------------------------
# TrajectoryAccumulator.save()
# ---------------------------------------------------------------------------

def test_trajectory_accumulator_save(tmp_path):
    """TrajectoryAccumulator.save() method works for both formats."""
    traj = TrajectoryAccumulator()
    # Add a few poses
    for i in range(4):
        R = np.eye(3, dtype=np.float64)
        t = np.array([float(i), 0.0, 0.0], dtype=np.float64)
        traj.add_pose(R, t)

    # Save KITTI
    kitti_path = tmp_path / "traj_kitti.txt"
    traj.save(str(kitti_path), format="kitti")
    loaded_kitti = load_kitti_format(str(kitti_path))
    np.testing.assert_allclose(traj.get_poses(), loaded_kitti, atol=1e-6)

    # Save TUM (5 poses: identity + 4 added)
    tum_path = tmp_path / "traj_tum.txt"
    traj.save(str(tum_path), format="tum", timestamps=[0.0, 1.0, 2.0, 3.0, 4.0])
    loaded_poses, loaded_ts = load_tum_format(str(tum_path))
    np.testing.assert_allclose(traj.get_poses(), loaded_poses, atol=1e-6)
    np.testing.assert_allclose([0.0, 1.0, 2.0, 3.0, 4.0], loaded_ts, atol=1e-6)

    # Invalid format
    with pytest.raises(ValueError, match="Unknown format"):
        traj.save(str(tmp_path / "bad.txt"), format="invalid")


def test_export_large_trajectory_100_poses(tmp_path):
    """Round-trip of 100 poses verifies export scales linearly (no truncation).

    Teaches: KITTI format writes one line per pose with 12 floats.
    There's no size limit — 100+ poses work identically to 5 poses.
    The file size grows linearly, making KITTI practical for trajectories
    with thousands of frames.
    """
    poses = []
    for i in range(100):
        T = np.eye(4, dtype=np.float64)
        angle = i * 0.01
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1],
        ])
        T[:3, :3] = R
        T[:3, 3] = [float(i), float(i * 0.5), 0.0]
        poses.append(T)

    kitti_path = tmp_path / "large_kitti.txt"
    export_kitti_format(poses, str(kitti_path))
    loaded = load_kitti_format(str(kitti_path))
    assert len(loaded) == 100
    for orig, rec in zip(poses, loaded):
        np.testing.assert_allclose(orig, rec, atol=1e-5)


def test_export_tum_default_timestamps_are_sequential(tmp_path):
    """TUM export without explicit timestamps uses 0, 1, 2, ... automatically.

    Teaches: callers don't need to construct a timestamp list for
    frame-indexed output — the default behavior is already sequential.
    """
    poses = [np.eye(4, dtype=np.float64) for _ in range(5)]
    filepath = tmp_path / "tum_default.txt"
    export_tum_format(poses, str(filepath))
    loaded_poses, loaded_ts = load_tum_format(str(filepath))
    assert len(loaded_poses) == 5
    np.testing.assert_allclose(loaded_ts, [0.0, 1.0, 2.0, 3.0, 4.0], atol=1e-6)
