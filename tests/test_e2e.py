"""End-to-end integration test for the visual odometry pipeline."""

import numpy as np
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "kitti_05_subset"
SCRIPT = Path(__file__).parent.parent / "run_vo.py"


@pytest.fixture(scope="module", autouse=True)
def vo_output(tmp_path_factory):
    """Run run_vo.py once and provide its output directory to all tests."""
    if not FIXTURE_DIR.exists():
        pytest.skip(f"Fixture directory not found: {FIXTURE_DIR}")

    out_dir = tmp_path_factory.mktemp("vo_e2e")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input", str(FIXTURE_DIR),
            "--output", str(out_dir),
            "--fov", "63",
            "--matcher", "lightglue",
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"run_vo.py failed with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    return out_dir


class TestPipelineExecution:
    """Verify the pipeline runs successfully and produces output."""

    def test_script_exists(self):
        assert SCRIPT.exists(), f"run_vo.py not found at {SCRIPT}"

    def test_kitti_trajectory_exists(self, vo_output):
        path = vo_output / "trajectory_kitti.txt"
        assert path.exists(), "trajectory_kitti.txt not created"
        assert path.stat().st_size > 0, "trajectory_kitti.txt is empty"

    def test_tum_trajectory_exists(self, vo_output):
        path = vo_output / "trajectory_tum.txt"
        assert path.exists(), "trajectory_tum.txt not created"
        assert path.stat().st_size > 0, "trajectory_tum.txt is empty"

    def test_plot_exists(self, vo_output):
        path = vo_output / "trajectory_plot.png"
        assert path.exists(), "trajectory_plot.png not created"
        assert path.stat().st_size > 0, "trajectory_plot.png is empty"


class TestKittiFormat:
    """Validate KITTI trajectory file format."""

    def test_minimum_poses(self, vo_output):
        lines = (vo_output / "trajectory_kitti.txt").read_text().strip().splitlines()
        n_poses = len(lines)
        assert n_poses >= 30, f"Expected >=30 poses, got {n_poses}"

    def test_each_line_has_12_floats(self, vo_output):
        lines = (vo_output / "trajectory_kitti.txt").read_text().strip().splitlines()
        for i, line in enumerate(lines):
            vals = line.strip().split()
            assert len(vals) == 12, f"Line {i}: expected 12 floats, got {len(vals)}"
            for j, v in enumerate(vals):
                float(v)

    def test_first_pose_is_identity(self, vo_output):
        lines = (vo_output / "trajectory_kitti.txt").read_text().strip().splitlines()
        vals = [float(x) for x in lines[0].split()]
        T = np.array(vals).reshape(3, 4)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-5)
        np.testing.assert_allclose(T[:3, 3], np.zeros(3), atol=1e-5)

    def test_nonzero_translation(self, vo_output):
        lines = (vo_output / "trajectory_kitti.txt").read_text().strip().splitlines()
        if len(lines) < 2:
            pytest.skip("Not enough poses to check")
        vals = [float(x) for x in lines[-1].split()]
        t = np.array(vals[3::4])
        assert np.linalg.norm(t) > 0.01, "Final translation is near-zero"


class TestTumFormat:
    """Validate TUM trajectory file format."""

    def test_line_count_matches_kitti(self, vo_output):
        kitti_lines = (vo_output / "trajectory_kitti.txt").read_text().strip().splitlines()
        tum_lines = (vo_output / "trajectory_tum.txt").read_text().strip().splitlines()
        assert len(tum_lines) == len(kitti_lines), (
            f"TUM lines ({len(tum_lines)}) != KITTI lines ({len(kitti_lines)})"
        )

    def test_each_line_has_8_floats(self, vo_output):
        lines = (vo_output / "trajectory_tum.txt").read_text().strip().splitlines()
        for i, line in enumerate(lines):
            vals = line.strip().split()
            assert len(vals) == 8, f"Line {i}: expected 8 floats, got {len(vals)}"
            for v in vals:
                float(v)

    def test_quaternions_normalized(self, vo_output):
        lines = (vo_output / "trajectory_tum.txt").read_text().strip().splitlines()
        for i, line in enumerate(lines):
            vals = [float(x) for x in line.split()]
            qx, qy, qz, qw = vals[4], vals[5], vals[6], vals[7]
            qnorm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            assert abs(qnorm - 1.0) < 0.01, (
                f"Line {i}: quaternion norm {qnorm:.4f} != 1.0"
            )

    def test_timestamps_monotonic(self, vo_output):
        lines = (vo_output / "trajectory_tum.txt").read_text().strip().splitlines()
        timestamps = [float(line.split()[0]) for line in lines]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Timestamps not monotonic at index {i}: "
                f"{timestamps[i-1]} > {timestamps[i]}"
            )


class TestExports:
    """Verify __init__.py exports work correctly."""

    def test_import_all_components(self):
        from slam_dnn import (
            SuperPointExtractor,
            LightGlueMatcher,
            ClassicMatcher,
            estimate_essential,
            TrackingLostError,
            TrajectoryAccumulator,
            K_from_fov,
        )
        assert SuperPointExtractor is not None
        assert LightGlueMatcher is not None
        assert ClassicMatcher is not None
        assert estimate_essential is not None
        assert TrackingLostError is not None
        assert TrajectoryAccumulator is not None
        assert K_from_fov is not None
