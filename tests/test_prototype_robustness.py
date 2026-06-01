"""Prototype robustness / stress tests for run_vo.py.

Each test invokes the VO pipeline as a subprocess to exercise real-world
failure modes and configuration variations.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "kitti_05_subset"
SCRIPT = Path(__file__).parent.parent / "run_vo.py"


def _run_vo(extra_args: list, timeout: int = 300) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT)] + extra_args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def fixture_available():
    if not FIXTURE_DIR.exists():
        pytest.skip(f"Fixture directory not found: {FIXTURE_DIR}")


@pytest.fixture
def tmp_output(tmp_path_factory):
    return tmp_path_factory.mktemp("vo_proto")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunVoClassicMatcher:

    def test_run_vo_with_classic_matcher(self, fixture_available, tmp_output):
        result = _run_vo([
            "--input", str(FIXTURE_DIR),
            "--output", str(tmp_output),
            "--matcher", "classic",
            "--device", "cpu",
        ])

        assert result.returncode == 0, f"Failed:\n{result.stderr}\n{result.stdout}"
        assert (tmp_output / "trajectory_kitti.txt").exists()
        assert (tmp_output / "trajectory_tum.txt").exists()

        lines = (tmp_output / "trajectory_kitti.txt").read_text().strip().splitlines()
        assert len(lines) >= 30, f"Classic matcher produced only {len(lines)} poses"


class TestRunVoFovVariation:

    def test_run_vo_with_fov_50(self, fixture_available, tmp_output):
        out = tmp_output / "fov50"
        result = _run_vo([
            "--input", str(FIXTURE_DIR),
            "--output", str(out),
            "--fov", "50",
            "--device", "cpu",
        ])
        assert result.returncode == 0, f"FOV=50 failed:\n{result.stderr}"
        assert (out / "trajectory_kitti.txt").exists()

    def test_run_vo_with_fov_80(self, fixture_available, tmp_output):
        out = tmp_output / "fov80"
        result = _run_vo([
            "--input", str(FIXTURE_DIR),
            "--output", str(out),
            "--fov", "80",
            "--device", "cpu",
        ])
        assert result.returncode == 0, f"FOV=80 failed:\n{result.stderr}"
        assert (out / "trajectory_kitti.txt").exists()


class TestRunVoScaleVariation:

    def test_run_vo_scale_changes_translation(self, fixture_available, tmp_output):
        out_small = tmp_output / "scale_small"
        out_large = tmp_output / "scale_large"

        r_small = _run_vo([
            "--input", str(FIXTURE_DIR),
            "--output", str(out_small),
            "--scale", "0.5",
            "--device", "cpu",
        ])
        r_large = _run_vo([
            "--input", str(FIXTURE_DIR),
            "--output", str(out_large),
            "--scale", "2.0",
            "--device", "cpu",
        ])

        assert r_small.returncode == 0, f"scale=0.5 failed:\n{r_small.stderr}"
        assert r_large.returncode == 0, f"scale=2.0 failed:\n{r_large.stderr}"

        def _final_tnorm(tum_path: Path) -> float:
            lines = tum_path.read_text().strip().splitlines()
            if not lines:
                return 0.0
            vals = lines[-1].split()
            tx, ty, tz = float(vals[1]), float(vals[2]), float(vals[3])
            return float(np.linalg.norm([tx, ty, tz]))

        norm_small = _final_tnorm(out_small / "trajectory_tum.txt")
        norm_large = _final_tnorm(out_large / "trajectory_tum.txt")
        assert norm_large > norm_small, (
            f"scale=2.0 norm ({norm_large:.3f}) should exceed scale=0.5 norm ({norm_small:.3f})"
        )


class TestRunVoCpuDevice:

    def test_run_vo_with_explicit_cpu(self, fixture_available, tmp_output):
        result = _run_vo([
            "--input", str(FIXTURE_DIR),
            "--output", str(tmp_output),
            "--device", "cpu",
        ])
        assert result.returncode == 0, f"CPU run failed:\n{result.stderr}"
        assert (tmp_output / "trajectory_kitti.txt").exists()
        assert (tmp_output / "trajectory_tum.txt").exists()


class TestRunVoErrorHandling:

    def test_run_vo_handles_missing_directory(self, tmp_output):
        result = _run_vo([
            "--input", "/nonexistent/path/that/should/not/exist",
            "--output", str(tmp_output),
            "--device", "cpu",
        ])
        assert result.returncode != 0

    def test_run_vo_handles_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out_dir = tmp_path / "out"

        result = _run_vo([
            "--input", str(empty_dir),
            "--output", str(out_dir),
            "--device", "cpu",
        ])
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "no images" in combined.lower() or "error" in combined.lower()


class TestRunVoAggressiveKeypoints:

    def test_run_vo_with_max_keypoints_10(self, fixture_available, tmp_output):
        result = _run_vo([
            "--input", str(FIXTURE_DIR),
            "--output", str(tmp_output),
            "--max-keypoints", "10",
            "--device", "cpu",
        ])
        combined = result.stdout + result.stderr
        if result.returncode == 0:
            assert (tmp_output / "trajectory_kitti.txt").exists()
        else:
            assert any(
                tok in combined.lower()
                for tok in ["match", "keypoint", "error", "fail", "lost"]
            )


class TestRunVoHelpFlag:

    def test_run_vo_help_flag(self):
        result = _run_vo(["--help"])
        assert result.returncode == 0
        out = result.stdout
        assert "--input" in out
        assert "--matcher" in out
        assert "--fov" in out
        assert "--device" in out
        assert "--scale" in out
        assert "--ground-truth" in out
        assert "--no-plot" in out


class TestRunVoOutputDirectoryCreation:

    def test_run_vo_output_directory_created(self, fixture_available, tmp_path):
        nested = tmp_path / "deeply" / "nested" / "new" / "dir"
        assert not nested.exists()

        result = _run_vo([
            "--input", str(FIXTURE_DIR),
            "--output", str(nested),
            "--device", "cpu",
        ])

        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert nested.exists()
        assert (nested / "trajectory_kitti.txt").exists()
        assert (nested / "trajectory_tum.txt").exists()


class TestRunVoNoPlot:

    def test_run_vo_no_plot_skips_plot_file(self, fixture_available, tmp_output):
        result = _run_vo([
            "--input", str(FIXTURE_DIR),
            "--output", str(tmp_output),
            "--device", "cpu",
            "--no-plot",
        ])
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert (tmp_output / "trajectory_kitti.txt").exists()
        assert (tmp_output / "trajectory_tum.txt").exists()
        assert not (tmp_output / "trajectory_plot.png").exists()
