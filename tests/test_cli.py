"""Tests for slam_dnn CLI entry point."""
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.fixture
def cli_cmd():
    """Base command to invoke CLI."""
    return [sys.executable, "-m", "slam_dnn"]


def test_cli_help(cli_cmd, capsys):
    """--help prints usage and exits 0."""
    result = subprocess.run(cli_cmd + ["--help"], capture_output=True, text=True)
    assert result.returncode == 0, f"Unexpected output: {result.stderr}"
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--matcher" in result.stdout
    assert "--device" in result.stdout
    assert "--verbose" in result.stdout


def test_cli_nonexistent_input(cli_cmd, tmp_path):
    """CLI errors gracefully on nonexistent input."""
    result = subprocess.run(
        cli_cmd + ["--input", "/nonexistent/path", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_runs_on_kitti_fixtures(cli_cmd, tmp_path):
    """Full run on KITTI fixtures produces output files."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "kitti_05_subset"
    if not fixtures_dir.exists():
        pytest.skip("KITTI fixtures not available")

    result = subprocess.run(
        cli_cmd + [
            "--input", str(fixtures_dir),
            "--output", str(tmp_path),
            "--fov", "63",
            "--device", "cpu",
            "--matcher", "classic",  # Faster than LightGlue
            "--no-plot",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert (tmp_path / "trajectory_kitti.txt").exists()
    assert (tmp_path / "trajectory_tum.txt").exists()


def test_cli_verbose_logging(cli_cmd, tmp_path):
    """--verbose flag enables INFO logging."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "kitti_05_subset"
    if not fixtures_dir.exists():
        pytest.skip("KITTI fixtures not available")

    result = subprocess.run(
        cli_cmd + [
            "--input", str(fixtures_dir),
            "--output", str(tmp_path),
            "-v",
            "--matcher", "classic",
            "--device", "cpu",
            "--no-plot",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert (tmp_path / "trajectory_kitti.txt").exists()


def test_cli_no_plot_flag(cli_cmd, tmp_path):
    """--no-plot skips plot generation."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "kitti_05_subset"
    if not fixtures_dir.exists():
        pytest.skip("KITTI fixtures not available")

    result = subprocess.run(
        cli_cmd + [
            "--input", str(fixtures_dir),
            "--output", str(tmp_path),
            "--no-plot",
            "--matcher", "classic",
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert (tmp_path / "trajectory_kitti.txt").exists()
    assert not (tmp_path / "trajectory_plot.png").exists()


def test_cli_output_directory_auto_created(cli_cmd, tmp_path):
    """Output directory is created if missing."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "kitti_05_subset"
    if not fixtures_dir.exists():
        pytest.skip("KITTI fixtures not available")

    new_out = tmp_path / "nested" / "new" / "dir"
    result = subprocess.run(
        cli_cmd + [
            "--input", str(fixtures_dir),
            "--output", str(new_out),
            "--no-plot",
            "--matcher", "classic",
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert new_out.exists()
    assert (new_out / "trajectory_kitti.txt").exists()


def test_cli_backward_compat_run_vo(cli_cmd, tmp_path):
    """run_vo.py still works as backward-compatible entry point."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "kitti_05_subset"
    if not fixtures_dir.exists():
        pytest.skip("KITTI fixtures not available")

    result = subprocess.run(
        [sys.executable, "run_vo.py",
         "--input", str(fixtures_dir),
         "--output", str(tmp_path),
         "--fov", "63",
         "--device", "cpu",
         "--matcher", "classic",
         "--no-plot",
         ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"run_vo.py failed:\n{result.stderr}"
    assert (tmp_path / "trajectory_kitti.txt").exists()
    assert (tmp_path / "trajectory_tum.txt").exists()
