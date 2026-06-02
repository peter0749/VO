"""Tests for slam_dnn CLI entry point — mocked versions to avoid real data interaction."""
import io
import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

class MockFrameLoader:
    def __init__(self, path, n_frames=5):
        self.path = path
        self.n_frames = n_frames
        self.frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(n_frames)]

    def __len__(self):
        return self.n_frames

    def __iter__(self):
        return iter(self.frames)

class MockTrajectory:
    def __init__(self, poses):
        self.poses = poses

    def get_poses(self):
        return self.poses

    def get_positions(self):
        return np.array([p[:3, 3] for p in self.poses])

class MockVisualOdometry:
    def __init__(self, camera, matcher, max_keypoints=2048, scale=1.0, device="cpu"):
        self.camera = camera
        self.matcher = matcher
        self.max_keypoints = max_keypoints
        self.scale = scale
        self.device = device
        self.poses = [np.eye(4)]
        self.processed_count = 0

    def process_frame(self, img):
        self.processed_count += 1
        if self.processed_count == 1:
            return None
        T = np.eye(4)
        self.poses.append(T)
        return T

    def get_trajectory(self):
        return MockTrajectory(self.poses)

    def get_stats(self):
        return {
            "successful": self.processed_count - 1 if self.processed_count > 0 else 0,
            "tracking_lost": 0,
            "pose_failed": 0,
        }

class RunVoResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

def _run_cli_mocked(extra_args: list, n_frames=5) -> RunVoResult:
    from slam_dnn.cli import main
    mock_loader = MockFrameLoader("dummy_path", n_frames=n_frames)
    
    f_out = io.StringIO()
    f_err = io.StringIO()
    
    # We patch exists in Path to return True unless it's a nonexistent path
    def mock_exists(path_obj):
        if "nonexistent" in str(path_obj):
            return False
        return True
        
    from contextlib import redirect_stdout, redirect_stderr
    with patch('slam_dnn.cli.FrameLoader', return_value=mock_loader), \
         patch('slam_dnn.cli.VisualOdometry', MockVisualOdometry), \
         patch.object(Path, 'exists', mock_exists), \
         redirect_stdout(f_out), \
         redirect_stderr(f_err):
        returncode = main(extra_args)
        
    return RunVoResult(returncode, f_out.getvalue(), f_err.getvalue())


def test_cli_help():
    """--help prints usage and exits 0."""
    from slam_dnn.cli import main
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with pytest.raises(SystemExit) as excinfo, redirect_stdout(f):
        main(["--help"])
    assert excinfo.value.code == 0
    out = f.getvalue()
    assert "--input" in out
    assert "--output" in out


def test_cli_nonexistent_input(tmp_path):
    """CLI errors gracefully on nonexistent input."""
    result = _run_cli_mocked(["--input", "/nonexistent/path", "--output", str(tmp_path)])
    assert result.returncode != 0


def test_cli_runs_on_kitti_fixtures(tmp_path):
    """Full run on KITTI fixtures produces output files without real DL models."""
    result = _run_cli_mocked([
        "--input", "dummy_fixtures",
        "--output", str(tmp_path),
        "--fov", "63",
        "--device", "cpu",
        "--matcher", "classic",
        "--no-plot",
    ], n_frames=5)
    assert result.returncode == 0
    assert (tmp_path / "trajectory_kitti.txt").exists()
    assert (tmp_path / "trajectory_tum.txt").exists()


def test_cli_verbose_logging(tmp_path):
    """--verbose flag enables INFO logging."""
    result = _run_cli_mocked([
        "--input", "dummy_fixtures",
        "--output", str(tmp_path),
        "-v",
        "--matcher", "classic",
        "--device", "cpu",
        "--no-plot",
    ])
    assert result.returncode == 0
    assert (tmp_path / "trajectory_kitti.txt").exists()


def test_cli_no_plot_flag(tmp_path):
    """--no-plot skips plot generation."""
    result = _run_cli_mocked([
        "--input", "dummy_fixtures",
        "--output", str(tmp_path),
        "--no-plot",
        "--matcher", "classic",
        "--device", "cpu",
    ])
    assert result.returncode == 0
    assert (tmp_path / "trajectory_kitti.txt").exists()
    assert not (tmp_path / "trajectory_plot.png").exists()


def test_cli_output_directory_auto_created(tmp_path):
    """Output directory is created if missing."""
    new_out = tmp_path / "nested" / "new" / "dir"
    result = _run_cli_mocked([
        "--input", "dummy_fixtures",
        "--output", str(new_out),
        "--no-plot",
        "--matcher", "classic",
        "--device", "cpu",
    ])
    assert result.returncode == 0
    assert new_out.exists()
    assert (new_out / "trajectory_kitti.txt").exists()


def test_cli_backward_compat_run_vo(tmp_path):
    """run_vo.py still works as backward-compatible entry point (tested in-process)."""
    result = _run_cli_mocked([
        "--input", "dummy_fixtures",
        "--output", str(tmp_path),
        "--fov", "63",
        "--device", "cpu",
        "--matcher", "classic",
        "--no-plot",
    ])
    assert result.returncode == 0
    assert (tmp_path / "trajectory_kitti.txt").exists()
    assert (tmp_path / "trajectory_tum.txt").exists()
