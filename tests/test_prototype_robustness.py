"""Prototype robustness / stress tests for run_vo.py — mocked versions to avoid real DL models and subprocesses."""
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
        T[0, 3] = 0.1 * (self.processed_count - 1) * self.scale
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

def _run_vo_mocked(extra_args: list, n_frames=35, frame_loader_mock=None) -> RunVoResult:
    from slam_dnn.cli import main
    if frame_loader_mock is None:
        mock_loader = MockFrameLoader("dummy_path", n_frames=n_frames)
        loader_patch = patch('slam_dnn.cli.FrameLoader', return_value=mock_loader)
    else:
        loader_patch = patch('slam_dnn.cli.FrameLoader', frame_loader_mock)
    
    f_out = io.StringIO()
    f_err = io.StringIO()
    
    # We patch exists in Path to return True unless it's a nonexistent path
    def mock_exists(path_obj):
        if "nonexistent" in str(path_obj):
            return False
        return True
        
    from contextlib import redirect_stdout, redirect_stderr
    with loader_patch, \
         patch('slam_dnn.cli.VisualOdometry', MockVisualOdometry), \
         patch.object(Path, 'exists', mock_exists), \
         redirect_stdout(f_out), \
         redirect_stderr(f_err):
        returncode = main(extra_args)
        
    return RunVoResult(returncode, f_out.getvalue(), f_err.getvalue())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunVoClassicMatcher:
    def test_run_vo_with_classic_matcher(self, tmp_path):
        tmp_output = tmp_path / "classic"
        result = _run_vo_mocked([
            "--input", "dummy_fixtures",
            "--output", str(tmp_output),
            "--matcher", "classic",
            "--device", "cpu",
        ], n_frames=35)

        assert result.returncode == 0
        assert (tmp_output / "trajectory_kitti.txt").exists()
        assert (tmp_output / "trajectory_tum.txt").exists()

        lines = (tmp_output / "trajectory_kitti.txt").read_text().strip().splitlines()
        assert len(lines) == 35


class TestRunVoFovVariation:
    def test_run_vo_with_fov_50(self, tmp_path):
        out = tmp_path / "fov50"
        result = _run_vo_mocked([
            "--input", "dummy_fixtures",
            "--output", str(out),
            "--fov", "50",
            "--device", "cpu",
        ])
        assert result.returncode == 0
        assert (out / "trajectory_kitti.txt").exists()

    def test_run_vo_with_fov_80(self, tmp_path):
        out = tmp_path / "fov80"
        result = _run_vo_mocked([
            "--input", "dummy_fixtures",
            "--output", str(out),
            "--fov", "80",
            "--device", "cpu",
        ])
        assert result.returncode == 0
        assert (out / "trajectory_kitti.txt").exists()


class TestRunVoScaleVariation:
    def test_run_vo_scale_changes_translation(self, tmp_path):
        out_small = tmp_path / "scale_small"
        out_large = tmp_path / "scale_large"

        r_small = _run_vo_mocked([
            "--input", "dummy_fixtures",
            "--output", str(out_small),
            "--scale", "0.5",
            "--device", "cpu",
        ])
        r_large = _run_vo_mocked([
            "--input", "dummy_fixtures",
            "--output", str(out_large),
            "--scale", "2.0",
            "--device", "cpu",
        ])

        assert r_small.returncode == 0
        assert r_large.returncode == 0

        def _final_tnorm(tum_path: Path) -> float:
            lines = tum_path.read_text().strip().splitlines()
            if not lines:
                return 0.0
            vals = lines[-1].split()
            tx, ty, tz = float(vals[1]), float(vals[2]), float(vals[3])
            return float(np.linalg.norm([tx, ty, tz]))

        norm_small = _final_tnorm(out_small / "trajectory_tum.txt")
        norm_large = _final_tnorm(out_large / "trajectory_tum.txt")
        assert norm_large > norm_small


class TestRunVoCpuDevice:
    def test_run_vo_with_explicit_cpu(self, tmp_path):
        tmp_output = tmp_path / "cpu"
        result = _run_vo_mocked([
            "--input", "dummy_fixtures",
            "--output", str(tmp_output),
            "--device", "cpu",
        ])
        assert result.returncode == 0
        assert (tmp_output / "trajectory_kitti.txt").exists()
        assert (tmp_output / "trajectory_tum.txt").exists()


class TestRunVoErrorHandling:
    def test_run_vo_handles_missing_directory(self, tmp_path):
        tmp_output = tmp_path / "missing"
        result = _run_vo_mocked([
            "--input", "/nonexistent/path/that/should/not/exist",
            "--output", str(tmp_output),
            "--device", "cpu",
        ])
        assert result.returncode != 0

    def test_run_vo_handles_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out_dir = tmp_path / "out"

        def mock_raise_loader(*args, **kwargs):
            raise ValueError("No images found")

        result = _run_vo_mocked([
            "--input", str(empty_dir),
            "--output", str(out_dir),
            "--device", "cpu",
        ], frame_loader_mock=mock_raise_loader)
        assert result.returncode != 0


class TestRunVoAggressiveKeypoints:
    def test_run_vo_with_max_keypoints_10(self, tmp_path):
        tmp_output = tmp_path / "kp10"
        result = _run_vo_mocked([
            "--input", "dummy_fixtures",
            "--output", str(tmp_output),
            "--max-keypoints", "10",
            "--device", "cpu",
        ])
        assert result.returncode == 0
        assert (tmp_output / "trajectory_kitti.txt").exists()


class TestRunVoHelpFlag:
    def test_run_vo_help_flag(self):
        from slam_dnn.cli import main
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with pytest.raises(SystemExit) as excinfo, redirect_stdout(f):
            main(["--help"])
        assert excinfo.value.code == 0
        out = f.getvalue()
        assert "--input" in out
        assert "--matcher" in out
        assert "--fov" in out
        assert "--device" in out
        assert "--scale" in out
        assert "--ground-truth" in out
        assert "--no-plot" in out


class TestRunVoOutputDirectoryCreation:
    def test_run_vo_output_directory_created(self, tmp_path):
        nested = tmp_path / "deeply" / "nested" / "new" / "dir"
        assert not nested.exists()

        result = _run_vo_mocked([
            "--input", "dummy_fixtures",
            "--output", str(nested),
            "--device", "cpu",
        ])

        assert result.returncode == 0
        assert nested.exists()
        assert (nested / "trajectory_kitti.txt").exists()
        assert (nested / "trajectory_tum.txt").exists()


class TestRunVoNoPlot:
    def test_run_vo_no_plot_skips_plot_file(self, tmp_path):
        tmp_output = tmp_path / "noplot"
        result = _run_vo_mocked([
            "--input", "dummy_fixtures",
            "--output", str(tmp_output),
            "--device", "cpu",
            "--no-plot",
        ])
        assert result.returncode == 0
        assert (tmp_output / "trajectory_kitti.txt").exists()
        assert (tmp_output / "trajectory_tum.txt").exists()
        assert not (tmp_output / "trajectory_plot.png").exists()
