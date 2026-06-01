"""Tests for the minislam baseline wrapper."""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.minislam_wrapper import (
    check_minislam_available,
    run_minislam_on_kitti,
    _parse_kitti_calib,
    PINNED_COMMIT,
)

MINISLAM_AVAILABLE = check_minislam_available()


@pytest.fixture
def synthetic_kitti_dir():
    """Synthetic KITTI directory with 10 small checkerboard PNGs + calib.txt."""
    import cv2

    tmp_dir = tempfile.mkdtemp(prefix="test_baseline_")
    image_dir = os.path.join(tmp_dir, "image_0")
    os.makedirs(image_dir)

    np.random.seed(42)
    for i in range(10):
        img = np.zeros((128, 256), dtype=np.uint8)
        for r in range(0, 128, 16):
            for c in range(0, 256, 16):
                if (r // 16 + c // 16 + i) % 2 == 0:
                    img[r : r + 16, c : c + 16] = 255
        img = np.roll(img, i * 3, axis=1)
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        cv2.imwrite(os.path.join(image_dir, f"{i:06d}.png"), img)

    calib_content = (
        "P0: 7.188560e+02 0.000000e+00 6.072000e+02 0.000000e+00 "
        "0.000000e+00 7.188560e+02 1.852000e+02 0.000000e+00 "
        "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n"
        "P1: 7.188560e+02 0.000000e+02 6.072000e+02 -3.866000e+02 "
        "0.000000e+00 7.188560e+02 1.852000e+02 0.000000e+00 "
        "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n"
    )
    with open(os.path.join(tmp_dir, "calib.txt"), "w") as f:
        f.write(calib_content)

    with open(os.path.join(tmp_dir, "poses.txt"), "w") as f:
        for _ in range(10):
            f.write("1 0 0 0 0 1 0 0 0 0 1 0\n")

    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestCheckMinislamAvailable:
    def test_returns_bool(self):
        result = check_minislam_available()
        assert isinstance(result, bool)

    def test_true_when_installed(self):
        if not MINISLAM_AVAILABLE:
            pytest.skip("minislam not installed")
        assert check_minislam_available() is True

    def test_false_when_import_fails(self):
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("minislam"):
                raise ImportError(f"Mocked: no module named {name!r}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            assert check_minislam_available() is False


class TestParseKittiCalib:
    def test_parse_standard_calib(self, synthetic_kitti_dir):
        result = _parse_kitti_calib(os.path.join(synthetic_kitti_dir, "calib.txt"))
        assert abs(result["fx"] - 718.856) < 0.01
        assert abs(result["fy"] - 718.856) < 0.01
        assert abs(result["cx"] - 607.2) < 0.1
        assert abs(result["cy"] - 185.2) < 0.1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _parse_kitti_calib("/nonexistent/calib.txt")

    def test_missing_p0_raises(self, tmp_path):
        bad_calib = tmp_path / "calib.txt"
        bad_calib.write_text("P1: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        with pytest.raises(ValueError, match="P0 not found"):
            _parse_kitti_calib(str(bad_calib))


@pytest.mark.baseline
@pytest.mark.skipif(not MINISLAM_AVAILABLE, reason="minislam not installed")
class TestRunMinislamOnKitti:
    def test_produces_poses(self, synthetic_kitti_dir, tmp_path):
        output_dir = str(tmp_path / "output")
        poses = run_minislam_on_kitti(
            data_dir=synthetic_kitti_dir,
            output_dir=output_dir,
            use_calib_intrinsics=True,
        )
        assert isinstance(poses, list)
        for pose in poses:
            assert isinstance(pose, np.ndarray)
            assert pose.shape == (4, 4)

    def test_writes_trajectory_file(self, synthetic_kitti_dir, tmp_path):
        output_dir = str(tmp_path / "output")
        poses = run_minislam_on_kitti(
            data_dir=synthetic_kitti_dir,
            output_dir=output_dir,
            use_calib_intrinsics=True,
        )
        traj_file = os.path.join(output_dir, "minislam_trajectory.txt")
        assert os.path.isfile(traj_file)

        if poses:
            with open(traj_file) as f:
                lines = f.readlines()
            assert len(lines) == len(poses)
            for line in lines:
                assert len(line.strip().split()) == 12

    def test_use_calib_true(self, synthetic_kitti_dir, tmp_path):
        output_dir = str(tmp_path / "output_calib")
        poses = run_minislam_on_kitti(
            data_dir=synthetic_kitti_dir,
            output_dir=output_dir,
            use_calib_intrinsics=True,
        )
        assert isinstance(poses, list)
        assert os.path.isfile(os.path.join(output_dir, "minislam_trajectory.txt"))

    def test_use_calib_false(self, synthetic_kitti_dir, tmp_path):
        output_dir = str(tmp_path / "output_fov")
        poses = run_minislam_on_kitti(
            data_dir=synthetic_kitti_dir,
            output_dir=output_dir,
            use_calib_intrinsics=False,
        )
        assert isinstance(poses, list)
        assert os.path.isfile(os.path.join(output_dir, "minislam_trajectory.txt"))

    def test_max_frames_limiting(self, synthetic_kitti_dir, tmp_path):
        output_dir = str(tmp_path / "output_limited")
        poses = run_minislam_on_kitti(
            data_dir=synthetic_kitti_dir,
            output_dir=output_dir,
            use_calib_intrinsics=True,
            max_frames=3,
        )
        assert isinstance(poses, list)
        assert len(poses) <= 2

    def test_invalid_data_dir_returns_empty(self, tmp_path):
        poses = run_minislam_on_kitti(
            data_dir="/nonexistent/path",
            output_dir=str(tmp_path / "output"),
        )
        assert poses == []

    def test_no_image_dir_returns_empty(self, tmp_path):
        data_dir = str(tmp_path / "empty_data")
        os.makedirs(data_dir)
        poses = run_minislam_on_kitti(
            data_dir=data_dir,
            output_dir=str(tmp_path / "output"),
        )
        assert poses == []


class TestGracefulFailure:
    def test_run_returns_empty_when_unavailable(self, tmp_path):
        if MINISLAM_AVAILABLE:
            saved = {}
            for key in list(sys.modules.keys()):
                if key.startswith("minislam"):
                    saved[key] = sys.modules.pop(key)

            original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

            def mock_import(name, *args, **kwargs):
                if name.startswith("minislam"):
                    raise ImportError(f"Mocked: {name}")
                return original_import(name, *args, **kwargs)

            try:
                with patch("builtins.__import__", side_effect=mock_import):
                    result = run_minislam_on_kitti(
                        data_dir=str(tmp_path),
                        output_dir=str(tmp_path / "out"),
                    )
                    assert result == []
            finally:
                sys.modules.update(saved)
        else:
            result = run_minislam_on_kitti(
                data_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
            )
            assert result == []


class TestPinnedCommit:
    def test_pinned_commit_constant(self):
        assert PINNED_COMMIT == "962096d5bb8919317cceef9c0f2f98f023d9fcf3"
