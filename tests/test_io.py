"""Tests for slam_dnn/io — FrameLoader and helper functions."""

import cv2
import numpy as np
import pytest
import tempfile
from pathlib import Path

from slam_dnn.io import FrameLoader, to_grayscale, to_float


def _make_image(size: tuple[int, int] = (64, 48), color: tuple[int, int, int] = (255, 128, 0)):
    """Create a small BGR test image."""
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)


def _make_png_path(tmpdir: Path, name: str, size: tuple[int, int] = (64, 48), color: tuple[int, int, int] = (255, 128, 0)) -> Path:
    """Create a PNG file and return its path."""
    img = _make_image(size, color)
    path = tmpdir / name
    cv2.imwrite(str(path), img)
    return path


class TestFrameLoaderImageDir:
    """FrameLoader with image directory source."""

    def test_image_dir_loading(self, tmp_path: Path):
        """Loading from an image directory yields all images."""
        for i in range(5):
            _make_png_path(tmp_path, f"frame_{i:03d}.png", color=(255, i * 50, 0))
        loader = FrameLoader(str(tmp_path))
        frames = list(loader)
        assert len(frames) == 5
        for i, frame in enumerate(frames):
            assert frame.shape == (48, 64, 3)
            assert frame[0, 0, 1] == i * 50  # green channel

    def test_image_dir_sorted_by_filename(self, tmp_path: Path):
        """Images are yielded in sorted filename order."""
        _make_png_path(tmp_path, "z.png", color=(255, 0, 0))
        _make_png_path(tmp_path, "a.png", color=(0, 255, 0))
        _make_png_path(tmp_path, "m.png", color=(0, 0, 255))
        loader = FrameLoader(str(tmp_path))
        frames = list(loader)
        assert len(frames) == 3
        assert frames[0][0, 0, 1] == 255  # a.png
        assert frames[1][0, 0, 2] == 255  # m.png
        assert frames[2][0, 0, 0] == 255  # z.png

    def test_image_dir_len(self, tmp_path: Path):
        """__len__ returns correct count for image directory."""
        for i in range(7):
            _make_png_path(tmp_path, f"img_{i:02d}.png")
        loader = FrameLoader(str(tmp_path))
        assert len(loader) == 7

    def test_nonexistent_path_raises(self):
        """Loading from a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FrameLoader("/nonexistent/path/to/images")

    def test_nonexistent_video_raises(self):
        """Loading a non-existent video file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FrameLoader("/nonexistent/video.mp4")

    def test_empty_directory_raises(self, tmp_path: Path):
        """Empty directory raises ValueError."""
        with pytest.raises(ValueError, match="No image files"):
            FrameLoader(str(tmp_path))

    def test_max_frames_limits_output(self, tmp_path: Path):
        """max_frames limits the number of yielded frames."""
        for i in range(5):
            _make_png_path(tmp_path, f"frame_{i:03d}.png")
        loader = FrameLoader(str(tmp_path), max_frames=2)
        frames = list(loader)
        assert len(frames) == 2
        assert len(loader) == 2  # len also respects max_frames

    def test_resize_changes_dimensions(self, tmp_path: Path):
        """resize=(100, 100) produces (100, 100, 3) frames."""
        for i in range(3):
            _make_png_path(tmp_path, f"frame_{i:03d}.png", size=(64, 48))
        loader = FrameLoader(str(tmp_path), resize=(100, 100))
        frames = list(loader)
        assert len(frames) == 3
        for frame in frames:
            assert frame.shape == (100, 100, 3)

    def test_resize_video(self, tmp_path: Path):
        """resize works with video source too."""
        video_path = tmp_path / "test.mp4"
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for _ in range(5):
            video.write(_make_image((64, 48)))
        video.release()
        loader = FrameLoader(str(video_path), resize=(100, 100))
        frames = list(loader)
        assert len(frames) == 5
        for frame in frames:
            assert frame.shape == (100, 100, 3)

    def test_video_file_loading(self, tmp_path: Path):
        """Loading from a video file yields all frames."""
        video_path = tmp_path / "test.mp4"
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for i in range(5):
            video.write(_make_image((64, 48), color=(255, i * 50, 0)))
        video.release()
        loader = FrameLoader(str(video_path))
        frames = list(loader)
        assert len(frames) == 5
        for frame in frames:
            assert frame.shape == (48, 64, 3)

    def test_video_len(self, tmp_path: Path):
        """__len__ returns correct frame count for video."""
        video_path = tmp_path / "test.mp4"
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for _ in range(8):
            video.write(_make_image((64, 48)))
        video.release()
        loader = FrameLoader(str(video_path))
        assert len(loader) == 8

    def test_video_max_frames(self, tmp_path: Path):
        """max_frames limits video frame output."""
        video_path = tmp_path / "test.mp4"
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for _ in range(10):
            video.write(_make_image((64, 48)))
        video.release()
        loader = FrameLoader(str(video_path), max_frames=3)
        frames = list(loader)
        assert len(frames) == 3
        assert len(loader) == 3

    def test_video_nonexistent_raises(self):
        """Loading a non-existent video file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FrameLoader("/nonexistent/video.mp4")

    def test_unsupported_extension_ignored(self, tmp_path: Path):
        """Non-image files in directory are ignored."""
        _make_png_path(tmp_path, "good.png")
        # Create a non-image file
        (tmp_path / "bad.txt").write_text("hello")
        loader = FrameLoader(str(tmp_path))
        frames = list(loader)
        assert len(frames) == 1

    def test_multiple_image_extensions(self, tmp_path: Path):
        """Different image extensions are all loaded."""
        _make_png_path(tmp_path, "a.png")
        # cv2 can write JPEG
        img = _make_image()
        cv2.imwrite(str(tmp_path / "b.jpg"), img)
        cv2.imwrite(str(tmp_path / "c.jpeg"), img)
        cv2.imwrite(str(tmp_path / "d.bmp"), img)
        loader = FrameLoader(str(tmp_path))
        frames = list(loader)
        assert len(frames) == 4

    def test_video_max_frames_len(self, tmp_path: Path):
        """Video len respects max_frames."""
        video_path = tmp_path / "test.mp4"
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for _ in range(10):
            video.write(_make_image((64, 48)))
        video.release()
        loader = FrameLoader(str(video_path), max_frames=5)
        assert len(loader) == 5


class TestHelpers:
    """Helper function tests."""

    def test_to_grayscale(self):
        """to_grayscale converts BGR to grayscale."""
        bgr = np.array([[[255, 128, 64]]], dtype=np.uint8)
        gray = to_grayscale(bgr)
        assert gray.shape == (1, 1)
        assert gray.dtype == np.uint8
        # cv2's grayscale conversion produces a value in [0, 255]
        assert 0 <= gray[0, 0] <= 255

    def test_to_grayscale_already_gray(self):
        """to_grayscale returns grayscale image unchanged."""
        gray = np.array([[100, 150]], dtype=np.uint8)
        result = to_grayscale(gray)
        np.testing.assert_array_equal(result, gray)

    def test_to_float(self):
        """to_float converts uint8 to float32 in [0, 1]."""
        img = np.array([[[255, 128, 0]]], dtype=np.uint8)
        f = to_float(img)
        assert f.dtype == np.float32
        assert f.shape == (1, 1, 3)
        assert f[0, 0, 0] == pytest.approx(1.0)
        assert f[0, 0, 1] == pytest.approx(128 / 255)
        assert f[0, 0, 2] == pytest.approx(0.0)

    def test_to_float_zero(self):
        """to_float with zero values gives 0.0."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        f = to_float(img)
        assert f.dtype == np.float32
        assert np.allclose(f, 0.0)
