"""Frame loading from image directories and video files."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator


class FrameLoader:
    """Load frames from an image directory or video file.

    Automatically detects the input type. If the source is a directory, it
    scans for image files with extensions in {.png, .jpg, .jpeg, .ppm, .bmp},
    sorted alphabetically. If the source is a file, it opens it as a video
    via OpenCV's VideoCapture.

    FrameLoader is iterable and supports len(). Optional parameters cap the
    total frames and allow resizing.

    Args:
        source: Path to an image directory or video file.
        max_frames: If provided, stop after yielding this many frames.
            Useful for debugging or processing a small subset.
        resize: If provided as (W, H), each frame is resized to this shape
            before yielding. None keeps the original resolution.

    Raises:
        FileNotFoundError: If source does not exist.
        ValueError: If directory contains no image files, video file
            cannot be opened, or source is neither a file nor directory.

    Example:
        >>> loader = FrameLoader('path/to/images/', max_frames=10)
        >>> len(loader)
        10
        >>> for i, frame in enumerate(loader):
        ...     print(f"Frame {i}: shape={frame.shape}")
    """

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}

    def __init__(
        self,
        source: str,
        max_frames: int | None = None,
        resize: tuple[int, int] | None = None,
    ):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        self.source = source_path
        self.max_frames = max_frames
        self.resize = resize
        self._images: list[Path] | None = None
        self._video: cv2.VideoCapture | None = None

        if source_path.is_dir():
            self._images = sorted(
                p for p in source_path.iterdir() if p.suffix.lower() in self._IMAGE_EXTENSIONS
            )
            if not self._images:
                raise ValueError(f"No image files found in {source}")
        elif source_path.is_file():
            self._video = cv2.VideoCapture(str(source_path))
            if not self._video.isOpened():
                raise ValueError(f"Cannot open video: {source}")
        else:
            raise ValueError(f"Unknown source type: {source}")

    def __len__(self) -> int:
        """Return the total number of frames available (capped by max_frames)."""
        if self._images is not None:
            n = len(self._images)
        else:
            n = int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.max_frames is not None:
            n = min(n, self.max_frames)
        return n

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield BGR frames as uint8 ndarrays, one at a time.

        For image directories, each file is loaded and decoded via OpenCV.
        For video files, the stream is read frame by frame. The video
        stream is seeked to frame 0 at the start of iteration so that
        the loader can be iterated multiple times.
        """
        count = 0
        if self._images is not None:
            for img_path in self._images:
                if self.max_frames is not None and count >= self.max_frames:
                    break
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    if self.resize is not None:
                        frame = cv2.resize(frame, self.resize)
                    yield frame
                    count += 1
        else:
            self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                if self.max_frames is not None and count >= self.max_frames:
                    break
                ret, frame = self._video.read()
                if not ret:
                    break
                if self.resize is not None:
                    frame = cv2.resize(frame, self.resize)
                yield frame
                count += 1

    def __del__(self):
        """Release the video capture if one is open."""
        video = getattr(self, "_video", None)
        if video is not None:
            video.release()


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR color image to grayscale uint8.

    If the input is already 2D (grayscale), it is returned unchanged.

    Args:
        img: uint8 ndarray of shape (H, W) or (H, W, 3) in BGR order.

    Returns:
        uint8 ndarray of shape (H, W).
    """
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_float(img: np.ndarray) -> np.ndarray:
    """Convert a uint8 image to float32 in the range [0, 1].

    Args:
        img: uint8 ndarray (any shape).

    Returns:
        float32 ndarray with values normalized to [0, 1].
    """
    return img.astype(np.float32) / 255.0
