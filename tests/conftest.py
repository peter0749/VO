"""Shared pytest fixtures for slam_dnn tests."""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_image():
    """Synthetic 200x200 grayscale image with checkerboard pattern."""
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(0, 200, 40):
        for j in range(0, 200, 40):
            if (i // 40 + j // 40) % 2 == 0:
                img[i:i+40, j:j+40] = 255
    return img


@pytest.fixture
def tmp_image_dir():
    """Temporary directory with 5 PNG test images, cleaned up after test."""
    tmp_dir = tempfile.mkdtemp(prefix="slam_dnn_test_")
    for i in range(5):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        np.save(Path(tmp_dir) / f"frame_{i:04d}.npy", img)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
