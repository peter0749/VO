"""Shared pytest fixtures for slam_dnn test suite.

These fixtures provide reusable building blocks so that individual test files
stay focused on behavior, not setup. Each docstring doubles as a mini-tutorial
explaining *why* the fixture exists and *what* it gives you.
"""
import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

from slam_dnn import PinholeCamera



@pytest.fixture
def sample_image():
    """Synthetic 200x200 grayscale image with checkerboard pattern.

    The checkerboard gives strong corner features, making this ideal for
    unit tests that need a guaranteed-non-trivial input image.
    """
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(0, 200, 40):
        for j in range(0, 200, 40):
            if (i // 40 + j // 40) % 2 == 0:
                img[i:i+40, j:j+40] = 255
    return img


@pytest.fixture
def tmp_image_dir():
    """Temporary directory with 5 PNG test images, cleaned up after test.

    Useful for IO tests that need a directory with some image-like files.
    Files are .npy arrays (not real PNGs), so only use this fixture with
    logic that doesn't inspect image contents.
    """
    tmp_dir = tempfile.mkdtemp(prefix="slam_dnn_test_")
    for i in range(5):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        np.save(Path(tmp_dir) / f"frame_{i:04d}.npy", img)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)



@pytest.fixture
def sample_k_camera():
    """Standard pinhole camera (640x480, FOV 63°) — typical phone wide-angle.

    Why 63°?  Most smartphone main cameras sit around 60-65° horizontal FOV.
    This fixture provides a reusable camera for tests that need intrinsics
    but don't care about exact values.

    Usage:
        cam = sample_k_camera   # in a test function via parameter injection
        assert cam.K.shape == (3, 3)
        assert cam.K_inv.shape == (3, 3)
    """
    return PinholeCamera(width=640, height=480, fov_deg=63.0)


@pytest.fixture
def sample_image_pair():
    """Synthetic image pair with distinctive features for matching tests.

    Returns (img0, img1) as (H, W) uint8 grayscale arrays.
    img1 is img0 shifted right by 20 pixels (simulates forward+right motion).

    Why not two random images?  A deterministic shift gives a known ground-truth
    "motion" that a matcher should detect without needing a real VO pipeline.

    Usage:
        img0, img1 = sample_image_pair
        feats0 = extractor.extract(img0)
        feats1 = extractor.extract(img1)
        matches = matcher.match(feats0, feats1)
    """
    np.random.seed(123)
    h, w = 480, 640
    img0 = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, 40):
        for j in range(0, w, 40):
            if (i // 40 + j // 40) % 2 == 0:
                img0[i:i+40, j:j+40] = 255
    noise = np.random.randint(0, 30, (h, w), dtype=np.uint8)
    img0 = np.clip(
        img0.astype(np.int16) + noise.astype(np.int16), 0, 255
    ).astype(np.uint8)
    img1 = np.roll(img0, 20, axis=1)
    return img0, img1


@pytest.fixture
def sample_3d_points():
    """50 random 3D points in camera view (z > 3).

    Why z > 3?  Points too close to the camera (< 1m) cause numerical
    instability in projection. z ∈ [3, 15] keeps all points well in front
    of both cameras for typical ±2m baseline motion.

    Returns:
        (50, 3) float64 array with X ∈ [-5, 5], Y ∈ [-3, 3], Z ∈ [3, 15]

    Usage:
        pts = sample_3d_points
        pts_2d = (K @ pts.T).T   # project to pixel coords
    """
    rng = np.random.default_rng(42)
    pts = np.zeros((50, 3), dtype=np.float64)
    pts[:, 0] = rng.uniform(-5.0, 5.0, 50)
    pts[:, 1] = rng.uniform(-3.0, 3.0, 50)
    pts[:, 2] = rng.uniform(3.0, 15.0, 50)
    return pts



def pytest_addoption(parser):
    parser.addoption(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="Skip tests that require baseline submodules (e.g. minislam).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "baseline: test requires a baseline submodule (e.g. minislam)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-baseline"):
        skip_marker = pytest.mark.skip(reason="--skip-baseline flag set")
        for item in items:
            if "baseline" in item.keywords:
                item.add_marker(skip_marker)
