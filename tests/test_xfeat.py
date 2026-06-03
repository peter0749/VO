"""Tests for XFeatExtractor and XFeatMatcher."""

import numpy as np
import numpy.testing as npt
import pytest
import torch

from slam_dnn.features import XFeatExtractor
from slam_dnn.matching import XFeatMatcher, create_matcher
from slam_dnn.vo import VisualOdometry
from slam_dnn.camera import PinholeCamera
from slam_dnn.config import VOConfig


@pytest.fixture(scope="module")
def extractor():
    """Shared XFeatExtractor on CPU with low max_keypoints for speed."""
    return XFeatExtractor(max_keypoints=100, device="cpu")


@pytest.fixture
def random_grayscale():
    """240x320 grayscale image with random content."""
    return np.random.randint(0, 255, (240, 320), dtype=np.uint8)


@pytest.fixture
def random_color():
    """240x320 color image with random content."""
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


class TestXFeatOutputFormat:
    """Tests for correct output format from XFeatExtractor."""

    def test_extract_returns_dict_with_required_keys(self, extractor, random_grayscale):
        """extract() returns dict with keypoints, descriptors, scores."""
        result = extractor.extract(random_grayscale)
        assert isinstance(result, dict)
        assert "keypoints" in result
        assert "descriptors" in result
        assert "scores" in result

    def test_output_shapes(self, extractor, random_grayscale):
        """Output shapes: (N,2), (N,64), (N,) with consistent N."""
        result = extractor.extract(random_grayscale)
        kpts = result["keypoints"]
        desc = result["descriptors"]
        scores = result["scores"]

        assert kpts.ndim == 2 and kpts.shape[1] == 2, \
            f"keypoints shape {kpts.shape}, expected (N, 2)"
        assert desc.ndim == 2 and desc.shape[1] == 64, \
            f"descriptors shape {desc.shape}, expected (N, 64)"
        assert scores.ndim == 1, \
            f"scores shape {scores.shape}, expected (N,)"

        n = kpts.shape[0]
        assert desc.shape[0] == n, "descriptors count mismatch with keypoints"
        assert scores.shape[0] == n, "scores count mismatch with keypoints"

    def test_output_dtypes_float32(self, extractor, random_grayscale):
        """All output arrays are float32."""
        result = extractor.extract(random_grayscale)
        assert result["keypoints"].dtype == np.float32
        assert result["descriptors"].dtype == np.float32
        assert result["scores"].dtype == np.float32

    def test_descriptors_are_l2_normalized(self, extractor, random_grayscale):
        """Each descriptor row has L2 norm ≈ 1.0."""
        result = extractor.extract(random_grayscale)
        desc = result["descriptors"]
        if desc.shape[0] == 0:
            pytest.skip("No keypoints detected")

        norms = np.linalg.norm(desc, axis=1)
        npt.assert_allclose(norms, 1.0, atol=1e-5,
                            err_msg="Descriptors are not L2-normalized")


class TestXFeatMatcher:
    """Tests for XFeatMatcher."""

    def test_matching_synthetic(self):
        """Verify matcher correctly finds matches between identical feature sets."""
        matcher = create_matcher("xfeat", min_cossim=0.8, device="cpu")
        assert isinstance(matcher, XFeatMatcher)

        # Create dummy features
        kpts = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        desc = np.random.randn(3, 64).astype(np.float32)
        norms = np.linalg.norm(desc, axis=1, keepdims=True)
        desc = desc / np.clip(norms, 1e-8, None)
        scores = np.array([0.9, 0.8, 0.75], dtype=np.float32)

        feats0 = {"keypoints": kpts, "descriptors": desc, "scores": scores}
        feats1 = {"keypoints": kpts.copy(), "descriptors": desc.copy(), "scores": scores.copy()}

        result = matcher.match(feats0, feats1)
        assert "points0" in result
        assert "points1" in result
        assert "scores" in result
        assert "indices" in result

        # Since features are identical, MNN matcher should match all of them
        assert len(result["points0"]) == 3
        npt.assert_allclose(result["points0"], kpts)
        npt.assert_allclose(result["points1"], kpts)


class TestXFeatVOIntegration:
    """End-to-end VO pipeline integration tests with XFeat."""

    def test_vo_run_with_xfeat(self):
        """Verify that VisualOdometry can run using XFeat extractor and matcher."""
        camera = PinholeCamera(width=320, height=240, fov_deg=60.0)
        config = VOConfig(
            extractor="xfeat",
            matcher="xfeat",
            max_keypoints=50,
            device="cpu",
        )
        vo = VisualOdometry(camera=camera, config=config)

        # Process two dummy images
        img1 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img2 = img1.copy()

        # Just verify it doesn't crash on processing
        pose1 = vo.process_frame(img1)
        assert pose1 is None  # First frame is always keyframe, returns None

        pose2 = vo.process_frame(img2)
        # Should process without crash (pose2 can be relative pose or None if tracking failed/insufficient matches)
