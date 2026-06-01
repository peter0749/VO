"""Tests for SuperPointExtractor in features.py."""

import numpy as np
import numpy.testing as npt
import pytest

from slam_dnn.features import SuperPointExtractor


@pytest.fixture(scope="module")
def extractor():
    """Shared SuperPointExtractor on CPU with low max_keypoints for speed."""
    return SuperPointExtractor(max_keypoints=100, device="cpu")


@pytest.fixture
def random_grayscale():
    """240x320 grayscale image with random content."""
    return np.random.randint(0, 255, (240, 320), dtype=np.uint8)


@pytest.fixture
def random_color():
    """240x320 color image with random content."""
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


class TestSuperPointOutputFormat:
    """Tests for correct output format from extract()."""

    def test_extract_returns_dict_with_required_keys(self, extractor, random_grayscale):
        """extract() returns dict with keypoints, descriptors, scores."""
        result = extractor.extract(random_grayscale)
        assert isinstance(result, dict)
        assert "keypoints" in result
        assert "descriptors" in result
        assert "scores" in result

    def test_output_shapes(self, extractor, random_grayscale):
        """Output shapes: (N,2), (N,256), (N,) with consistent N."""
        result = extractor.extract(random_grayscale)
        kpts = result["keypoints"]
        desc = result["descriptors"]
        scores = result["scores"]

        assert kpts.ndim == 2 and kpts.shape[1] == 2, \
            f"keypoints shape {kpts.shape}, expected (N, 2)"
        assert desc.ndim == 2 and desc.shape[1] == 256, \
            f"descriptors shape {desc.shape}, expected (N, 256)"
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


class TestDescriptorNormalization:
    """Tests for L2-normalization of descriptors."""

    def test_descriptors_are_l2_normalized(self, extractor, random_grayscale):
        """Each descriptor row has L2 norm ≈ 1.0."""
        result = extractor.extract(random_grayscale)
        desc = result["descriptors"]
        if desc.shape[0] == 0:
            pytest.skip("No keypoints detected")

        norms = np.linalg.norm(desc, axis=1)
        npt.assert_allclose(norms, 1.0, atol=1e-5,
                            err_msg="Descriptors are not L2-normalized")

    def test_descriptors_normalized_color_input(self, extractor, random_color):
        """L2-normalization holds for color input too."""
        result = extractor.extract(random_color)
        desc = result["descriptors"]
        if desc.shape[0] == 0:
            pytest.skip("No keypoints detected on color image")

        norms = np.linalg.norm(desc, axis=1)
        npt.assert_allclose(norms, 1.0, atol=1e-5)


class TestInputVariants:
    """Tests for different input types."""

    def test_color_image_input(self, extractor, random_color):
        """3-channel color image produces valid output."""
        result = extractor.extract(random_color)
        assert result["keypoints"].shape[1] == 2
        assert result["descriptors"].shape[1] == 256

    def test_checkerboard_sample_image(self, extractor, sample_image):
        """Checkerboard fixture produces at least some keypoints."""
        result = extractor.extract(sample_image)
        assert result["keypoints"].shape[0] > 0, \
            "Expected keypoints on checkerboard pattern"

    def test_max_keypoints_respected(self, extractor, random_grayscale):
        """Number of keypoints does not exceed max_keypoints."""
        result = extractor.extract(random_grayscale)
        assert result["keypoints"].shape[0] <= 100
