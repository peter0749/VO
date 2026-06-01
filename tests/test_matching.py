"""Tests for ClassicMatcher and LightGlueMatcher in matching.py."""
import numpy as np
import numpy.testing as npt
import pytest
import torch

from slam_dnn.matching import (
    ClassicMatcher,
    LightGlueMatcher,
    MatcherBase,
    create_matcher,
)


class TestMatcherBase:
    """Tests for MatcherBase abstract class."""

    def test_lightglue_inherits_from_base(self):
        """LightGlueMatcher inherits from MatcherBase."""
        assert issubclass(LightGlueMatcher, MatcherBase)

    def test_classic_inherits_from_base(self):
        """ClassicMatcher inherits from MatcherBase."""
        assert issubclass(ClassicMatcher, MatcherBase)

    def test_matcher_base_is_abstract(self):
        """MatcherBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MatcherBase()

    def test_create_matcher_factory_lightglue(self):
        """create_matcher('lightglue') returns LightGlueMatcher."""
        matcher = create_matcher("lightglue", device="cpu")
        assert isinstance(matcher, LightGlueMatcher)

    def test_create_matcher_factory_classic(self):
        """create_matcher('classic') returns ClassicMatcher."""
        matcher = create_matcher("classic")
        assert isinstance(matcher, ClassicMatcher)

    def test_create_matcher_factory_unknown_raises(self):
        """create_matcher with unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown matcher method"):
            create_matcher("nonexistent")

    def test_create_matcher_forwards_kwargs(self):
        """create_matcher passes kwargs to constructor."""
        matcher = create_matcher("classic", ratio=0.5)
        assert isinstance(matcher, ClassicMatcher)
        assert matcher.ratio == 0.5

    def test_matcher_base_subclass_must_implement_match(self):
        """Subclass without match() cannot be instantiated."""
        class IncompleteMatcher(MatcherBase):
            pass

        with pytest.raises(TypeError):
            IncompleteMatcher()


class TestClassicMatcher:
    """Tests for ClassicMatcher."""

    @pytest.fixture
    def similar_features(self):
        """Create two feature sets with overlapping keypoints and similar descriptors."""
        np.random.seed(42)
        n = 50
        m = 50

        # Shared keypoints (first 30 are the same)
        shared_kp = np.random.rand(30, 2) * 100
        kp0 = np.vstack([shared_kp, np.random.rand(20, 2) * 100])
        kp1 = np.vstack([shared_kp, np.random.rand(20, 2) * 100])

        # Similar descriptors for shared keypoints (first 30)
        base_desc = np.random.rand(30, 256).astype(np.float32)
        desc0 = np.vstack([
            base_desc + np.random.randn(30, 256).astype(np.float32) * 0.1,
            np.random.rand(20, 256).astype(np.float32),
        ])
        desc1 = np.vstack([
            base_desc + np.random.randn(30, 256).astype(np.float32) * 0.1,
            np.random.rand(20, 256).astype(np.float32),
        ])

        return {
            "keypoints": kp0,
            "descriptors": desc0,
        }, {
            "keypoints": kp1,
            "descriptors": desc1,
        }

    def test_classic_finds_matches(self, similar_features):
        """ClassicMatcher finds matches between similar images using ratio test."""
        feats0, feats1 = similar_features
        matcher = ClassicMatcher(ratio=0.75)
        result = matcher.match(feats0, feats1)

        # Should find at least some matches (shared keypoints have similar descriptors)
        assert len(result["points0"]) > 0, "Should find matches between similar features"
        assert len(result["points1"]) > 0, "Should find matches between similar features"
        assert len(result["points0"]) == len(result["points1"])

    def test_classic_output_format(self, similar_features):
        """Output format matches LightGlueMatcher (points0, points1, scores, indices)."""
        feats0, feats1 = similar_features
        matcher = ClassicMatcher(ratio=0.75)
        result = matcher.match(feats0, feats1)

        # Check keys
        assert set(result.keys()) == {"points0", "points1", "scores", "indices"}

        # Check shapes
        K = len(result["points0"])
        assert result["points0"].shape == (K, 2)
        assert result["points1"].shape == (K, 2)
        assert result["scores"].shape == (K,)
        assert result["indices"].shape == (K, 2)

        # Check dtypes
        assert result["points0"].dtype == np.float32
        assert result["points1"].dtype == np.float32
        assert result["scores"].dtype == np.float32
        assert result["indices"].dtype == np.int32

        # Check scores are in [0, 1] range
        assert np.all(result["scores"] <= 1.0)
        assert np.all(result["scores"] >= 0)

        # Check indices correspond to points
        for i in range(K):
            assert result["indices"][i, 0] < len(feats0["keypoints"])
            assert result["indices"][i, 1] < len(feats1["keypoints"])

    def test_classic_ratio_filtering(self, similar_features):
        """Stricter ratio (lower value) = fewer matches."""
        feats0, feats1 = similar_features
        matcher_strict = ClassicMatcher(ratio=0.5)
        matcher_loose = ClassicMatcher(ratio=0.9)

        result_strict = matcher_strict.match(feats0, feats1)
        result_loose = matcher_loose.match(feats0, feats1)

        # Strict ratio should find fewer or equal matches
        assert len(result_strict["points0"]) <= len(result_loose["points0"])

    def test_classic_flann_method(self, similar_features):
        """ClassicMatcher works with FLANN method."""
        feats0, feats1 = similar_features
        matcher = ClassicMatcher(ratio=0.75, method="flann")
        result = matcher.match(feats0, feats1)

        # Should find matches with FLANN too
        assert len(result["points0"]) > 0
        assert set(result.keys()) == {"points0", "points1", "scores", "indices"}

    def test_classic_strict_ratio_no_matches(self):
        """Very strict ratio on dissimilar features yields no matches."""
        np.random.seed(123)
        n = 50
        m = 50

        kp0 = np.random.rand(n, 2) * 100
        kp1 = np.random.rand(m, 2) * 100
        desc0 = np.random.rand(n, 256).astype(np.float32)
        desc1 = np.random.rand(m, 256).astype(np.float32)

        feats0 = {"keypoints": kp0, "descriptors": desc0}
        feats1 = {"keypoints": kp1, "descriptors": desc1}

        matcher = ClassicMatcher(ratio=0.01)  # Very strict
        result = matcher.match(feats0, feats1)

        # Should find no matches with such strict ratio on random data
        assert len(result["points0"]) == 0

    def test_classic_scores_higher_better(self, similar_features):
        """Scores are normalized so higher = better match."""
        feats0, feats1 = similar_features
        matcher = ClassicMatcher(ratio=0.75)
        result = matcher.match(feats0, feats1)

        scores = result["scores"]
        # Scores should be non-negative
        assert np.all(scores >= 0)
        # Scores should be <= 1
        assert np.all(scores <= 1)
        # If there are matches, the best match should have the highest score
        if len(scores) > 0:
            best_idx = np.argmax(scores)
            assert scores[best_idx] == scores.max()


def _make_checkerboard(h: int = 240, w: int = 320, cell: int = 30) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, cell):
        for j in range(0, w, cell):
            if (i // cell + j // cell) % 2 == 0:
                img[i:i + cell, j:j + cell] = 255
    return img


def _extract_superpoint_feats(image: np.ndarray, max_kpts: int = 200) -> dict:
    """Extract SuperPoint features via lightglue's SuperPoint for test setup.

    Returns feature dict matching SuperPointExtractor output format.
    """
    from lightglue import SuperPoint

    extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to("cpu")
    img_tensor = (
        torch.from_numpy(image).float().div(255).unsqueeze(0).unsqueeze(0).to("cpu")
    )

    with torch.no_grad():
        feats = extractor({"image": img_tensor})

    return {
        "keypoints": feats["keypoints"][0].cpu().numpy().astype(np.float32),
        "descriptors": feats["descriptors"][0].cpu().numpy().astype(np.float32),
        "scores": feats["keypoint_scores"][0].cpu().numpy().astype(np.float32),
    }


@pytest.fixture(scope="module")
def lightglue_matcher():
    return LightGlueMatcher(filter_threshold=0.1, device="cpu")


@pytest.fixture(scope="module")
def pair_feats():
    """Two similar images (checkerboard + small shift) with pre-extracted features."""
    img0 = _make_checkerboard()
    img1 = np.roll(img0, 8, axis=1)

    feats0 = _extract_superpoint_feats(img0)
    feats1 = _extract_superpoint_feats(img1)
    return feats0, feats1, img0.shape


class TestLightGlueMatcher:
    """Tests for LightGlueMatcher."""

    def test_lightglue_instantiation(self):
        matcher = LightGlueMatcher(filter_threshold=0.1, device="cpu")
        assert matcher.matcher is not None
        assert str(matcher.device) == "cpu"

    def test_lightglue_finds_matches(self, lightglue_matcher, pair_feats):
        feats0, feats1, img_shape = pair_feats
        result = lightglue_matcher.match(feats0, feats1, image_size=img_shape)

        assert len(result["points0"]) > 0, (
            "Expected at least some matches between similar images"
        )

    def test_lightglue_output_format(self, lightglue_matcher, pair_feats):
        feats0, feats1, img_shape = pair_feats
        result = lightglue_matcher.match(feats0, feats1, image_size=img_shape)

        for key in ("points0", "points1", "scores", "indices"):
            assert key in result, f"Missing key: {key}"

        K = len(result["points0"])

        assert result["points0"].shape == (K, 2)
        assert result["points1"].shape == (K, 2)
        assert result["scores"].shape == (K,)
        assert result["indices"].shape == (K, 2)

        assert result["points0"].dtype == np.float32
        assert result["points1"].dtype == np.float32
        assert result["scores"].dtype == np.float32

        assert np.all(result["scores"] >= 0.0)
        assert np.all(result["scores"] <= 1.0)
        assert np.all(result["indices"] >= 0)
