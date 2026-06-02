"""Unit tests for slam_dnn.testdata.synthetic — SyntheticVODataset.

Covers:
- All 3 scenarios produce valid output (images, poses, K, points)
- Round-trip: save → load with FrameLoader + load_kitti_format
- Translation scenario: identity rotation within tolerance
- Rotation scenario: zero translation within tolerance
- CLI wrapper produces correct output
- Feature rendering: non-trivial images (not all black or white)
- Intrinsic matrix K matches expected values
- Invalid scenario raises ValueError
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from slam_dnn.export import load_kitti_format
from slam_dnn.io import FrameLoader
from slam_dnn.testdata.synthetic import SyntheticVODataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_ds(
    scenario: str,
    n_frames: int = 5,
    n_points: int = 50,
    **kwargs,
) -> SyntheticVODataset:
    """Create a small dataset for fast unit testing."""
    defaults = dict(
        image_size=(160, 120),
        fov_deg=63.0,
        noise_px=0.5,
        seed=42,
    )
    defaults.update(kwargs)
    return SyntheticVODataset(
        scenario=scenario,
        n_frames=n_frames,
        n_points=n_points,
        **defaults,
    )


# ---------------------------------------------------------------------------
# Test 1–3: Each scenario generates valid output
# ---------------------------------------------------------------------------

class TestScenarioOutput:
    """Each scenario produces non-empty images, 4x4 poses, 3x3 K."""

    @pytest.mark.parametrize("scenario", ["translation", "rotation", "mixed"])
    def test_scenario_generates_valid_output(self, scenario: str):
        """Scenario produces correct shapes and non-empty images."""
        ds = _small_ds(scenario, n_frames=5)
        data = ds.generate()

        assert len(data["images"]) == 5
        assert len(data["gt_poses"]) == 5
        assert data["K"].shape == (3, 3)
        assert data["points_3d"].shape == (50, 3)

        for img in data["images"]:
            assert img.shape == (120, 160)
            assert img.dtype == np.uint8

        for T in data["gt_poses"]:
            assert T.shape == (4, 4)
            # Bottom row should be [0, 0, 0, 1]
            np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-12)


# ---------------------------------------------------------------------------
# Test 4: Translation scenario has identity rotation
# ---------------------------------------------------------------------------

class TestTranslationIdentityRotation:
    """Translation scenario: all poses have identity rotation."""

    def test_identity_rotation_within_tolerance(self):
        """All rotation sub-matrices are identity within 1e-10."""
        ds = _small_ds("translation", n_frames=10)
        data = ds.generate()

        for i, T in enumerate(data["gt_poses"]):
            R = T[:3, :3]
            np.testing.assert_allclose(
                R, np.eye(3),
                atol=1e-10,
                err_msg=f"Frame {i}: rotation is not identity",
            )

    def test_translation_along_x_axis(self):
        """Camera translates along X; Y and Z translation remain zero."""
        ds = _small_ds("translation", n_frames=10)
        data = ds.generate()

        for i, T in enumerate(data["gt_poses"]):
            t = T[:3, 3]
            assert abs(t[1]) < 1e-10, f"Frame {i}: Y translation != 0"
            assert abs(t[2]) < 1e-10, f"Frame {i}: Z translation != 0"

    def test_translation_monotonic(self):
        """Camera X translation is monotonically non-decreasing (moving right)."""
        ds = _small_ds("translation", n_frames=10)
        data = ds.generate()

        prev_x = None
        for i, T in enumerate(data["gt_poses"]):
            # Camera center in world = -R^T @ t = -t (since R=I)
            cam_x = -T[0, 3]
            if prev_x is not None:
                assert cam_x >= prev_x - 1e-10, (
                    f"Frame {i}: camera X not monotonic "
                    f"({cam_x:.4f} < {prev_x:.4f})"
                )
            prev_x = cam_x


# ---------------------------------------------------------------------------
# Test 5: Rotation scenario has zero translation
# ---------------------------------------------------------------------------

class TestRotationZeroTranslation:
    """Rotation scenario: all poses have zero translation."""

    def test_zero_translation_within_tolerance(self):
        """All translation vectors are zero within 1e-10."""
        ds = _small_ds("rotation", n_frames=10)
        data = ds.generate()

        for i, T in enumerate(data["gt_poses"]):
            t = T[:3, 3]
            np.testing.assert_allclose(
                t, np.zeros(3),
                atol=1e-10,
                err_msg=f"Frame {i}: translation is not zero",
            )

    def test_rotation_is_valid_orthogonal(self):
        """All rotation matrices are proper orthogonal (det=+1)."""
        ds = _small_ds("rotation", n_frames=10)
        data = ds.generate()

        for i, T in enumerate(data["gt_poses"]):
            R = T[:3, :3]
            np.testing.assert_allclose(
                R @ R.T, np.eye(3), atol=1e-10,
                err_msg=f"Frame {i}: R @ R.T != I",
            )
            det = np.linalg.det(R)
            assert abs(det - 1.0) < 1e-10, (
                f"Frame {i}: det(R) = {det}, expected 1.0"
            )


# ---------------------------------------------------------------------------
# Test 6: Round-trip save → load
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Generate → save → load with FrameLoader + load_kitti_format."""

    def test_images_loadable_by_frame_loader(self, tmp_path: Path):
        """Saved images are loadable by FrameLoader."""
        ds = _small_ds("mixed", n_frames=4)
        out_dir = tmp_path / "synth_rt"
        ds.save(str(out_dir))

        loader = FrameLoader(str(out_dir / "image_0"))
        frames = list(loader)
        assert len(frames) == 4
        # FrameLoader yields BGR by default
        for frame in frames:
            assert frame.shape[0] == 120
            assert frame.shape[1] == 160

    def test_poses_round_trip(self, tmp_path: Path):
        """Saved poses match original poses when loaded back."""
        ds = _small_ds("translation", n_frames=6)
        data = ds.generate()
        out_dir = tmp_path / "synth_poses"
        ds.save(str(out_dir))

        loaded = load_kitti_format(str(out_dir / "poses.txt"))
        assert len(loaded) == 6

        for i, (orig, lod) in enumerate(zip(data["gt_poses"], loaded)):
            np.testing.assert_allclose(
                orig, lod, atol=1e-5,
                err_msg=f"Pose {i} mismatch after round-trip",
            )

    def test_calib_file_exists(self, tmp_path: Path):
        """save() creates calib.txt with valid content."""
        ds = _small_ds("rotation", n_frames=3)
        out_dir = tmp_path / "synth_calib"
        ds.save(str(out_dir))

        calib_path = out_dir / "calib.txt"
        assert calib_path.exists()

        content = calib_path.read_text()
        assert "P0:" in content
        assert "R0_rect:" in content
        assert "Tr_velo_to_cam:" in content


# ---------------------------------------------------------------------------
# Test 7: Feature rendering produces non-trivial images
# ---------------------------------------------------------------------------

class TestFeatureRendering:
    """Rendered images contain actual features (not all black/white)."""

    @pytest.mark.parametrize("scenario", ["translation", "rotation", "mixed"])
    def test_images_not_all_black_or_white(self, scenario: str):
        """Rendered images have intermediate intensity values."""
        ds = _small_ds(scenario, n_frames=3, n_points=100)
        data = ds.generate()

        for i, img in enumerate(data["images"]):
            mean_val = float(np.mean(img))
            std_val = float(np.std(img))

            # Not all black: mean should be > 10
            assert mean_val > 10, (
                f"Frame {i} ({scenario}): image too dark "
                f"(mean={mean_val:.1f})"
            )
            # Not all white: mean should be < 245
            assert mean_val < 245, (
                f"Frame {i} ({scenario}): image too bright "
                f"(mean={mean_val:.1f})"
            )
            # Has variance (features present)
            assert std_val > 5, (
                f"Frame {i} ({scenario}): image too uniform "
                f"(std={std_val:.1f})"
            )

    def test_different_frames_differ(self):
        """Consecutive frames are not identical (camera actually moves)."""
        ds = _small_ds("translation", n_frames=5)
        data = ds.generate()

        for i in range(1, len(data["images"])):
            diff = np.abs(
                data["images"][i].astype(int)
                - data["images"][i - 1].astype(int)
            ).mean()
            assert diff > 0.5, (
                f"Frames {i-1} and {i} are nearly identical "
                f"(mean diff={diff:.2f})"
            )


# ---------------------------------------------------------------------------
# Test 8: Intrinsic matrix K
# ---------------------------------------------------------------------------

class TestIntrinsicMatrix:
    """K matrix has correct structure and focal length."""

    def test_k_structure(self):
        """K is 3x3 with principal point at image center."""
        ds = _small_ds("mixed", n_frames=2)
        data = ds.generate()
        K = data["K"]

        assert K.shape == (3, 3)
        assert K[2, 2] == 1.0
        assert K[1, 0] == 0.0
        assert K[0, 1] == 0.0
        assert K[2, 0] == 0.0
        assert K[2, 1] == 0.0

        # Principal point at image center
        w, h = 160, 120
        assert abs(K[0, 2] - w / 2.0) < 1e-10
        assert abs(K[1, 2] - h / 2.0) < 1e-10

    def test_k_focal_length(self):
        """Focal length matches FOV formula."""
        ds = SyntheticVODataset(
            scenario="translation",
            n_frames=2,
            image_size=(640, 480),
            fov_deg=63.0,
        )
        data = ds.generate()
        K = data["K"]

        expected_fx = (640 / 2.0) / np.tan(np.deg2rad(63.0) / 2.0)
        assert abs(K[0, 0] - expected_fx) < 1e-6
        assert abs(K[1, 1] - expected_fx) < 1e-6  # square pixels


# ---------------------------------------------------------------------------
# Test 9: Invalid scenario raises ValueError
# ---------------------------------------------------------------------------

class TestInvalidInput:
    """Invalid inputs raise appropriate errors."""

    def test_invalid_scenario_raises(self):
        """Unknown scenario string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            SyntheticVODataset(scenario="flyby")

    def test_too_few_frames_raises(self):
        """n_frames < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_frames must be >= 2"):
            SyntheticVODataset(scenario="mixed", n_frames=1)

    def test_zero_points_raises(self):
        """n_points < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_points must be >= 1"):
            SyntheticVODataset(scenario="mixed", n_points=0)


# ---------------------------------------------------------------------------
# Test 10: CLI wrapper
# ---------------------------------------------------------------------------

class TestCLI:
    """CLI script generates output correctly."""

    def test_cli_mixed_scenario(self, tmp_path: Path):
        """CLI generates files for mixed scenario."""
        out_dir = tmp_path / "cli_test"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "scripts" / "generate_synthetic.py"),
                "--scenario", "mixed",
                "--n-frames", "5",
                "--n-points", "30",
                "--width", "160",
                "--height", "120",
                "--output", str(out_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Check output files exist
        assert (out_dir / "image_0").is_dir()
        assert (out_dir / "poses.txt").exists()
        assert (out_dir / "calib.txt").exists()

        # Load and verify
        loader = FrameLoader(str(out_dir / "image_0"))
        frames = list(loader)
        assert len(frames) == 5

        poses = load_kitti_format(str(out_dir / "poses.txt"))
        assert len(poses) == 5

    def test_cli_translation_scenario(self, tmp_path: Path):
        """CLI works for translation scenario."""
        out_dir = tmp_path / "cli_trans"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "scripts" / "generate_synthetic.py"),
                "--scenario", "translation",
                "--n-frames", "3",
                "--output", str(out_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert (out_dir / "poses.txt").exists()

    def test_cli_invalid_scenario_fails(self, tmp_path: Path):
        """CLI rejects invalid scenario."""
        out_dir = tmp_path / "cli_bad"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "scripts" / "generate_synthetic.py"),
                "--scenario", "flyby",
                "--output", str(out_dir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # argparse exits with code 2 for invalid arguments
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Test 11: Deterministic output with same seed
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same seed produces identical output."""

    def test_same_seed_same_output(self):
        """Two datasets with same seed produce identical images."""
        ds1 = _small_ds("mixed", n_frames=3)
        ds2 = _small_ds("mixed", n_frames=3)
        d1 = ds1.generate()
        d2 = ds2.generate()

        for i in range(3):
            np.testing.assert_array_equal(
                d1["images"][i], d2["images"][i],
                err_msg=f"Frame {i} differs with same seed",
            )

        for i in range(3):
            np.testing.assert_allclose(d1["gt_poses"][i], d2["gt_poses"][i])

    def test_different_seed_different_output(self):
        """Different seeds produce different images."""
        ds1 = SyntheticVODataset(
            scenario="mixed", n_frames=3, n_points=50,
            image_size=(160, 120), seed=42,
        )
        ds2 = SyntheticVODataset(
            scenario="mixed", n_frames=3, n_points=50,
            image_size=(160, 120), seed=99,
        )
        d1 = ds1.generate()
        d2 = ds2.generate()

        # At least one frame should differ
        any_differ = False
        for i in range(3):
            if not np.array_equal(d1["images"][i], d2["images"][i]):
                any_differ = True
                break
        assert any_differ, "Different seeds produced identical images"
