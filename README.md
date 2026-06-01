# slam_dnn

SuperPoint-based monocular visual odometry library for camera trajectory estimation.

[![tests](https://img.shields.io/badge/tests-170%2B-green.svg)]()
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## Features

- Learned feature extraction via SuperPoint (256-d descriptors)
- Dual matcher support: LightGlue (neural) and Classic (BF+ratio)
- Robust pose estimation via 5-point Essential matrix RANSAC
- End-to-end VO pipeline with frame-to-frame SE3 accumulation
- Pure rotation fallback via Homography detection
- Standard trajectory formats: KITTI and TUM
- Trajectory evaluation with Umeyama alignment and APE/RTE metrics
- Flexible input: image directories and video files
- Trajectory visualization with matplotlib

### What It Actually Does

slam_dnn takes a sequence of images (from a directory or video file) and estimates
camera motion frame by frame. It uses SuperPoint for keypoint extraction, matches
those keypoints between consecutive frames, and recovers relative camera pose from
the Essential matrix. The result is a trajectory of camera positions and orientations
in 3D space.

This is frame-to-frame visual odometry. It does not do loop closure, SLAM, or
landmark management.

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+ (for SuperPoint and LightGlue)
- OpenCV 4.8+
- NumPy, SciPy, matplotlib

### Install from Source

```bash
git clone <repo-url>
cd SLAM-DNN
pip install -e .
```

### Install LightGlue (required for the default neural matcher)

```bash
pip install git+https://github.com/cvg/LightGlue
```

This installs both SuperPoint and LightGlue from the cvg package. SuperPoint is
the feature extractor. LightGlue is one of two available matchers.

## Quick Start

### CLI Usage

Run visual odometry on an image sequence:

```bash
python -m slam_dnn \
    --input path/to/images/ \
    --output results/ \
    --fov 63.0 \
    --matcher lightglue \
    --device auto
```

Arguments:

- `--input, -i`: Image directory (PNG/JPG) or video file
- `--output, -o`: Output directory (auto-created)
- `--fov`: Horizontal field of view in degrees (default: 63.0, phone wide-angle)
- `--matcher`: `lightglue` (default) or `classic`
- `--max-keypoints`: Max keypoints per frame (default: 2048)
- `--scale`: Trajectory scale factor (default: 1.0)
- `--device`: `auto`, `cuda`, `mps`, or `cpu` (default: auto)
- `--no-plot`: Skip trajectory plot generation
- `--verbose, -v`: Verbosity level. Repeat for more: `-vv` for DEBUG

### Output Files

The CLI produces three files:

- `trajectory_kitti.txt` in KITTI format (12 floats per line, row-major 3x4 matrix)
- `trajectory_tum.txt` in TUM format (timestamp + xyz + quaternion)
- `trajectory_plot.png` as a top-down trajectory visualization

### With Ground Truth Evaluation

```bash
python -m slam_dnn \
    --input images/ \
    --output results/ \
    --ground-truth path/to/gt.txt \
    --fov 63.0
```

When ground truth is provided, the CLI generates an additional `evaluation_report.txt`
with APE and RTE metrics computed after Umeyama alignment.

## Python API

### Basic Usage: Visual Odometry in Seven Lines

```python
from slam_dnn import VisualOdometry, PinholeCamera, FrameLoader

camera = PinholeCamera(width=640, height=480, fov_deg=63.0)
vo = VisualOdometry(camera, matcher='lightglue', device='cpu')

# Load frames from a directory of images or a video file
loader = FrameLoader('path/to/images/', max_frames=100)

# Process each frame
for frame in loader:
    pose = vo.process_frame(frame)  # Returns 3x4 pose or None

# Get the full trajectory
poses = vo.get_trajectory().get_poses()  # list of 4x4 SE3 matrices
stats = vo.get_stats()
print(f"Processed {stats['total']} frames, {stats['successful']} successful")
```

### Export Trajectory

```python
from slam_dnn import export_kitti_format, export_tum_format

poses = vo.get_trajectory().get_poses()
export_kitti_format(poses, "trajectory_kitti.txt")
export_tum_format(poses, "trajectory_tum.txt")

# Or use the convenience method on the trajectory object
vo.get_trajectory().save("trajectory.txt", format='kitti')
```

### Evaluate Against Ground Truth

```python
from slam_dnn import (
    load_kitti_format,
    load_tum_format,
    evaluate,
    align_umeyama,
)

# Load both trajectories
est_poses = load_kitti_format("estimated.txt")
gt_poses = load_kitti_format("ground_truth.txt")

# Full evaluation report
report = evaluate(est_poses, gt_poses, with_scale=True)
print(f"APE RMSE: {report['ape_rmse']:.4f}")
print(f"RTE RMSE: {report['rte_rmse']:.4f}")
print(f"Umeyama scale: {report['scale']:.4f}")
```

The `evaluate` function aligns the estimated trajectory to ground truth via
Umeyama (Sim(3) least-squares), then computes Absolute Pose Error and Relative
Trajectory Error. The `with_scale=True` flag corrects the monocular scale
ambiguity during alignment.

### Component-Level Usage

If you need fine-grained control over the pipeline stages, use the components
directly:

```python
from slam_dnn import (
    SuperPointExtractor,
    LightGlueMatcher,
    ClassicMatcher,
    estimate_essential,
    K_from_fov,
)

# Feature extraction
extractor = SuperPointExtractor(max_keypoints=1024, device='cpu')
feats = extractor.extract(image)
# feats = {"keypoints": (N,2), "descriptors": (N,256), "scores": (N,)}

# Match features between two frames
matcher = LightGlueMatcher(device='cpu')  # or ClassicMatcher()
matches = matcher.match(feats0, feats1)
# matches = {"points0", "points1", "scores", "indices"}

# Estimate relative pose from matched keypoints
K = K_from_fov(640, 480, fov_deg=63.0)
result = estimate_essential(matches['points0'], matches['points1'], K)
if result is not None:
    R, t, inliers = result
    print(f"Rotation:\n{R}")
    print(f"Translation: {t}")
    print(f"Inliers: {inliers.sum()}/{len(inliers)}")
```

### Configuration

```python
from slam_dnn import VOConfig, VisualOdometry, PinholeCamera

config = VOConfig(
    max_keypoints=2048,
    detection_threshold=0.0005,
    matcher='lightglue',
    fov_deg=63.0,
    min_matches=20,
    scale=1.0,
    device='auto',
)

camera = PinholeCamera(width=640, height=480, fov_deg=config.fov_deg)
vo = VisualOdometry(
    camera,
    matcher=config.matcher,
    max_keypoints=config.max_keypoints,
    device=config.device,
)
```

`VOConfig` is a dataclass with sensible defaults tuned for mobile phone
wide-angle cameras. Pass it to `VisualOdometry` via the `config` parameter to
override individual keyword arguments.

## Visualization

```python
from slam_dnn.visualization import plot_trajectory_comparison
import numpy as np

poses = vo.get_trajectory().get_poses()
plot_trajectory_comparison(
    estimated=poses,
    ground_truth=None,
    title="My Trajectory",
    save_path="output/trajectory.png",
    show=True,
)
```

The `plot_trajectory_comparison` function renders a top-down 2D view of the
trajectory. If ground truth poses are provided, they are overlaid in dashed gray
for comparison. The plot always shows start and end markers, a grid overlay, and
an equal aspect ratio for spatial accuracy.

For 3D visualization, use `plot_trajectory_3d` from the same module.

## Testing

Run the full test suite:

```bash
pytest tests/
```

Run only fast unit tests (skips slow integration tests):

```bash
pytest tests/ --ignore=tests/test_synthetic_vo.py
```

The test suite covers unit tests for every module, integration tests for the full
VO pipeline, and edge-case handling for tracking loss and pure rotation.

## Known Limitations

**Scale ambiguity.** Monocular VO only recovers the direction of translation, not
its magnitude. Estimated trajectories will be correct in shape but wrong in scale.
Use `evaluate(..., with_scale=True)` to handle this during evaluation.

**Pure rotation.** When the camera only rotates without translating, the Essential
matrix degenerates and cannot produce a valid pose. Use `detect_pure_rotation()`
or `estimate_essential_or_homography()` for a homography-based fallback.

**Tracking loss.** If fewer than `min_matches` features match between consecutive
frames (default: 20), the pose estimation fails and the frame is skipped. The VO
pipeline logs a warning and continues.

**No loop closure.** This library performs frame-to-frame visual odometry only.
It does not maintain a map of 3D landmarks, optimize accumulated drift, or detect
revisited locations.

**Static scenes.** Low-texture or featureless images may fail to produce enough
matches. SuperPoint works best on scenes with distinct corners and edges.

## Project Structure

```
slam_dnn/
  __init__.py          Public API exports
  camera.py            PinholeCamera, K_from_fov
  config.py            VOConfig dataclass
  eval.py              Umeyama alignment, APE/RTE metrics
  exceptions.py        TrackingLostError
  export.py            KITTI and TUM format I/O
  features.py          SuperPoint wrapper
  io.py                FrameLoader (directories and videos)
  matching.py          MatcherBase, LightGlueMatcher, ClassicMatcher
  pose.py              Essential matrix and homography fallback
  trajectory.py        TrajectoryAccumulator (SE3 composition)
  visualization.py     Matplotlib trajectory plotting
  vo.py                VisualOdometry facade
```

## References

- **SuperPoint**: DeTone et al., "SuperPoint: Self-Supervised Interest Point
  Detection and Description" (CVPRW 2018)
- **LightGlue**: Lindenberger et al., "LightGlue: Local Feature Matching at
  Light Speed" (ICCV 2023)
- **5-point algorithm**: Nister, "An Efficient Solution to the Five-Point
  Relative Pose Problem" (PAMI 2004)
- **Umeyama alignment**: Umeyama, "Least-Squares Estimation of Transformation
  Parameters Between Two Point Patterns" (PAMI 1991)

## License

MIT
