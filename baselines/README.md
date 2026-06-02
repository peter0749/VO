# Baselines

External SLAM/VO systems used for comparison benchmarks against slam_dnn.

## minislam

| Field | Value |
|-------|-------|
| **Source** | https://github.com/markoelez/minislam |
| **Pinned commit** | `962096d5bb8919317cceef9c0f2f98f023d9fcf3` |
| **Author** | Marko Elez |
| **License** | No explicit license file (all rights reserved by default) |
| **Purpose** | Baseline comparison: ORB-based monocular VO with loop closure |

### What minislam does

minislam is a minimal monocular visual SLAM implementation featuring:
- ORB feature detection and matching
- Essential matrix decomposition for relative pose estimation
- Loop closure detection via descriptor similarity + geometric verification
- Real-time 3D visualization

### Setup

```bash
bash scripts/setup_baseline.sh
```

This initializes the git submodule, installs minislam's dependencies, and
verifies the wrapper is importable.

### Usage via wrapper

```python
from baselines.minislam_wrapper import run_minislam_on_kitti, check_minislam_available

if check_minislam_available():
    poses = run_minislam_on_kitti(
        data_dir="data/kitti05",
        output_dir="results/minislam",
        use_calib_intrinsics=True,
        max_frames=100,
    )
    print(f"Estimated {len(poses)} poses")
else:
    print("minislam not available - run: bash scripts/setup_baseline.sh")
```

### Important notes

- **Do NOT modify files inside `minislam/`** — it is a git submodule.
- **Do NOT import from `baselines/` in `slam_dnn/`** — package independence is required.
- The wrapper (`minislam_wrapper.py`) belongs to slam_dnn, not minislam.
- minislam requires Python >= 3.11.6 and depends on opencv-python, pygame,
  pyopengl, and pyyaml.

### Running tests

```bash
# Run baseline wrapper tests
pytest tests/test_baseline_wrapper.py -v

# Skip baseline tests (when minislam not installed)
pytest tests/ --skip-baseline
```
