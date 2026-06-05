# Comprehensive Visual Odometry Evaluation Report (V2)

**Date**: 2026-06-06  
**Hardware Accelerations**: Apple Silicon GPU (MPS)  
**Software Stack**: PyTorch, Transformers, OpenCV, SciPy  

---

## 1. Parameter Settings & System Configuration

All evaluations were run using the [slam_dnn](file:///Users/kuang-yujeng/SLAM-DNN/slam_dnn) visual odometry pipeline on MPS. The specific configurations and parameter defaults are detailed below:

### 1.1 Core Tracker Parameters
* **Feature Extractor**: `SuperPoint` (2048 keypoints, detection threshold = `0.0005`)
  - *Exception*: XFeat configurations use the `XFeat` extractor.
* **Feature Matcher**: `Classic` (BFMatcher with ratio test = `0.75`)
  - *Exception*: XFeat configurations use the custom `XFeat` matcher. LightGlue configurations use `LightGlue` with threshold = `0.1`.
* **RANSAC Pose Estimation**:
  - Outlier threshold: `1.0` pixel
  - Confidence: `0.999`
  - Minimum matches: `20`
* **Keyframe Selector Heuristics** (Default/Dense):
  - Minimum median parallax: `8.0` pixels
  - Maximum overlap ratio: `0.85`
  - Maximum keyframe interval: `10` frames
* **Motion Model**: Constant velocity motion model with alpha (EMA smoothing) = `0.5`

### 1.2 Depth Prior Parameters
* **Depth Source**: `model` (on-the-fly estimation)
* **Depth Resolution**: `(320, 192)`
* **Scale Mode**:
  - `fixed`: Uses the absolute depth in meters directly with a fixed scale factor of `1.0` (optimal for metric anchoring).
  - `median_ratio`: Calibrates scale dynamically against the running trajectory.

---

## 2. Complete Benchmark Results

We evaluated multiple configurations across two real-world datasets: **KITTI Sequence 05** (2761 frames, outdoor driving) and the **Parking Sequence** (600 frames, outdoor circular driving).

### 2.1 KITTI Sequence 05 (2761 frames)

| Configuration | Depth Model | Scale Mode | APE RMSE (m) | RTE RMSE (m) | Umeyama Scale | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Monocular** (SuperPoint) | None | N/A | 34.45 | 3.84 | 0.8628 | **26.74** |
| **Pure Monocular** (XFeat) | None | N/A | 43.72 | 3.81 | 0.8325 | 24.21 |
| **Dynamic Scaling Relative Prior** (Buggy) | Small | `median_ratio` | 150.97 | 4.60 | 0.6638 | 15.06 |
| **Dynamic Scaling Metric Prior** (Fixed) | Small | `median_ratio` | 106.27 | 4.64 | 1.5301 | 15.06 |
| **Fixed Metric Prior** | Small | `fixed` | 20.60 | 0.85 | 0.7254 | 15.80 |
| **Fixed Metric Prior (Best)** | **Large** | **`fixed`** | **10.58** | **0.63** | **0.7177** | **8.67** |

### 2.2 Parking Sequence (599 frames)

| Configuration | Depth Model | Scale Mode | APE RMSE (m) | RTE RMSE (m) | Umeyama Scale | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Monocular** (SuperPoint) | None | N/A | **0.97** | **0.40** | **0.1506** | **27.49** |
| **Fixed Metric Prior** | Small | `fixed` | 6.52 | 2.78 | 0.2610 | 17.64 |
| **Fixed Metric Prior** | **Large** | **`fixed`** | **2.72** | **1.23** | **0.2693** | **8.79** |

---

## 3. Deep-Dive Performance Analysis

### 3.1 KITTI 05 Analysis: The Power of Fixed Anchoring
* The transition from **Pure Monocular** to **Fixed Metric Prior (Large)** resulted in an **83.6% reduction in local pose drift (RTE)** and slashed APE from 34.45m to **10.58m**.
* **Dynamic Scaling (`median_ratio`) vs Fixed Scale**:
  As shown in the table, running the corrected Small model with `median_ratio` resulted in an APE of **106.27m**, compared to **20.60m** in `fixed` scale mode. This confirms that dynamic calibration creates a feedback loop that propagates and amplifies tracking drift, whereas `fixed` mode anchors the camera trajectory to the model's global absolute scale reference.

### 3.2 Parking Dataset Analysis: Scale Factor Mismatch
* Interestingly, on the **Parking** dataset, **Pure Monocular** achieves a lower APE (0.97m) after 7-DoF Sim(3) alignment compared to the Fixed Metric Prior runs.
* **Why this occurs**:
  The Umeyama Scale for the Fixed Metric Prior runs is **~0.27**, meaning the depth model's absolute scale is about **3.7x larger** than the ground truth coordinates of the Parking dataset.
  - In Pure Monocular, the trajectory shape is accurate and gets scaled globally during evaluation (Umeyama scale = `0.15`), giving a low error.
  - In Fixed Metric Prior, because the scale is locked at `1.0`, the system is forced to track at the larger model scale. This mismatch causes slight geometric distortions during PnP back-projections, leading to higher trajectory errors.
* **Model Comparison**: Even with the scale mismatch, the **Large model** (2.72m) still outperforms the **Small model** (6.52m) by **2.4x**, verifying that the high-quality depth maps of the Large model translate directly to better relative camera geometry.

---

## 4. Key Takeaways and SOTA Recipes
1. **Best Configuration (Accuracy)**:
   - Extractor: `SuperPoint`
   - Matcher: `Classic` (Ratio Test)
   - Depth Prior: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`
   - Scale Mode: `fixed` (Scale Factor = `1.0`)
   - Device: `mps`
2. **Real-time Deployment**:
   If 8.7 FPS is too slow for deployment, substituting the **Small** metric model increases speed to **15.8 FPS** while maintaining strong accuracy (APE RMSE 20.60m, RTE RMSE 0.85m).
3. **Dataset Portability Warning**:
   Before deploying a fixed metric prior on a new dataset, verify that the ground truth coordinate scale matches the metric depth network's scale (e.g. both in physical meters). If the target dataset uses arbitrary units or has scale offsets, a calibration factor must be applied to `--depth-scale-factor` instead of `1.0`.
