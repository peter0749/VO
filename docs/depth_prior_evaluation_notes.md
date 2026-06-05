# Depth-Prior Visual Odometry Integration and Optimization Notes

This document records the discovery, bug fixes, scale calibration analysis, and benchmarking results of incorporating monocular deep depth priors into the visual odometry pipeline.

---

## 1. Executive Summary
Integrating deep metric depth models (e.g., Depth Anything V2 Metric) with sparse feature tracking (SuperPoint + Classic/LightGlue Matcher) provides a substantial boost in visual odometry accuracy and scale consistency. On the complete KITTI 05 sequence (2761 frames), we achieved an **83.6% reduction in local pose drift (RTE RMSE)** and cut the absolute trajectory error (APE RMSE) by more than two-thirds (from 34.45m down to **10.58m**) compared to the monocular baseline.

This performance was unlocked by:
1. Fixing a critical bug that inverted depth maps from metric models.
2. Replacing dynamic scale calibration (`median_ratio`) with a **Fixed Metric Anchoring** strategy to break scale drift feedback loops.
3. Scaling up the depth model to the **Large (ViT-L)** version for high-fidelity absolute depth maps.

---

## 2. Bug Analysis: Metric Depth Inversion
Prior to this fix, the VO pipeline assumed all deep depth networks estimated *relative disparity-like* values (where larger values mean closer objects) and unconditionally applied the inversion formula:
$$\text{depth} = \frac{1.0}{\text{disparity} + \epsilon}$$

### The Issue
For relative models (e.g., `LiheYoung/depth-anything-small-hf`), this is correct. However, for dedicated metric models (e.g., `depth-anything/depth-anything-v2-metric-outdoor-small-hf`), the model outputs absolute depth in meters directly. Applying the inversion turned the geometry "inside out" (mapping $80\text{ m} \to 0.0125\text{ m}$ and $7\text{ m} \to 0.14\text{ m}$), causing PnP and triangulation to fail.

### The Fix
In [depth.py](file:///Users/kuang-yujeng/SLAM-DNN/slam_dnn/depth.py#L127-L143), we introduced an automatic checkpoint detector:
```python
self.is_metric = "metric" in model_name.lower()
```
During estimation, the pipeline bypasses the inversion for metric models:
```python
if self.is_metric:
    depth = np.maximum(pred_resized, 0.0)
else:
    depth = 1.0 / (np.maximum(pred_resized, 0.0) + 1e-6)
```

---

## 3. Scale Calibration Analysis: Dynamic vs. Fixed Scaling

### The Dynamic Scaling Feedback Loop (`median_ratio`)
In relative depth mode, or under unanchored scales, the scale factor $s$ is updated frame-by-frame by matching predicted depths against the estimated 3D map:
$$s = \text{median}\left(\frac{d_{\text{trajectory}}}{d_{\text{predicted}}}\right)$$
While mathematically sound for short sequences, this introduces a **positive feedback loop of scale drift** over long sequences:
1. Small accumulation errors drift the trajectory scale.
2. The drifted trajectory distorts the calculated scale factor $s$.
3. The distorted scale factor propagates drifted depths into new 3D map points.
4. The distorted map points further amplify the trajectory drift.

This explains why, in long sequences, the dynamic scale factor fluctuates dramatically (dropping from `260.0` in early frames to `0.6` in later frames) and degrades performance.

### The Fixed Anchoring Solution
Since the metric model yields absolute meters, we can completely disable dynamic scaling by setting the scale mode to `fixed` and the scale factor to `1.0`. The model outputs act as a **global scale anchor**, resolving the feedback loop and keeping the scale stable throughout the 2761 frames.

---

## 4. Benchmark Results: KITTI Sequence 05 (2761 frames)

Evaluated on Apple Silicon GPU (`device mps`):

| Pipeline Configuration | APE RMSE (m) | APE Mean (m) | RTE RMSE (m) | RTE Mean (m) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure Monocular** (SuperPoint + Classic) | 34.45 | 30.13 | 3.84 | 2.39 | **26.74** |
| **Pure Monocular** (XFeat + XFeat) | 43.72 | 38.90 | 3.81 | 2.45 | 24.21 |
| **Dynamic Scaling Relative Prior** (Buggy) | 150.97 | 133.58 | 4.60 | 4.49 | 15.06 |
| **Fixed Metric Prior (Small Model)** | 20.60 | 18.44 | 0.85 | 0.78 | 15.80 |
| **Fixed Metric Prior (Large Model)** | **10.58** | **9.77** | **0.63** | **0.57** | 8.67 |

### Visualizations Output Paths
* **Current Best (Large Model) Trajectory Plot**: [trajectory_comparison_large_metric.png](file:///Users/kuang-yujeng/SLAM-DNN/eval/reports/trajectory_comparison_large_metric.png)
* **Current Best (Large Model) Side Plot**: [trajectory_comparison_side_large_metric.png](file:///Users/kuang-yujeng/SLAM-DNN/eval/reports/trajectory_comparison_side_large_metric.png)
* **Current Best (Large Model) Error Plot**: [error_comparison_large_metric.png](file:///Users/kuang-yujeng/SLAM-DNN/eval/reports/error_comparison_large_metric.png)
* **Previous Best (Small Model) Trajectory Plot**: [trajectory_comparison_small_metric.png](file:///Users/kuang-yujeng/SLAM-DNN/eval/reports/trajectory_comparison_small_metric.png)
* **XFeat No-Prior Trajectory Plot**: [trajectory_comparison_xfeat_no_prior.png](file:///Users/kuang-yujeng/SLAM-DNN/eval/reports/trajectory_comparison_xfeat_no_prior.png)


---

## 5. Parameter Tuning and Ablation Studies (100 Frames)

To search for the optimal configuration, we conducted ablation studies on a 100-frame slice using the Apple Silicon GPU (MPS) under the **Fixed Metric Prior** configuration:

### 5.1 Matcher Ablation: Classic vs. LightGlue
* **Classic Matcher** (Ratio Test):
  - APE RMSE: **0.2505 m**
  - Matching Time: **0.49s** (~16.5 FPS overall)
* **LightGlue Matcher** (Neural Network):
  - APE RMSE: **0.2540 m** (tested on 50 frames: 0.1240m vs 0.1195m for Classic)
  - Matching Time: **21.36s** (~2.0 FPS overall, 10x slower)
* **Conclusion**: The neural matcher (LightGlue) is 10x slower and offers no accuracy gains over the classic matcher for this sequence. **Classic Matcher is optimal**.

### 5.2 Extractor Ablation: SuperPoint vs. XFeat
* **SuperPoint Extractor**:
  - APE RMSE: **0.1195 m** (50 frames)
* **XFeat Extractor**:
  - APE RMSE: **0.1884 m** (50 frames)
  - Speed: Slightly faster, but accuracy is noticeably lower.
* **Conclusion**: SuperPoint provides higher-quality keypoints and lower drift. **SuperPoint is optimal**.

### 5.3 Keyframe Density Ablation
* **Dense Keyframes** (Default: Every frame is a keyframe, 100 KF / 100 frames):
  - APE RMSE: **0.2505 m**
  - RTE RMSE: **0.2637 m**
* **Sparse Keyframes** (Modified: `--min-parallax 15.0 --max-overlap 0.5`, 79 KF / 100 frames):
  - APE RMSE: **0.3613 m**
  - RTE RMSE: **0.3450 m**
* **Conclusion**: Because the metric depth model is highly accurate, updating the depth map on every frame ensures the PnP tracking always operates with the freshest depth estimates. Restricting keyframe density increases pose error by keeping outdated depth constraints longer. **Dense Keyframes (default) is optimal**.

### 5.4 Depth Model Scale Ablation
* **Small Model** (`...-Metric-Outdoor-Small-hf`):
  - APE RMSE: **0.1195 m** (50 frames)
  - Processing Speed: **16.5 FPS**
* **Base Model** (`...-Metric-Outdoor-Base-hf`):
  - APE RMSE: **0.1365 m** (50 frames)
  - Processing Speed: **12.3 FPS**
* **Large Model** (`...-Metric-Outdoor-Large-hf`):
  - APE RMSE: **0.1109 m** (50 frames)
  - Processing Speed: **9.2 FPS**
* **Conclusion**: The **Large model** achieves the highest accuracy by capturing fine geometric details in the scene, yielding a significant accuracy improvement on the full sequence (APE RMSE from 20.60m down to 10.58m), while still operating at a real-time capable 9 FPS on MPS.

---

## 6. Execution Command Reference
To replicate the optimal fixed metric depth prior run:
```bash
python3 eval/compare.py \
  --mode full \
  --dataset kitti05 \
  --max-frames 2761 \
  --device mps \
  --baselines none \
  --use-depth-prior \
  --depth-source model \
  --depth-model-name depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf \
  --depth-scale-mode fixed \
  --depth-scale-factor 1.0
```

---

## 7. Future Exploration
To improve accuracy beyond the current benchmark, we will explore:
1. **Bundle Adjustment**: Designing a custom frontend that joint-optimizes camera poses with metric depth prior constraints in local bundle adjustment.

