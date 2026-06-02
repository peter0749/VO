# Baseline Comparison: slam_dnn vs minislam

## Introduction

This document compares two implementations of the same visual odometry pipeline.
The first is **slam_dnn**, our library built around learned feature extraction and
neural matching. The second is **minislam**, a compact monocular VO system by
Marko Elez that uses classical OpenCV primitives throughout.

Both systems follow the same four-stage pipeline: detect features, match them
between consecutive frames, estimate relative camera pose, and accumulate those
poses into a global trajectory. The math is the same. The engineering choices are
what differ.

### Scope

We compare only the four core pipeline stages listed above. Loop closure,
bundle adjustment, and map management are intentionally excluded. slam_dnn does
not implement loop closure. minislam does, but that capability falls outside the
scope of this comparison.

### How to Read This Document

Each section covers one pipeline stage and contains three parts:

1. **Our Implementation** describes how slam_dnn handles the stage, with file
   references and code snippets you can cross-check in the source tree.
2. **Baseline** does the same for minislam.
3. **Comparison Table** summarizes the design differences side by side, with a
   rationale column explaining why each choice was made.

The goal is pedagogical. If you are a student trying to understand why there are
multiple ways to build a VO system, these tables show the trade-offs concretely.


---

## 1. Feature Extraction

Feature extraction answers one question: given an image, where are the
interesting points and how do we describe them?

### Our Implementation (slam_dnn)

- **File:** `slam_dnn/features.py`
- **Key class:** `SuperPointExtractor` (line 8)
- **Key function:** `SuperPointExtractor.extract(image)` (line 51)

slam_dnn uses the SuperPoint neural network for both detection and description.
SuperPoint is a VGG-based CNN trained in a self-supervised manner on synthetic
images with homographic augmentation. It produces 256-dimensional float
descriptors that are L2-normalized to unit length.

The `extract` method handles the full pipeline: convert uint8 input to a float
tensor, run a single forward pass, then extract keypoints, descriptors, and
confidence scores from the output dict. A safety re-normalization step ensures
the descriptors remain unit vectors after the float32 casting.

```python
# slam_dnn/features.py, lines 78-99
img = image.astype(np.float32) / 255.0

if img.ndim == 2:
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
elif img.ndim == 3:
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

tensor = tensor.to(self.device)

with torch.no_grad():
    pred = self.superpoint({"image": tensor})

keypoints = pred["keypoints"][0].cpu().numpy()
descriptors = pred["descriptors"][0].cpu().numpy()
scores = pred["keypoint_scores"][0].cpu().numpy()
```

The model is initialized with configurable parameters for non-maximum suppression
radius, maximum keypoints, and detection threshold. These are passed to the
LightGlue package's SuperPoint wrapper at lines 45-49.

```python
# slam_dnn/features.py, lines 45-49
self.superpoint = SuperPoint(
    max_num_keypoints=max_keypoints,
    detection_threshold=conf_thresh,
    nms_radius=nms_radius,
).to(self.device).eval()
```

SuperPoint runs on GPU when available, with automatic device selection at
lines 41-43. The `auto` device mode checks for CUDA and falls back to CPU.

### Baseline (minislam)

- **File:** `baselines/minislam/src/minislam/features.py`
- **Key class:** `FeatureManager` (line 5)
- **Key function:** `FeatureManager.detect(frame)` (line 9)

minislam splits feature detection and description into two separate steps using
classical OpenCV algorithms. Detection uses Shi-Tomasi corner detection via
`cv2.goodFeaturesToTrack`. Description uses ORB (Oriented FAST and Rotated
BRIEF), which produces 32-byte binary descriptors.

```python
# baselines/minislam/src/minislam/features.py, lines 5-11
class FeatureManager:
  def __init__(self):
    self.feature_extractor = cv2.ORB_create()

  def detect(self, frame):
    pts = cv2.goodFeaturesToTrack(frame, 2000,
        qualityLevel=0.01, minDistance=3)
    return [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=5) for p in pts]
```

The `detect` method finds up to 2000 corners with a quality threshold of 0.01
and minimum spacing of 3 pixels. Each detected corner gets wrapped in a
`cv2.KeyPoint` with a fixed size of 5. The `compute` method (line 13) then runs
ORB on those keypoints to produce binary descriptors.

```python
# baselines/minislam/src/minislam/features.py, lines 13-16
def compute(self, frame, kps):
    kps, des = self.feature_extractor.compute(frame, kps)
    self.kps, self.des = np.array(kps), np.array(des)
    return (self.kps, self.des)
```

The `detect_and_compute` convenience method (line 18) chains both steps
together. All feature state lives on the `FeatureManager` instance as mutable
attributes.

### Comparison Table

| Aspect | slam_dnn | minislam | Rationale |
|--------|----------|----------|-----------|
| Detector | SuperPoint CNN (VGG backbone) | Shi-Tomasi corners (`goodFeaturesToTrack`) | SuperPoint learns what corners matter from data; Shi-Tomasi uses a hand-crafted corner response function |
| Descriptor type | 256-d float vector | 32-byte binary string (ORB) | Float descriptors carry richer information; binary descriptors prioritize speed and memory |
| Descriptor dimension | 256 | 256 bits (32 bytes) | SuperPoint: 256 continuous values. ORB: 256 binary bits packed into 32 bytes |
| Distance metric | L2 (Euclidean) | Hamming | L2 is natural for continuous vectors; Hamming counts differing bits for binary strings |
| Max keypoints | Configurable (default 1024) | Fixed at 2000 | slam_dnn trades count for descriptor quality; minislam uses more keypoints to compensate for weaker descriptors |
| Detection + description | Fused in one forward pass | Two separate steps | Single CNN pass is more efficient; split design is simpler to understand |
| Hardware acceleration | GPU/MPS/CPU (auto-detect) | CPU only | Neural models benefit from GPUs; classical OpenCV works everywhere without setup |
| Confidence scores | Per-keypoint score returned | No per-keypoint score | Scores enable downstream filtering and weighting of matches |
| Normalization | L2-normalized descriptors, re-normalized after casting | No descriptor normalization | ORB binary descriptors don't need normalization; float descriptors do |


---

## 2. Feature Matching

Once features are extracted from two consecutive frames, the matcher pairs them
up. The quality of these pairs directly affects pose estimation accuracy.

### Our Implementation (slam_dnn)

- **File:** `slam_dnn/matching.py`
- **Key classes:** `LightGlueMatcher` (line 58), `ClassicMatcher` (line 160)
- **Key interface:** `MatcherBase` abstract class (line 9)

slam_dnn provides two matchers behind a common interface, selectable via the
`create_matcher` factory function (line 37). The strategy pattern lets
`VisualOdometry` swap matchers without code changes.

**LightGlueMatcher** wraps the LightGlue neural matcher, which was co-designed
with SuperPoint. It takes keypoints and descriptors from both frames, runs them
through a graph neural network with attention layers, and returns filtered
matches with confidence scores.

```python
# slam_dnn/matching.py, lines 126-137
with torch.no_grad():
    matches_dict = self.matcher(
        {"image0": input0, "image1": input1}
    )

matches = matches_dict["matches"][0]

valid = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
matches = matches[valid]
```

LightGlue uses a -1 sentinel to mark unmatched keypoints. The matcher filters
those out at line 136 and gathers matched point coordinates by index at lines
140-144. Confidence scores come from `matching_scores0` in the output dict.

**ClassicMatcher** provides a non-neural fallback using OpenCV's brute-force
matcher with Lowe's ratio test. It builds a `BFMatcher` with L2 distance at
line 182, runs k-NN matching with k=2, and keeps matches where the best
distance is less than 0.75 times the second-best.

```python
# slam_dnn/matching.py, lines 207-217
desc0 = feats0["descriptors"].astype(np.float32)
desc1 = feats1["descriptors"].astype(np.float32)

matches = self.matcher.knnMatch(desc0, desc1, k=2)

good_matches = []
for m, n in matches:
    if m.distance < self.ratio * n.distance:
        good_matches.append(m)
```

Both matchers return the same output format: a dict with `points0`, `points1`,
`scores`, and `indices`. This uniform interface means the rest of the pipeline
never needs to know which matcher produced the results.

### Baseline (minislam)

- **File:** `baselines/minislam/src/minislam/features.py`
- **Key function:** `FeatureManager.get_matches(cur_kps, ref_kps, cur_des, ref_des)` (line 22)

minislam uses a single matching approach: brute-force with Hamming distance and
Lowe's ratio test. The matcher is created fresh on every call rather than being
reused across frames.

```python
# baselines/minislam/src/minislam/features.py, lines 22-37
def get_matches(self, cur_kps, ref_kps, cur_des, ref_des):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(ref_des, cur_des, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    assert len(good) > 8

    ref_mask = np.array([x.queryIdx for x in good])
    cur_mask = np.array([x.trainIdx for x in good])

    ref_pts = np.array([x.pt for x in ref_kps[ref_mask]])
    cur_pts = np.array([x.pt for x in cur_kps[cur_mask]])

    return (ref_pts, cur_pts)
```

The method asserts that at least 9 matches survive the ratio test (line 28).
If fewer survive, the program crashes with an `AssertionError`. The matched
points are extracted by indexing into the original keypoint arrays and returned
as a pair of numpy arrays.

Matching lives on the `FeatureManager` class rather than having its own module.
This keeps the code compact but mixes feature extraction and matching concerns
into one class.

### Comparison Table

| Aspect | slam_dnn | minislam | Rationale |
|--------|----------|----------|-----------|
| Matcher type | LightGlue (neural, default) or ClassicMatcher (BF) | Brute-force Hamming only | Neural matching learns contextual relationships; BF is simpler and faster for binary descriptors |
| Distance metric (default) | L2 (both matchers) | Hamming | slam_dnn uses float descriptors requiring L2; minislam uses binary ORB requiring Hamming |
| Ratio test threshold | ClassicMatcher: 0.75 (configurable) | 0.75 (hardcoded) | Same heuristic, but slam_dnn exposes it as a parameter |
| Match filtering | LightGlue: confidence threshold; Classic: ratio test | Ratio test only, asserts > 8 matches | slam_dnn returns gracefully on too few matches; minislam crashes with AssertionError |
| Score output | Per-match confidence scores | No scores | Scores enable downstream weighting or filtering in pose estimation |
| Return format | Dict: `{points0, points1, scores, indices}` | Tuple: `(ref_pts, cur_pts)` | Dict is more extensible; tuple is more compact |
| Matcher lifecycle | Instantiated once, reused per frame | Created fresh each call (`BFMatcher()` at line 23) | Reuse avoids repeated allocation; fresh creation avoids stale state |
| Interface design | `MatcherBase` ABC with strategy pattern | Method on `FeatureManager` | Separation of concerns vs. simplicity. Strategy pattern allows matcher swapping |
| Hardware | LightGlue supports GPU; Classic is CPU-only | CPU only | Neural matching benefits from GPU acceleration |


---

## 3. Pose Estimation

Pose estimation recovers the relative rotation and translation between two
camera views from matched point pairs. Both systems use the Essential matrix
with RANSAC, but they differ in coordinate handling and error recovery.

### Our Implementation (slam_dnn)

- **File:** `slam_dnn/pose.py`
- **Key function:** `estimate_essential(points0, points1, K, ...)` (line 6)

slam_dnn's pose estimation is a standalone function, not a class method. It
takes matched points in pixel coordinates, normalizes them to camera coordinates
manually, estimates the Essential matrix, and decomposes it into R and t.

The normalization step is explicit. It subtracts the principal point and divides
by focal length, converting pixel coordinates to normalized camera coordinates.

```python
# slam_dnn/pose.py, lines 44-48
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

points0_norm = (points0 - np.array([cx, cy])) / np.array([fx, fy])
points1_norm = (points1 - np.array([cx, cy])) / np.array([fx, fy])
```

The Essential matrix is estimated with OpenCV's 5-point algorithm and RANSAC
at lines 52-60. The RANSAC threshold is scaled by the mean focal length to
convert from pixel space to normalized coordinates.

```python
# slam_dnn/pose.py, lines 52-60
f_mean = (fx + fy) / 2.0
E, mask = cv2.findEssentialMat(
    points0_norm,
    points1_norm,
    cameraMatrix=np.eye(3),
    method=cv2.RANSAC,
    prob=conf,
    threshold=ransac_thresh / f_mean,
)
```

After recovering the pose via `cv2.recoverPose` (lines 74-76), the function
validates that R is a proper rotation by checking that its determinant is close
to +1 (line 84). The translation is returned as a unit-norm vector since
monocular VO cannot determine scale.

slam_dnn also includes a pure-rotation fallback. The `detect_pure_rotation`
function (line 99) compares Homography inlier counts against Essential matrix
inlier counts. When the Essential matrix degenerates (pure rotation case),
`estimate_essential_or_homography` (line 152) falls back to Homography
decomposition and recovers rotation alone via `R = K_inv @ H @ K`.

```python
# slam_dnn/pose.py, lines 200-213
K_inv = np.linalg.inv(K)
R_unnormalized = K_inv @ H @ K

U, _, Vt = np.linalg.svd(R_unnormalized)
R = U @ Vt

if np.linalg.det(R) < 0:
    R = -R

t = np.zeros(3)
```

### Baseline (minislam)

- **File:** `baselines/minislam/src/minislam/odometry.py`
- **Key function:** `VisualOdometry.estimate_pose(use_fundamental_matrix=False)` (line 99)

minislam's pose estimation is a method on the `VisualOdometry` class that
operates on the internally stored matched keypoints. By default, it
denormalizes the matched points from image coordinates to camera coordinates
using the camera's inverse intrinsic matrix, then estimates the Essential matrix.

```python
# baselines/minislam/src/minislam/odometry.py, lines 106-118
cur_kps = self.camera.denormalize_pts(self.cur_matched_kps)
ref_kps = self.camera.denormalize_pts(self.ref_matched_kps)
E, mask = cv2.findEssentialMat(
    cur_kps,
    ref_kps,
    focal=1,
    pp=(0.0, 0.0),
    method=cv2.RANSAC,
    prob=0.999,
    threshold=0.003,
)

_, R, t, mask = cv2.recoverPose(E, cur_kps, ref_kps,
    focal=1, pp=(0.0, 0.0))
```

The denormalization uses the camera's `Kinv` matrix stored on the `Camera`
object (`baselines/minislam/src/minislam/camera.py`, lines 15 and 25-27).
Points are converted to homogeneous coordinates, multiplied by `Kinv`, and
stripped back to 2D. After that, `findEssentialMat` is called with `focal=1`
and `pp=(0,0)` since the points are already in normalized coordinates.

minislam also offers an alternative path through the Fundamental matrix
(lines 100-104). When `use_fundamental_matrix=True`, it estimates F directly
in pixel coordinates, then constructs E as `K^T @ F @ K`.

```python
# baselines/minislam/src/minislam/odometry.py, lines 100-104
if use_fundamental_matrix:
    cur_kps = self.cur_matched_kps
    ref_kps = self.ref_matched_kps
    F, mask = cv2.findFundamentalMat(cur_kps, ref_kps, method=cv2.RANSAC)
    E = np.dot(self.camera.K.T, F).dot(self.camera.K)
```

The method returns R, t, and an array of inlier indices (line 121). Unlike
slam_dnn, there is no validity check on the rotation matrix determinant and
no pure-rotation fallback.

### Comparison Table

| Aspect | slam_dnn | minislam | Rationale |
|--------|----------|----------|-----------|
| Normalization | Manual: subtract cx,cy, divide by fx,fy | Via `camera.denormalize_pts()` using Kinv matrix | Same math, different packaging. slam_dnn is explicit; minislam hides it in the Camera class |
| Essential matrix call | `findEssentialMat` with identity camera matrix (normalized coords) | `findEssentialMat` with focal=1, pp=(0,0) (normalized coords) | Functionally equivalent. Both pass pre-normalized points with identity intrinsics |
| RANSAC threshold | `ransac_thresh / f_mean` (pixel threshold scaled to normalized coords) | 0.003 hardcoded (in normalized coords) | slam_dnn accepts pixel-space thresholds scaled to normalized space; minislam uses a fixed normalized-space value |
| RANSAC confidence | Configurable (default 0.999) | Hardcoded at 0.999 | slam_dnn exposes this for experimentation |
| Rotation validation | Checks `abs(det(R) - 1.0) < 0.01` | No check | Catches degenerate decompositions early |
| Pure rotation handling | Homography fallback with SVD orthogonalization | None | Essential matrix degenerates under pure rotation; Homography still works |
| Minimum matches check | Returns None if < 8 matches (line 39) | Asserts > 8 in matching step, not in pose estimation | slam_dnn fails gracefully; minislam crashes earlier in the pipeline |
| Return type | `(R, t, inlier_mask)` or `None` | `(R, t, inlier_indices)` | slam_dnn uses boolean mask; minislam uses integer index array |
| Translation norm | Unit-normalized (inherent to `recoverPose`) | Unit-normalized (inherent to `recoverPose`) | Both inherit unit-norm from OpenCV; scaling is handled in trajectory stage |
| Alternative paths | `estimate_essential_or_homography` for rotation fallback | `use_fundamental_matrix` flag for F-matrix path | Different fallback strategies for degenerate cases |


---

## 4. Trajectory Accumulation

The final stage composes frame-to-frame relative poses into a global camera
trajectory. This is where both systems implement SE(3) composition, and the
similarities are striking.

### Our Implementation (slam_dnn)

- **File:** `slam_dnn/trajectory.py`
- **Key class:** `TrajectoryAccumulator` (line 84)
- **Key function:** `TrajectoryAccumulator.add_pose(R, t)` (line 128)

`TrajectoryAccumulator` maintains a running rotation and translation in their
raw forms rather than storing only 4x4 matrices. The `add_pose` method takes a
relative rotation R and translation t from the pose estimator, normalizes the
translation to unit length, and composes it into the global state.

```python
# slam_dnn/trajectory.py, lines 142-163
t = np.asarray(t, dtype=np.float64).flatten()
R = np.asarray(R, dtype=np.float64)

t_norm = np.linalg.norm(t)
if t_norm > 1e-8:
    t_unit = t / t_norm
else:
    t_unit = t

self._current_t = self._current_t + self.scale * (self._current_R @ t_unit)
self._current_R = self._current_R @ R

T_current = pose_Rt(self._current_R, self._current_t)
self.poses.append(T_current)
```

The translation update is the key line. It rotates the unit translation into
the global frame via `R_global @ t_unit`, scales it, and adds it to the
running global translation. The rotation update is straightforward matrix
multiplication.

The class also handles pure-rotation cases gracefully: if the translation norm
is below 1e-8 (lines 148-151), it skips normalization and adds the translation
as-is (which will be near zero). This prevents division by zero when the
camera rotates in place.

Helper functions `pose_Rt` (line 5) and `compose_pose` (line 25) are provided
as standalone utilities for building and composing 4x4 SE(3) matrices.

The `get_positions` method (line 174) extracts camera centers from the
accumulated poses using the formula `C_world = -R^T @ t`. This converts from
the world-to-camera representation stored internally to camera positions in
world coordinates.

### Baseline (minislam)

- **File:** `baselines/minislam/src/minislam/odometry.py`
- **Key function:** SE3 composition in `VisualOdometry.process_frame(img, frame_id)` (line 51)

minislam accumulates trajectory inside `process_frame` rather than in a
dedicated class. The composition happens at lines 71-76 of `odometry.py`.

```python
# baselines/minislam/src/minislam/odometry.py, lines 71-76
self.cur_t = self.cur_t + (self.scale * self.cur_R.dot(t))
self.cur_R = self.cur_R.dot(R)

self.translations.append(self.cur_t)
pose = pose_Rt(self.cur_R, self.cur_t)
self.poses.append(pose)
```

The math is the same as slam_dnn: rotate the relative translation by the
current rotation, scale it, and add it to the global translation. Then compose
rotations and build a 4x4 pose.

However, minislam does not normalize the translation to unit length before
composition. The `t` returned by `cv2.recoverPose` is already unit-norm, so the
result is correct. But this relies on an implicit assumption about the upstream
pose estimator's behavior.

The scale factor is hardcoded at 0.8 (`odometry.py` line 14). slam_dnn
defaults to 1.0 but exposes scale as a constructor parameter.

The `pose_Rt` helper is defined in `baselines/minislam/src/minislam/util.py`
(line 10) and does the same thing as slam_dnn's version: build a 4x4 identity,
fill in R and t.

```python
# baselines/minislam/src/minislam/util.py, lines 10-14
def pose_Rt(R, t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t.ravel()
  return ret
```

Trajectory state lives directly on the `VisualOdometry` object as
`self.translations`, `self.poses`, `self.cur_R`, and `self.cur_t`. There is no
reset method. To start a new trajectory, you create a new `VisualOdometry`
instance.

### Comparison Table

| Aspect | slam_dnn | minislam | Rationale |
|--------|----------|----------|-----------|
| Architecture | Dedicated `TrajectoryAccumulator` class | Inline in `process_frame` method | Separation of concerns vs. compactness. Dedicated class is testable in isolation |
| Translation normalization | Explicit: normalize to unit length before composition (line 149) | Implicit: relies on `recoverPose` returning unit-norm t | Explicit is safer. If upstream changes, slam_dnn still works correctly |
| Scale factor | Configurable (default 1.0) | Hardcoded at 0.8 (line 14) | Both deal with monocular scale ambiguity differently. Configurable allows per-dataset tuning |
| Zero-translation handling | Guards against norm < 1e-8, skips normalization (line 148) | No guard | slam_dnn handles pure-rotation frames without crashing |
| Pose storage | `self.poses` list of 4x4 matrices | `self.poses` list of 4x4 matrices + separate `self.translations` list | Redundant storage in minislam; slam_dnn derives positions from poses on demand |
| Camera center extraction | `get_positions()` computes `-R^T @ t` per pose | Not provided; `self.translations` stores t directly (not camera center) | t and camera center are different things. `C = -R^T @ t` gives the actual camera position |
| Reset capability | `reset()` method clears all state (line 195) | Must create new instance | Reset enables reuse in batch processing |
| Data types | Explicit `float64` throughout | Default numpy precision | slam_dnn pins precision for reproducibility |
| Export methods | `save(filepath, format)` with KITTI/TUM support (line 213) | None on trajectory object | slam_dnn couples export to the accumulator for convenience |


---

## Conclusion

### Key Architectural Similarities

Both systems solve the same problem with the same fundamental pipeline.
Frame-to-frame feature matching, Essential matrix decomposition via 5-point
RANSAC, and SE(3) composition for trajectory accumulation are present in both
codebases. The math is identical: the translation update formula
`t_global = t_global + scale * (R_global @ t_rel)` appears in both systems
with only syntactic differences.

Both use `cv2.findEssentialMat` and `cv2.recoverPose` as their core pose
estimation engine. Both handle the monocular scale ambiguity by working with
unit-norm translations. Both store trajectories as lists of 4x4 SE(3) matrices.

### Key Differences

The most significant difference is the feature pipeline. slam_dnn uses learned
features (SuperPoint) with learned matching (LightGlue), producing 256-d float
descriptors matched via attention-based neural networks on GPU. minislam uses
hand-crafted features (Shi-Tomasi + ORB) with classical brute-force matching
and Hamming distance on CPU.

slam_dnn is more defensive. It checks rotation validity after decomposition,
guards against zero translations, returns `None` instead of crashing on too few
matches, and provides a Homography fallback for pure rotation. minislam uses
assertions and lets failures propagate as exceptions.

slam_dnn separates concerns more aggressively. Each pipeline stage lives in its
own module with a focused class or function. Matching uses the strategy pattern
behind an abstract base class. Trajectory accumulation has its own class with
reset and export capabilities. minislam concentrates more functionality into
fewer classes, keeping the total codebase small.

Scale handling differs: minislam hardcodes 0.8 (likely tuned for KITTI), while
slam_dnn defaults to 1.0 and lets the user configure it. Both acknowledge the
monocular scale ambiguity but handle it at different points in the pipeline.

### Educational Takeaways

For students studying visual odometry, comparing these two systems reveals
several lessons:

**There is no single right way to build a VO pipeline.** Both systems produce
valid trajectories on the same datasets. The engineering choices reflect
different priorities: minislam optimizes for simplicity and small code size;
slam_dnn optimizes for extensibility and robustness.

**Learned features improve robustness at the cost of complexity.** SuperPoint
and LightGlue handle challenging scenes (low texture, motion blur, lighting
changes) better than ORB, but they require GPU hardware and larger dependencies.

**Defensive programming matters in geometric pipelines.** A single degenerate
frame (pure rotation, low texture, motion blur) can crash an unprotected system.
Graceful degradation, from returning `None` to Homography fallback, is what
separates a research prototype from a deployable system.

**The math is the same regardless of implementation style.** Essential matrix
decomposition, RANSAC, and SE(3) composition are universal. Whether you organize
them into three classes or one doesn't change the trajectory you get out. The
organization affects how easy the code is to test, extend, and debug.
