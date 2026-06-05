# Comprehensive Visual Odometry (VO) Survey: Theories, Mathematics, and Paradigms

This survey provides a mathematically rigorous review of Visual Odometry (VO) paradigms, geometric motion models, optimization backends, and scale recovery techniques. It serves as a foundation for understanding classical and deep-geometric hybrid VO/SLAM pipelines.

---

## 1. Introduction & The Monocular Scale Challenge

Visual Odometry (VO) is the process of incrementally estimating the 3D ego-motion of a camera from sequential images. In monocular VO, the fundamental limitation is the **scale ambiguity**:
* An image is a 2D projection of a 3D scene.
* The projection of a 3D point $\mathbf{X} = [X, Y, Z]^T$ using camera intrinsics $\mathbf{K}$ is given by:
  $$\mathbf{x} = \pi(\mathbf{K} \mathbf{X}) = \begin{bmatrix} f_x \frac{X}{Z} + c_x \\ f_y \frac{Y}{Z} + c_y \end{bmatrix}$$
* This projection is invariant to uniform scaling. If we scale the scene by a factor $s > 0$ and the camera translation by $s$, the projected image coordinates remain identical:
  $$\mathbf{x} = \pi(\mathbf{K} \mathbf{X}) = \pi(\mathbf{K} (s\mathbf{X}))$$
* Consequently, a monocular camera cannot recover absolute scale without external priors (e.g., known camera height, IMU acceleration, or metric depth predictions). Over time, small errors in scale propagate, leading to **scale drift**.

---

## 2. Axis A: Measurement Paradigms

Visual odometry systems differ in how they define the measurement model and map pixels to geometric constraints.

```
                      Measurement Paradigms in Visual Odometry
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  Feature-Based                    Direct Methods                    Deep Hybrid
(Reprojection Error)            (Photometric Error)            (Dense Flow + Diff BA)
 - ORB-SLAM3, SIFT                - DSO, LSD-SLAM                 - DROID-SLAM, DPVO
```

### 2.1 Feature-Based Methods (Reprojection Error Minimization)
Feature-based methods extract sparse keypoints, describe them, establish correspondences, and minimize the geometric distance between projected 3D landmarks and their 2D measurements.

#### Formulation
Let $\mathbf{X}_j \in \mathbb{R}^3$ be the 3D position of landmark $j$, and $\mathbf{x}_{ij} = [u_{ij}, v_{ij}]^T$ be its measured pixel coordinate in frame $i$. The projection model is:
$$\hat{\mathbf{x}}_{ij} = \pi(\mathbf{R}_i \mathbf{X}_j + \mathbf{t}_i)$$
where $\mathbf{R}_i \in SO(3)$ and $\mathbf{t}_i \in \mathbb{R}^3$ represent the camera pose (world-to-camera). The residual is the **reprojection error**:
$$\mathbf{e}_{ij} = \mathbf{x}_{ij} - \hat{\mathbf{x}}_{ij}$$
The optimization goal is to minimize the sum of squared mahalanobis distances (reprojection errors) over all frames and points:
$$E_{\text{feature}}(\mathbf{R}, \mathbf{t}, \mathbf{X}) = \sum_{i} \sum_{j} \mathbf{e}_{ij}^T \mathbf{\Omega}_{ij} \mathbf{e}_{ij}$$
where $\mathbf{\Omega}_{ij}$ is the information matrix (inverse covariance) of the keypoint detection.

* **Keypoints**: Classical (SIFT, SURF, ORB, FAST) vs. Learned (SuperPoint, XFeat, KeypointNet).
* **Matchers**: Nearest Neighbor (BF, FLANN) vs. Learned Graph Matching (LightGlue, SuperGlue).

### 2.2 Direct Methods (Photometric Error Minimization)
Direct methods skip keypoint extraction and descriptor matching. They minimize the intensity difference of pixels corresponding to the same physical 3D point projected onto different frames.

#### Formulation
Let $I_k$ and $I_m$ be the intensity images of frame $k$ and $m$. Let $\mathbf{p} \in \mathbb{R}^2$ be a pixel in frame $k$, and $d_p$ be its depth. The corresponding 3D point is:
$$\mathbf{X} = d_p \mathbf{K}^{-1} \begin{bmatrix} \mathbf{p} \\ 1 \end{bmatrix}$$
This point projects onto frame $m$ as:
$$\mathbf{p}' = \pi\left(\mathbf{K} \left(\mathbf{R}_{mk} \mathbf{X} + \mathbf{t}_{mk}\right)\right)$$
where $\mathbf{R}_{mk}, \mathbf{t}_{mk}$ is the relative pose from frame $k$ to $m$. The **photometric error** is:
$$E_{\text{direct}}(\mathbf{R}_{mk}, \mathbf{t}_{mk}, d) = \sum_{\mathbf{p} \in \mathcal{P}} \left\| I_k(\mathbf{p}) - I_m(\mathbf{p}') \right\|^2$$
To handle auto-exposure and illumination changes, DSO (Direct Sparse Odometry) adds affine parameters $a, b$:
$$E_{\text{DSO}} = \sum_{\mathbf{p}} w_p \left\| \left(I_m(\mathbf{p}') - b_m\right) - \frac{e^{a_m}}{e^{a_k}} \left(I_k(\mathbf{p}) - b_k\right) \right\|_{\gamma}$$
where $\|\cdot\|_{\gamma}$ is the Huber robust norm.

### 2.3 Semi-Direct Methods (SVO)
Semi-direct Visual Odometry (SVO) uses direct pixel tracking for fast frame-to-frame alignment, followed by sparse feature-based bundle adjustment for refinement.

### 2.4 Deep-Geometric Hybrid Methods (SOTA)
Modern state-of-the-art systems (e.g., *DROID-SLAM*, *DPVO*) extract dense features via convolutional neural networks and build dense 4D correlation volumes (all-pairs similarities) between frames.

Instead of treating the net output as black-box poses, they use a **differentiable Bundle Adjustment layer**:
1. A GRU-based update operator predicts a dense optical flow correction (residual flow $\mathbf{r}_{ij}$).
2. The network formulates a Dense Bundle Adjustment (DBA) problem using these residual flows as weights.
3. The system solves the Gauss-Newton step inside the PyTorch computational graph, backpropagating gradients through the 3D reconstruction backend:
   $$\mathbf{H} \Delta \mathbf{\xi} = \mathbf{v}$$
   where $\mathbf{H}$ is the Hessian and $\mathbf{\xi}$ are the camera poses/depths. This yields extreme robustness to low-texture areas and aggressive motion.

---

## 3. Axis B: Geometric Motion Models

Motion estimation models are defined by the dimensionality of the correspondence landmarks.

### 3.1 2D-to-2D: Epipolar Geometry
Used for bootstrapping and initialization when no 3D landmarks are yet established.

#### Mathematics
Let $\mathbf{x}_1$ and $\mathbf{x}_2$ be corresponding points in normalized image coordinates (after multiplying by $\mathbf{K}^{-1}$). They satisfy the **epipolar constraint**:
$$\mathbf{x}_2^T \mathbf{E} \mathbf{x}_1 = 0$$
where $\mathbf{E} \in \mathbb{R}^{3 \times 3}$ is the **Essential Matrix**, defined as:
$$\mathbf{E} = [\mathbf{t}]_{\times} \mathbf{R}$$
$[\mathbf{t}]_{\times}$ is the skew-symmetric matrix of the translation vector $\mathbf{t}$:
$$[\mathbf{t}]_{\times} = \begin{bmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0 \end{bmatrix}$$
* **5-point Algorithm (Nistér)**: Solves for $\mathbf{E}$ using 5 point correspondences. $\mathbf{E}$ has 5 degrees of freedom (3 for rotation, 2 for translation direction).
* **Decomposition**: Singular Value Decomposition (SVD) of $\mathbf{E} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ yields 4 mathematically valid pose solutions $(\mathbf{R}, \mathbf{t})$:
  $$\mathbf{R} = \mathbf{U} \mathbf{W} \mathbf{V}^T \quad \text{or} \quad \mathbf{R} = \mathbf{U} \mathbf{W}^T \mathbf{V}^T$$
  $$\mathbf{t} = \pm \mathbf{u}_3$$
  where $\mathbf{W} = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ and $\mathbf{u}_3$ is the third column of $\mathbf{U}$.
* **Chirality Check**: Points are triangulated using all 4 solutions. The pose that yields 3D points in front of both cameras ($Z > 0$) is chosen. Since $\mathbf{E}$ is computed up to scale, the translation is normalized: $\|\mathbf{t}\|_2 = 1.0$.

### 3.2 3D-to-2D: Perspective-n-Point (PnP)
Used to track the current camera pose relative to existing 3D map points.

#### Mathematics
Given 3D landmarks $\mathbf{X}_j \in \mathbb{R}^3$ and their 2D projections $\mathbf{u}_j \in \mathbb{R}^2$ in the current frame, PnP finds the rotation $\mathbf{R}$ and translation $\mathbf{t}$ that projects the 3D points onto their 2D measurements.
* **Minimal Solvers**: P3P (requires 3 points), EPnP (Efficient PnP, $O(N)$ linear solver).
* **RANSAC**: Integrated with solvers to filter out outlier matches.
* **Optimization**: The pose is refined by minimizing the reprojection error of inliers:
  $$\mathbf{R}^*, \mathbf{t}^* = \arg\min_{\mathbf{R}, \mathbf{t}} \sum_{j} \left\| \mathbf{u}_j - \pi(\mathbf{R} \mathbf{X}_j + \mathbf{t}) \right\|^2$$
  This step is known as **Pose-only Bundle Adjustment**.

### 3.3 3D-to-3D: ICP & Kabsch Alignment
Aligns two sets of 3D points. Used in stereo VO initialization and RGB-D tracking.
* **Kabsch-Umeyama Algorithm**: Solves for translation, rotation, and optionally scale factor $s$ by minimizing:
  $$E_{\text{ICP}}(\mathbf{R}, \mathbf{t}, s) = \sum_{j} \left\| \mathbf{Y}_j - (s \mathbf{R} \mathbf{X}_j + \mathbf{t}) \right\|^2$$
  using closed-form SVD decomposition of the cross-covariance matrix.

---

## 4. Axis C: Optimization Backends & Pose Refinement

State estimation filters out measurement noise and ensures temporal trajectory consistency.

### 4.1 Filter-Based Estimators (EKF vs. MSCKF)
* **Extended Kalman Filter (EKF)**: Tracks camera pose and 3D landmarks in a single state vector $\mathbf{x} = [\mathbf{x}_c^T, \mathbf{m}_1^T, \mathbf{m}_2^T, \dots]^T$. The covariance size grows quadratically with the number of features, limiting it to small maps.
* **Multi-State Constraint Kalman Filter (MSCKF)**: Retains a history of past camera poses in the state vector but marginalizes out the 3D landmarks. 2D measurements act as multi-state geometric constraints. This keeps the state size bounded, allowing real-time CPU performance.

### 4.2 Non-linear Least-Squares Optimization (Bundle Adjustment)
Bundle Adjustment (BA) is formulated as a non-linear factor graph. It minimizes the joint reprojection error of poses and 3D points:

```
                  Factor Graph Representation (Bundle Adjustment)
                      
                         [Pose 1] ──── [Pose 2] ──── [Pose 3]
                            │            │             │
                            ▼            ▼             ▼
                       (Reproj Err) (Reproj Err)  (Reproj Err)
                            │            │             │
                            └────────────┼─────────────┘
                                         ▼
                                   [3D Landmark]
```

#### Pose-only Bundle Adjustment vs. Joint Bundle Adjustment

| Metric | Pose-only Bundle Adjustment | Joint Bundle Adjustment (Local BA) |
|---|---|---|
| **Optimized Variables** | Camera Poses only ($\mathbf{R}_i, \mathbf{t}_i$). | Camera Poses ($\mathbf{R}_i, \mathbf{t}_i$) AND 3D Landmark Positions ($\mathbf{X}_j$). |
| **3D Points** | Kept fixed as constants. | Refined dynamically. |
| **Hessian Structure** | Small, dense matrix ($6N \times 6N$, where $N$ is number of optimized frames). | Large but sparse structure, solved efficiently via Schur Complement. |
| **Triangulation Noise** | **Propagates error**: Noisy depth landmarks warp the camera pose to fit bad 3D structure. | **Reduces error**: Reprojection residuals pull both camera poses and 3D coordinates toward geometric consistency. |
| **Scale Stability** | Highly susceptible to scale decay and scaling drift in monocular tracking. | Restrains scale drift by enforcing multi-view structural consistency. |

---

## 5. Metric Scale Recovery in Monocular VO

Because monocular VO operates up to scale, specialized constraints must be applied to recover physical dimensions (meters).

### 5.1 Ground Plane Estimation (Vehicle Kinematics)
For vehicle-mounted cameras, the height of the camera above the ground plane ($h$) is a constant physical prior.

#### Mathematical Steps
1. Perform 3D triangulation from 2D matches to get point cloud $\mathcal{P} = \{\mathbf{X}_j\}$.
2. Run RANSAC to fit the ground plane equation $\mathbf{n}^T \mathbf{X} + d = 0$.
   - A vertical constraint is enforced: the plane normal $\mathbf{n} = [n_x, n_y, n_z]^T$ must align with the camera gravity vector (in camera space, typically $[0, 1, 0]^T$ if Y points down) within a threshold (e.g., 20 degrees):
     $$\cos^{-1}(|\mathbf{n}^T [0, 1, 0]^T|) \le \theta_{\text{thresh}}$$
3. The distance of the plane from the camera center is given by $|d|$ (in unit translation units).
4. Since the physical camera height is $h_{\text{true}}$, the metric scale factor $s$ is:
  $$s = \frac{h_{\text{true}}}{|d|}$$
5. The translation vector and 3D map points are scaled:
  $$\mathbf{t}_{\text{metric}} = s \mathbf{t}_{\text{unit}}, \quad \mathbf{X}_{\text{metric}} = s \mathbf{X}_{\text{unit}}$$

### 5.2 Neural Metric Depth Priors
Monocular depth estimation networks (e.g. *Depth Anything v2*, *Metric3D v2*) predict metric or relative depth maps directly from single images.

```
Input Frame ──> Depth Anything v2 (ONNX/MPS) ──> Metric Depth Prior (d_j)
                                                          │
                                                          ▼
Reprojection Cost (2D-3D) ──> Joint Bundle Adjustment <── Depth Regularizer
```

#### Optimization Formulation
We can incorporate the neural depth prior $d_j$ for keypoint $j$ in the active keyframe as a regularizer in the Bundle Adjustment backend. The loss function becomes:
$$E_{\text{total}}(\mathbf{R}, \mathbf{t}, \mathbf{X}) = E_{\text{reproj}}(\mathbf{R}, \mathbf{t}, \mathbf{X}) + \lambda \sum_{j} w_j \left\| \mathbf{X}_j^{(z)} - d_j \right\|^2$$
where $\mathbf{X}_j^{(z)}$ is the depth (Z-coordinate) of the optimized 3D point in the camera frame, $w_j$ is the depth confidence, and $\lambda$ is a regularization weight. This bounds the triangulation depth uncertainty, maintaining absolute scale even during straight-line motions (low-parallax scenarios).

### 5.3 Visual-Inertial Odometry (VIO)
Fused visual-inertial systems utilize IMU measurements (accelerometer and gyroscope) to resolve scale.
* Gyroscopes measure angular velocity $\mathbf{\omega}$ directly, providing an absolute rotational prior.
* Accelerometers measure linear acceleration $\mathbf{a}$. Double integrating $\mathbf{a}$ recovers physical displacement:
  $$\mathbf{p}(t) = \mathbf{p}(t_0) + \mathbf{v}(t_0)\Delta t + \iint_{t_0}^{t} \mathbf{a}(\tau) d\tau^2$$
* **IMU Pre-integration (Forster et al.)**: Integrates high-frequency IMU measurements on the manifold $SO(3)$ between camera frames, defining relative motion factors that constrain scale and gravity orientation in a sliding window graph.

---

## 6. Summary & Recommendations

Modern monocular visual odometry systems have evolved from simple classical pipelines to highly robust, deep-geometric hybrids. To build a robust system:
1. **Measurement**: Combine learned local features (like **XFeat**) with graph-based matching (**LightGlue**) to handle illumination changes and low-texture road scenes.
2. **Motion Estimation**: Initialize with stable 2D-2D Epipolar Geometry. Transition to 3D-2D PnP tracking once keyframes are established.
3. **Backend Optimization**: **Always run joint Local Bundle Adjustment** (optimizing both poses and 3D structure) rather than pose-only PnP tracking, to prevent scale decay.
4. **Scale Recovery**: For automotive platforms, use Ground Plane scale fitting. For general robotics, integrate a lightweight monocular depth network (e.g. **Depth Anything v2 ONNX**) as regularizing constraints in the BA backend.
