# SuperPoint Visual Odometry Library

## TL;DR

> **Quick Summary**: 建置基於 SuperPoint 的教學級單目 Visual Odometry library，分兩階段執行。Phase 1 產出技術選型文件與可執行 prototype；Phase 2 重構為模組化 library 含完整 API、測試、CLI。
>
> **Deliverables**:
> - 技術選型比較文件（matching strategy: LightGlue vs Classic）
> - 可執行 VO prototype（處理 KITTI 序列，輸出軌跡）
> - 模組化 Python package（feature/match/pose/trajectory/eval/viz）
> - 可切換 matcher 介面（LightGlue / Classic BF+FLANN）
> - 雙格式軌跡輸出（KITTI + TUM）
> - Umeyama 對齊評估工具（APE/RTE metrics）
> - 單元測試（兼教學範例）
> - CLI 入口 + README
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 7 waves + Final verification
> **Critical Path**: T2 → T4 → T5/T6 → T7 → T9 → T12 → T17 → F1-F4

---

## Context

### Original Request
基於 SuperPoint 做簡單的 visual odometry (camera pose tracking) library，用於分析運鏡的局部運動。不需要 loop closure、複雜 localization 或 SLAM。針孔相機模型，焦距透過 FOV 度數定義（預設 63°）。

### Interview Summary
**Key Discussions**:
- **語言**: Pure Python（PyTorch + OpenCV + NumPy）
- **場景**: 學術研究 / 教學 — 重視可讀性、模組化、易於修改
- **匹配**: 研究 LightGlue（神經網路）和 Classic（BF/FLANN + Lowe's ratio），提供可切換介面
- **輸入**: 圖片序列 + 影片檔案
- **兩階段邊界**: Phase 1 = 研究文件 + runnable prototype; Phase 2 = 完整 library
- **測試**: 有單元測試，兼作教學參考
- **尺度**: Umeyama alignment 工具 + 可配置 scale 參數

**Research Findings**:
- SuperPoint (via LightGlue package): 256-d descriptors, batched PyTorch wrapper, auto-download weights
- LightGlue: 88.9% precision @3px, Adaptive 9-layer Transformer, ~7ms GPU, Apache 2.0
- Classic matching: ~72% precision, BF+Lowe's ratio test, 幾何驗證需 RANSAC
- Essential matrix: `cv2.findEssentialMat` (5-point RANSAC) + `cv2.recoverPose` (chirality check)
- FOV→K: `f = (W/2) / tan(FOV/2)`, square pixels, centered principal point
- Educational VO implementations (minislam, vo-from-scratch): ~300-500 LOC core, frame-to-frame E-matrix simplest

### Metis Review
**Identified Gaps** (addressed):
- Scale ambiguity → Fixed unit translation (||t||=1) per frame + post-hoc Umeyama alignment
- Pure rotation → Phase 1 skip with warning; Phase 2 = Homography decomposition fallback
- Lost tracking → len(matches) < 20 threshold, graceful degradation (skip frame, log warning)
- Reference frame → pose_0 = identity, world frame = first camera frame
- Edge cases → Zero matches, identical frames, out-of-bounds keypoints, recoverPose failure
- Scope creep → Explicit guardrails against BA, landmark management, IMU, multi-cam, custom training

---

## Work Objectives

### Core Objective
建置一個教學品質的、基於 SuperPoint 的單目 Visual Odometry library，可從圖片序列或影片中追蹤相機位姿軌跡，用於分析運鏡的局部運動。

### Concrete Deliverables
- `slam_dnn/` Python package with modular structure
- Tech decision document comparing LightGlue vs Classic matching
- Working VO prototype processing real image sequences
- Switchable matcher interface (LightGlue / Classic)
- KITTI + TUM trajectory export
- Umeyama alignment + APE/RTE evaluation tools
- Unit tests per module (teaching examples)
- CLI entry point (`python -m slam_dnn`)
- README with installation, quickstart, examples

### Definition of Done
- [ ] `python -m slam_dnn --input <images> --output <dir> --fov 63` produces KITTI + TUM trajectory files
- [ ] Trajectory plot (matplotlib) matches input motion qualitatively
- [ ] All unit tests pass: `python -m pytest tests/ -v`
- [ ] Synthetic pose recovery test: rotation error < 1°, translation direction error < 2°
- [ ] README has installation instructions and a working example

### Must Have
- SuperPoint feature extraction (via LightGlue package)
- Both LightGlue and Classic matchers with switchable interface
- Essential matrix pose estimation (`cv2.findEssentialMat` + `cv2.recoverPose`)
- SE3 trajectory accumulation (frame-to-frame)
- Pinhole camera model with FOV→K conversion (default 63°)
- Image sequence + video file input support
- KITTI + TUM trajectory output formats
- Umeyama alignment for trajectory evaluation
- Unit tests doubling as teaching examples
- Phase 1 prototype BEFORE Phase 2 refactor

### Must NOT Have (Guardrails)
- ❌ Loop closure or relocalization
- ❌ SLAM, map building, or 3D landmark management
- ❌ Bundle adjustment or pose graph optimization
- ❌ Triangulation or 3D point cloud generation
- ❌ IMU/sensor fusion
- ❌ Multi-camera support
- ❌ Real-time streaming API
- ❌ Custom neural network architectures or training
- ❌ Dataset-specific benchmarking harnesses
- ❌ GUI or interactive visualization framework
- ❌ Advanced camera models (fisheye, MEI)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO (new project)
- **Automated tests**: Tests-after (not TDD) — tests written per module as teaching references
- **Framework**: `pytest` with `numpy.testing` for numerical assertions
- **Test location**: `tests/` directory at project root

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation):
├── Task 1: Technical decision document [writing]
├── Task 2: Project scaffolding + dependencies [quick]
└── Task 3: Camera intrinsics + SE3 math utilities [quick]

Wave 2 (After Wave 1 - feature extraction + matching):
├── Task 4: SuperPoint feature extraction wrapper [unspecified-high]
├── Task 5: LightGlue matcher integration [unspecified-high]
└── Task 6: Classic matcher (BF + Lowe's ratio) [quick]

Wave 3 (After Wave 2 - pose estimation + integration):
├── Task 7: Essential matrix pose estimation [unspecified-high]
├── Task 8: SE3 trajectory accumulation [quick]
└── Task 9: End-to-end prototype script [deep]

Wave 4 (After Wave 3 - Phase 1 verification):
├── Task 10: Synthetic pose recovery tests [unspecified-high]
└── Task 11: Trajectory visualization + prototype QA [visual-engineering]

Wave 5 (After Phase 1 complete - module refactor, parallel):
├── Task 12: Modular package refactor + MatcherBase interface [unspecified-high]
├── Task 13: FrameLoader (image dirs + video files) [quick]
└── Task 14: Dual trajectory export (KITTI + TUM) [quick]

Wave 6 (After Wave 5 - evaluation + robustness):
├── Task 15: Umeyama alignment + APE/RTE evaluation [unspecified-high]
├── Task 16: Edge case handling (pure rotation, tracking lost) [unspecified-high]
└── Task 17: Comprehensive unit tests [unspecified-high]

Wave 7 (After Wave 6 - polish):
├── Task 18: CLI entry point + logging [quick]
└── Task 19: README + docstrings [writing]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
```

### Dependency Matrix

- **1**: None → blocks: none (standalone doc)
- **2**: None → blocks: 4, 5, 6
- **3**: None → blocks: 8
- **4**: 2 → blocks: 5, 6, 7, 9
- **5**: 2, 4 → blocks: 7, 9
- **6**: 2, 4 → blocks: 7, 9
- **7**: 4, 5, 6 → blocks: 9, 10
- **8**: 3 → blocks: 9
- **9**: 7, 8 → blocks: 10, 11
- **10**: 7, 9 → blocks: none (Phase 1 complete)
- **11**: 8, 9 → blocks: none (Phase 1 complete)
- **12**: 10, 11 → blocks: 13, 14, 15, 16, 17
- **13**: 12 → blocks: 17, 18
- **14**: 12 → blocks: 15, 17, 18
- **15**: 14 → blocks: 17, 19
- **16**: 12 → blocks: 17
- **17**: 12, 13, 14, 16 → blocks: 19
- **18**: 13, 14 → blocks: 19
- **19**: 15, 17, 18 → blocks: F1-F4

### Agent Dispatch Summary

- **Wave 1**: **3** - T1 → `writing`, T2 → `quick`, T3 → `quick`
- **Wave 2**: **3** - T4 → `unspecified-high`, T5 → `unspecified-high`, T6 → `quick`
- **Wave 3**: **3** - T7 → `unspecified-high`, T8 → `quick`, T9 → `deep`
- **Wave 4**: **2** - T10 → `unspecified-high`, T11 → `visual-engineering`
- **Wave 5**: **3** - T12 → `unspecified-high`, T13 → `quick`, T14 → `quick`
- **Wave 6**: **3** - T15 → `unspecified-high`, T16 → `unspecified-high`, T17 → `unspecified-high`
- **Wave 7**: **2** - T18 → `quick`, T19 → `writing`
- **Final**: **4** - F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> FORMAT: Task labels use bare numbers: `1.`, `2.`, etc.
> Final verification uses `F1.`, `F2.`, etc.

### Wave 1: Foundation (All Parallel)

- [x] 1. Technical Decision Document

  **What to do**:
  - 撰寫技術選型比較文件，存放於 `docs/tech-decisions.md`
  - 比較兩種 matching strategy：LightGlue vs Classic (BF/FLANN + Lowe's ratio)
  - 比較位姿估計方法：Essential matrix (5-point) vs Homography
  - 說明各項目的 pros/cons 和本專案的選擇理由
  - 包含 SuperPoint 架構說明（for 教學目的）

  **Must NOT do**:
  - 不包含實作程式碼（這是文件，不是程式）
  - 不涵蓋 loop closure, SLAM, bundle adjustment 等 out-of-scope 主題
  - 不過度深入論文數學推導（教學級別即可）

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 這是純文件撰寫任務
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: None (standalone doc)
  - **Blocked By**: None

  **References**:
  - Research findings (LightGlue 88.9% precision, Classic ~72%) from librarian agent results
  - SuperPoint paper: DeTone et al., 2018, arXiv:1712.07629
  - LightGlue paper: Lindenberger et al., 2023, arXiv:2306.13643
  - Essential matrix 5-point: Nistér, IEEE PAMI 2004

  **Acceptance Criteria**:
  - [ ] `docs/tech-decisions.md` exists with sections: Matching Strategy, Pose Estimation, SuperPoint Overview
  - [ ] 每節包含 pros/cons 比較表和本專案的選擇理由

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Document completeness check
    Tool: Bash (file check)
    Steps:
      1. Verify `docs/tech-decisions.md` exists
      2. Check sections: Matching Strategy, Pose Estimation, SuperPoint Architecture
      3. Verify each section has a "Decision" or "Recommendation" subsection
    Expected: File exists with all 3 sections, each with explicit decision
    Evidence: .omo/evidence/task-1-doc-completeness.txt
  ```

  **Commit**: NO (groups with Wave 1)

- [x] 2. Project Scaffolding + Dependencies

  **What to do**:
  - 建立專案結構：
    ```
    slam_dnn/
    ├── slam_dnn/
    │   ├── __init__.py
    │   ├── camera.py       (placeholder)
    │   ├── features.py     (placeholder)
    │   ├── matching.py     (placeholder)
    │   ├── pose.py         (placeholder)
    │   └── trajectory.py   (placeholder)
    ├── tests/
    │   ├── __init__.py
    │   └── conftest.py
    ├── pyproject.toml
    ├── requirements.txt
    └── README.md (placeholder)
    ```
  - `pyproject.toml` 設定 package metadata，name=`slam-dnn`，python>=3.9
  - `requirements.txt` 鎖定版本：
    ```
    torch>=2.0.0
    torchvision>=0.15.0
    opencv-python>=4.8.0
    numpy>=1.24.0
    scipy>=1.10.0
    matplotlib>=3.7.0
    pytest>=7.0.0
    ```
  - LightGlue 透過 `pip install git+https://github.com/cvg/LightGlue` 安裝
  - `conftest.py` 設定 pytest fixtures（temp dirs, sample arrays）

  **Must NOT do**:
  - 不寫任何實際實作邏輯（這只是骨架）
  - 不建立 configuration files (YAML/JSON)
  - 每個 module file 只放 `"""module docstring"""` + `pass` placeholder

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 建立目錄結構和設定檔，標準 scaffolding
  - **Skills**: `git-master` (for initial commit structure)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 4, 5, 6 (need package structure to exist)
  - **Blocked By**: None

  **References**:
  - LightGlue installation: `git clone https://github.com/cvg/LightGlue && pip install -e .`
  - Python packaging best practice: `pyproject.toml` (PEP 621)

  **Acceptance Criteria**:
  - [ ] `pyproject.toml` exists and `pip install -e .` succeeds
  - [ ] `python -c "import slam_dnn"` works without errors
  - [ ] `python -m pytest tests/` runs (0 tests, 0 errors)
  - [ ] All placeholder `.py` files exist with docstrings

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Package installs and imports
    Tool: Bash
    Steps:
      1. pip install -e .
      2. python -c "import slam_dnn; print(slam_dnn.__name__)"
    Expected: Prints "slam_dnn" without import errors
    Evidence: .omo/evidence/task-2-package-import.txt

  Scenario: Directory structure validation
    Tool: Bash
    Steps:
      1. Find all .py files in slam_dnn/ directory
      2. Verify: __init__.py, camera.py, features.py, matching.py, pose.py, trajectory.py exist
      3. Verify: tests/__init__.py and tests/conftest.py exist
    Expected: All 8 files present
    Evidence: .omo/evidence/task-2-dir-structure.txt
  ```

  **Commit**: NO (groups with Wave 1)
  - Message: `chore: project scaffolding with package structure and dependencies`
  - Files: `pyproject.toml`, `requirements.txt`, `slam_dnn/`, `tests/`

- [x] 3. Camera Intrinsics + SE3 Math Utilities

  **What to do**:
  - 在 `slam_dnn/camera.py` 實作：
    - `K_from_fov(width: int, height: int, fov_deg: float = 63.0) -> np.ndarray` — 從 FOV 建立 3x3 intrinsic matrix K
    - `PinholeCamera` dataclass: 儲存 width, height, fov, K, K_inv
  - 在 `slam_dnn/trajectory.py` 實作：
    - `pose_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray` — 從 R(3x3), t(3,) 建立 4x4 SE3 矩陣
    - `compose_pose(T_prev: np.ndarray, T_rel: np.ndarray) -> np.ndarray` — SE3 合成：T_global = T_prev @ T_rel
    - `normalize_translation(t: np.ndarray) -> np.ndarray` — 單位化平移向量
    - `extract_translations(poses: list) -> np.ndarray` — 從 pose list 提取 (N, 3) 位置陣列
  - 在 `tests/test_camera.py` 寫測試：
    - K 的 shape 和內容（fx=fy, cx=W/2, cy=H/2）
    - SE3 composition 的正確性（合成已知旋轉和平移）
    - normalize_translation 的行為

  **Must NOT do**:
  - 不處理 distortion（純 pinhole，無畸變）
  - 不加入其他相機模型（fisheye 等留給未來）
  - 不使用 scipy.spatial.transform.Rotation（保持 numpy 純數學，教學清楚）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 純數學函數，邏輯簡單清晰
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Task 8 (trajectory accumulation), Task 10 (synthetic tests)
  - **Blocked By**: None

  **References**:
  - FOV→K formula: `f = (W/2) / tan(FOV/2)`, square pixels, centered principal point
  - SE3 composition from minislam: `t_global = t_global + R_global @ t_rel`, `R_global = R_global @ R_rel`
  - SE3 source: `slam_dnn/trajectory.py` — pose_Rt, compose_pose

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_camera.py -v` → 3+ tests pass
  - [ ] `K_from_fov(640, 480, 63.0)` returns correct K (verify fx ≈ 640/(2*tan(31.5°)) ≈ 523)
  - [ ] SE3 composition: compose(I, T) == T, compose(T, T_inv) ≈ I

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: K matrix correctness
    Tool: Bash (pytest)
    Steps:
      1. python -m pytest tests/test_camera.py -v -k "test_k_from_fov"
    Expected:
      - K_from_fov(640, 480, 90) → fx=fy=320 (320/tan(45°) = 320)
      - K[0,2]==320, K[1,2]==240, K[2,2]==1
    Evidence: .omo/evidence/task-3-k-matrix.txt

  Scenario: SE3 composition identity test
    Tool: Bash (pytest)
    Steps:
      1. python -m pytest tests/test_camera.py -v -k "test_compose"
    Expected: compose(I, T) equals T within 1e-10 tolerance
    Evidence: .omo/evidence/task-3-se3-compose.txt

  Scenario: Normalize translation edge case
    Tool: Bash (pytest)
    Steps:
      1. Test normalize_translation(zero_vector) raises ValueError or returns zero
      2. Test normalize_translation([3,4,0]) returns [0.6, 0.8, 0.0]
    Expected: Handles zero gracefully, normalizes correctly
    Evidence: .omo/evidence/task-3-normalize-t.txt
  ```

  **Commit**: YES (with Wave 1 group)
  - Message: `feat: camera intrinsics from FOV and SE3 math utilities`
  - Files: `slam_dnn/camera.py`, `slam_dnn/trajectory.py`, `tests/test_camera.py`

### Wave 2: Feature Extraction + Matching (After Wave 1)

- [x] 4. SuperPoint Feature Extraction Wrapper

  **What to do**:
  - 在 `slam_dnn/features.py` 實作 `SuperPointExtractor` 類別：
    - `__init__(self, max_num_keypoints=2048, detection_threshold=0.0005, device='auto')` — device auto-detect GPU/CPU
    - `extract(self, image: np.ndarray) -> dict` — 輸入灰階或 BGR uint8 圖片
    - 回傳 dict: `{"keypoints": (N, 2), "descriptors": (N, 256), "scores": (N,)}`
  - 使用 `from lightglue import SuperPoint` 作為底層 extractor
  - 內部處理：BGR→Grayscale, uint8→float [0,1], add batch dim, move to device
  - 回傳前 remove batch dim，轉為 numpy arrays
  - 在 `tests/test_features.py` 寫測試：
    - 合成一張有角點的圖片（checkerboard），確認能偵測到特徵
    - 驗證輸出 shape：keypoints (N,2), descriptors (N,256), scores (N,)
    - 驗證 descriptors 是 L2-normalized（norm ≈ 1.0）

  **Must NOT do**:
  - 不從頭寫 SuperPoint 模型（直接用 LightGlue package 的）
  - 不加 NMS 或後處理（LightGlue 的 SuperPoint 已經做了）
  - 不處理 batch input（只處理單張圖片）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要理解 LightGlue superpoint API 並做適當包裝
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6)
  - **Blocks**: Tasks 5, 6, 7, 9
  - **Blocked By**: Task 2 (package structure must exist)

  **References**:
  - LightGlue SuperPoint API: `from lightglue import SuperPoint; model = SuperPoint(max_num_keypoints=2048)`
  - SuperPoint output format: `{"keypoints": [1,N,2], "keypoint_scores": [1,N], "descriptors": [1,N,256]}`
  - LightGlue source: `lightglue/superpoint.py` — SuperPoint class, Extractor base
  - Device auto-detect: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_features.py -v` → 3+ tests pass
  - [ ] extract() 輸入灰階 uint8 (H,W) 或 BGR uint8 (H,W,3) 都能正常工作
  - [ ] descriptors 是 L2-normalized（每列 norm ≈ 1.0, tol=0.01）
  - [ ] device auto-detect 在無 GPU 環境下回退到 CPU

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Feature extraction on synthetic image
    Tool: Bash (pytest)
    Steps:
      1. Create synthetic checkerboard image (200x200, uint8)
      2. extractor = SuperPointExtractor(max_num_keypoints=500, device='cpu')
      3. result = extractor.extract(img)
      4. Assert len(result["keypoints"]) > 0
      5. Assert result["descriptors"].shape[1] == 256
    Expected: Detects features, returns correct shapes
    Evidence: .omo/evidence/task-4-extraction.txt

  Scenario: Descriptor normalization
    Tool: Bash (pytest)
    Steps:
      1. Extract features from any test image
      2. norms = np.linalg.norm(result["descriptors"], axis=1)
      3. Assert all norms within [0.99, 1.01]
    Expected: All descriptors are L2-unit normalized
    Evidence: .omo/evidence/task-4-descriptor-norm.txt

  Scenario: Empty image (no features)
    Tool: Bash (pytest)
    Steps:
      1. Create blank white image (uniform, no corners)
      2. result = extractor.extract(img)
      3. Assert len(result["keypoints"]) == 0 (or very few)
    Expected: Returns empty arrays gracefully, no crash
    Evidence: .omo/evidence/task-4-empty-image.txt
  ```

  **Commit**: NO (groups with Wave 2)

- [x] 5. LightGlue Matcher Integration

  **What to do**:
  - 在 `slam_dnn/matching.py` 實作 `LightGlueMatcher` 類別：
    - `__init__(self, filter_threshold=0.1, device='auto')`
    - `match(self, feats0: dict, feats1: dict) -> dict` — 輸入 SuperPoint 的 feat dicts
    - 回傳 dict: `{"points0": (K, 2), "points1": (K, 2), "scores": (K,), "indices": (K, 2)}`
  - 使用 `from lightglue import LightGlue` 作為底層
  - 回傳的 points0/points1 已經是 matched pairs（pixel coordinates）
  - 在 `tests/test_matching.py` 寫測試：
    - 兩張相似圖片（位移少量），確認匹配數量 > 10
    - 驗證 points0 和 points1 shape 一致
    - 驗證 indices 的範圍在有效 keypoint index 內

  **Must NOT do**:
  - 不重複實作 SuperPoint（matcher 只負責匹配，接受 pre-computed features）
  - 不做 geometric verification（那是 Task 7 的 pose estimation 工作）
  - 不支援 batch matching（只處理一對圖片）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要理解 LightGlue matcher 的輸入輸出格式
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6)
  - **Blocks**: Tasks 7, 9
  - **Blocked By**: Tasks 2, 4 (package structure + SuperPoint output format)

  **References**:
  - LightGlue matcher API: `LightGlue(features='superpoint').eval().cuda()`
  - Input format: `{'image0': feats0, 'image1': feats1}` — each is SuperPoint output dict
  - Output format: `{'matches': [K,2], 'scores': [K]}` — matches 是 (img0_idx, img1_idx) pairs
  - `rbd()` util removes batch dim: `from lightglue.utils import rbd`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_matching.py -v -k "lightglue"` → 2+ tests pass
  - [ ] match() 回傳的 points0/points1 長度相同且 > 0
  - [ ] scores 範圍在 [0, 1]

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Matching two similar images
    Tool: Bash (pytest)
    Steps:
      1. Create two images: img_a (checkerboard), img_b (same with small shift)
      2. Extract features from both using SuperPointExtractor
      3. matcher = LightGlueMatcher(filter_threshold=0.1, device='cpu')
      4. result = matcher.match(feats_a, feats_b)
      5. Assert len(result["points0"]) > 10
      6. Assert result["points0"].shape == result["points1"].shape
    Expected: Finds substantial matches between similar images
    Evidence: .omo/evidence/task-5-lg-match.txt

  Scenario: Matching completely different images
    Tool: Bash (pytest)
    Steps:
      1. Use two completely different random noise images
      2. result = matcher.match(feats_a, feats_b)
      3. Assert len(result["points0"]) < len(result_a["keypoints"]) (some filtered)
    Expected: Low match count, no crash
    Evidence: .omo/evidence/task-5-lg-dissimilar.txt
  ```

  **Commit**: NO (groups with Wave 2)

- [x] 6. Classic Matcher (BF + Lowe's Ratio Test)

  **What to do**:
  - 在 `slam_dnn/matching.py` 增加 `ClassicMatcher` 類別（與 Task 5 同檔案）：
    - `__init__(self, ratio=0.75, ransac_reproj_threshold=3.0, method='bf')` — method: 'bf' or 'flann'
    - `match(self, feats0: dict, feats1: dict) -> dict` — 相同的輸入輸出格式
    - 使用 `cv2.BFMatcher(cv2.NORM_L2, crossCheck=False).knnMatch(desc0, desc1, k=2)`
    - Lowe's ratio test: keep match if `m.distance < ratio * n.distance`
    - 回傳格式完全同 Task 5 的 LightGlueMatcher（`points0, points1, scores, indices`）
  - 在 `tests/test_matching.py` 增加測試：
    - 同樣的測試圖片，確認 ClassicMatcher 也能找到足夠匹配
    - ratio test 效果：ratio=0.5 應比 ratio=0.8 返回更少匹配

  **Must NOT do**:
  - 不加入 FLANN 以外的 NN search 方法
  - 不做 geometric verification（同 Task 5，留給 Task 7）
  - 不修改 OpenCV 的 FLANN 參數（trees=5, checks=50 固定）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: BFMatcher + ratio test 是標準 OpenCV pattern，實作簡單
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: Tasks 7, 9
  - **Blocked By**: Tasks 2, 4

  **References**:
  - OpenCV BFMatcher: `cv2.BFMatcher(cv2.NORM_L2, crossCheck=False).knnMatch(desc0, desc1, k=2)`
  - Lowe's ratio test: keep match if `m.distance < 0.75 * n.distance`
  - FLANN index params: `dict(algorithm=1, trees=5)`, search params: `dict(checks=50)`
  - minislam features.py — same knnMatch + ratio test pattern

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_matching.py -v -k "classic"` → 2+ tests pass
  - [ ] ClassicMatcher 和 LightGlueMatcher 的 match() 輸出格式完全一致（可互換）
  - [ ] ratio=0.5 的匹配數量 < ratio=0.9 的匹配數量（驗證 ratio test 邏輯）

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Classic matcher produces matches
    Tool: Bash (pytest)
    Steps:
      1. Same test images as LightGlue test
      2. matcher = ClassicMatcher(ratio=0.75)
      3. result = matcher.match(feats_a, feats_b)
      4. Assert len(result["points0"]) > 5
      5. Assert result has same keys as LightGlueMatcher output
    Expected: Compatible output format, finds matches
    Evidence: .omo/evidence/task-6-classic-match.txt

  Scenario: Ratio test filtering
    Tool: Bash (pytest)
    Steps:
      1. matcher_strict = ClassicMatcher(ratio=0.5)
      2. matcher_loose = ClassicMatcher(ratio=0.9)
      3. result_strict = matcher_strict.match(feats_a, feats_b)
      4. result_loose = matcher_loose.match(feats_a, feats_b)
      5. Assert len(result_strict["points0"]) <= len(result_loose["points0"])
    Expected: Stricter ratio = fewer matches
    Evidence: .omo/evidence/task-6-ratio-test.txt
  ```

  **Commit**: YES (Wave 2 group)
  - Message: `feat: SuperPoint extraction + LightGlue and Classic matchers`
  - Files: `slam_dnn/features.py`, `slam_dnn/matching.py`, `tests/test_features.py`, `tests/test_matching.py`

### Wave 3: Pose Estimation + Integration (After Wave 2)

- [x] 7. Essential Matrix Pose Estimation

  **What to do**:
  - 在 `slam_dnn/pose.py` 實作：
    - `estimate_essential(points0: np.ndarray, points1: np.ndarray, K: np.ndarray, ransac_thresh=1.0, conf=0.999) -> tuple` — 回傳 `(R, t, inlier_mask)` 或 `None`（匹配不足時）
    - 內部流程：
      1. 將 pixel coordinates 轉換為 normalized camera coordinates：`(pts - [cx,cy]) / [fx,fy]`
      2. `cv2.findEssentialMat(pts0_norm, pts1_norm, np.eye(3), method=cv2.RANSAC, threshold=RANSAC_pixel_thresh/f_mean, prob=conf)`
      3. `cv2.recoverPose(E, pts0_norm, pts1_norm, np.eye(3))` — 4-way chirality disambiguation
      4. 回傳 `(R: 3x3, t: (3,), inlier_mask: (N,) bool)`
    - 最小匹配檢查：`len(points0) < 8 → return None`
    - E matrix shape 檢查：如果 findEssentialMat 回傳 stacked E [3k×3]，取前 3x3
  - 在 `slam_dnn/exceptions.py`（新建）定義 `TrackingLostError(Exception)` 例外
  - 在 `tests/test_pose.py` 寫測試：
    - 合成兩組對應點（known rotation + translation），驗證 recover 的 R, t 與真值接近
    - 測試匹配不足時（< 8 點）回傳 None

  **Must NOT do**:
  - 不做 PnP pose estimation（那是 map-based VO 的做法，這裡只用 E-matrix）
  - 不做 Homography decomposition fallback（Phase 2 的 edge case handling 再做）
  - 不處理 pure rotation 情形（Task 16 負責）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: E-matrix 估計涉及 coordinate normalization 和 edge case handling
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (needs match format from T4/T5/T6)
  - **Parallel Group**: Wave 3 (with Tasks 8, 9 but 9 depends on 7)
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Tasks 4, 5, 6 (matched points format)

  **References**:
  - Production example from cvg/LightGlue GitHub issue #56: `estimate_relative_pose()` function
  - pyslam pattern: normalize points to camera coords, use `focal=1, pp=(0,0)`
  - `cv2.findEssentialMat` Python signature: `(points1, points2, cameraMatrix, method, prob, threshold, mask)`
  - `cv2.recoverPose` Python signature: `(E, points1, points2, cameraMatrix, distanceThresh, mask) → (n_inliers, R, t, mask)`
  - Stacked E handling: `if E.shape != (3,3): E = E[:3, :]`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_pose.py -v` → 3+ tests pass
  - [ ] 合成場景（known R=5°, t=[1,0,0]）：rotation error < 2°, translation direction error < 5°
  - [ ] `estimate_essential(empty, empty, K)` 回傳 None
  - [ ] `TrackingLostError` exception class defined and importable

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Pose recovery from synthetic correspondences
    Tool: Bash (pytest)
    Steps:
      1. Create K = K_from_fov(640, 480, 63)
      2. Generate 100 random 3D points (z=5~15)
      3. Project to img1 (identity pose) and img2 (R=rodrigues([0.01,0.02,-0.01]), t=[0.5, -0.2, 1.0])
      4. Add 0.5px Gaussian noise to both sets
      5. R_est, t_est, mask = estimate_essential(pts1, pts2, K)
      6. Compute rotation error: arccos((trace(R_est @ R_gt.T) - 1) / 2)
      7. Compute translation direction error: arccos(dot(t_est, t_gt_norm))
    Expected: rotation error < 2°, translation direction error < 5°
    Evidence: .omo/evidence/task-7-pose-recovery.txt

  Scenario: Insufficient matches returns None
    Tool: Bash (pytest)
    Steps:
      1. pts = np.random.rand(3, 2) (only 3 points)
      2. result = estimate_essential(pts, pts, K)
      3. Assert result is None
    Expected: Graceful None return, no crash
    Evidence: .omo/evidence/task-7-insufficient-matches.txt
  ```

  **Commit**: NO (groups with Wave 3)

- [x] 8. SE3 Trajectory Accumulation

  **What to do**:
  - 在 `slam_dnn/trajectory.py` 增加（擴充 Task 3 的模組）：
    - `TrajectoryAccumulator` 類別：
      - `__init__(self, scale: float = 1.0)` — 平移尺度因子
      - `add_pose(self, R: np.ndarray, t: np.ndarray)` — 加入相對 pose (R,t)
      - `get_poses(self) -> list[np.ndarray]` — 回傳所有 4x4 全局 poses
      - `get_positions(self) -> np.ndarray` — 回傳 (N, 3) 位置陣列
      - `reset(self)` — 重置軌跡
    - 內部實作 SE3 合成：
      ```python
      t_norm = normalize_translation(t) if np.linalg.norm(t) > 1e-8 else t
      self.cur_t = self.cur_t + self.scale * (self.cur_R @ t_norm)
      self.cur_R = self.cur_R @ R
      ```
    - 初始狀態：`cur_R = np.eye(3)`, `cur_t = np.zeros(3)`, poses = [identity 4x4]
  - 在 `tests/test_trajectory.py` 增加測試：
    - 連續加入 10 個 identity poses → 軌跡保持原點
    - 連續加入相同 (R, t) → 軌跡沿固定方向移動
    - scale 因子：scale=2.0 的位移應是 scale=1.0 的兩倍

  **Must NOT do**:
  - 不做 loop closure detection
  - 不做 pose graph optimization
  - 不做 sliding window bundle adjustment
  - 不處理 rotation matrix 的重新正交化（數值穩定性留給未來優化）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 擴充已有的 trajectory.py，邏輯簡單
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 9)
  - **Blocks**: Task 9
  - **Blocked By**: Task 3 (math utilities)

  **References**:
  - minislam odometry.py: `self.cur_t = self.cur_t + (self.scale * self.cur_R.dot(t))`
  - minislam odometry.py: `self.cur_R = self.cur_R.dot(R)`
  - 4x4 pose construction: Task 3 的 `pose_Rt` function

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_trajectory.py -v` → 3+ tests pass
  - [ ] 10 個 identity poses → 所有位置都在 (0, 0, 0)
  - [ ] scale=2.0 的最終位置 ≈ 2 × scale=1.0 的最終位置

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Linear trajectory accumulation
    Tool: Bash (pytest)
    Steps:
      1. traj = TrajectoryAccumulator(scale=1.0)
      2. R_id = np.eye(3), t_right = np.array([1, 0, 0])
      3. Add 5 poses: for i in range(5): traj.add_pose(R_id, t_right)
      4. positions = traj.get_positions()
      5. Assert positions[-1] ≈ [5, 0, 0] (moved 5 units right)
    Expected: Linear trajectory along x-axis
    Evidence: .omo/evidence/task-8-linear-traj.txt

  Scenario: Scale factor doubles translation
    Tool: Bash (pytest)
    Steps:
      1. traj1 = TrajectoryAccumulator(scale=1.0)
      2. traj2 = TrajectoryAccumulator(scale=2.0)
      3. Same 5 poses added to both
      4. Assert traj2 positions[-1] ≈ 2 * traj1 positions[-1]
    Expected: Scale factor applied correctly
    Evidence: .omo/evidence/task-8-scale-factor.txt
  ```

  **Commit**: NO (groups with Wave 3)

- [x] 9. End-to-End Prototype Script

  **What to do**:
  - 建立 `run_vo.py`（專案根目錄），整合 Tasks 2-8 的所有模組：
    ```python
    # 偽代碼
    images = load_images(args.input)         # glob + sorted
    K = K_from_fov(images[0].shape[1], images[0].shape[0], args.fov)
    extractor = SuperPointExtractor(...)
    matcher = LightGlueMatcher(...)  # or ClassicMatcher
    trajectory = TrajectoryAccumulator(scale=args.scale)
    
    for i, (img_prev, img_curr) in enumerate(pairs(images)):
        feats_prev = extractor.extract(img_prev)
        feats_curr = extractor.extract(img_curr)
        match_result = matcher.match(feats_prev, feats_curr)
        
        if len(match_result["points0"]) < 20:
            print(f"Warning: frame {i} too few matches, skipping")
            continue
        
        pose = estimate_essential(match_result["points0"], match_result["points1"], K)
        if pose is None:
            print(f"Warning: frame {i} pose estimation failed")
            continue
        
        R, t, inlier_mask = pose
        trajectory.add_pose(R, t)
    
    # Output
    save_kitti_trajectory(trajectory.get_poses(), args.output + "/trajectory_kitti.txt")
    plot_trajectory(trajectory.get_positions(), args.output + "/trajectory_plot.png")
    ```
  - 支援 argparse 參數：
    - `--input`: 圖片目錄路徑
    - `--output`: 輸出目錄（不存在則建立）
    - `--fov`: FOV 度數 (default=63.0)
    - `--matcher`: 'lightglue' | 'classic' (default='lightglue')
    - `--max-keypoints`: int (default=2048)
    - `--scale`: float (default=1.0)
    - `--device`: 'auto' | 'cuda' | 'cpu' (default='auto')
  - 輸出 KITTI 格式的 `trajectory_kitti.txt`（每行 12 個 floats, 3x4 row-major）
  - 輸出 `trajectory_plot.png`（matplotlib 2D top-down view）
  - 輸出簡單的 console 進度：`Frame 0/100... Frame 1/100...`
  - 加入 `load_images(path)` 工具函數：glob PNG/JPG/JPEG, sorted by filename

  **Must NOT do**:
  - 不使用 logging module（print 即可，Phase 1）
  - 不做 YAML/JSON 設定檔
  - 不做 GUI visualization
  - 不處理影片輸入（Phase 2 的 FrameLoader）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 整合所有模組的 end-to-end pipeline，需要 debugging 各組件互動
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (needs T4-T8)
  - **Parallel Group**: Wave 3 (sequential after T7, T8)
  - **Blocks**: Tasks 10, 11
  - **Blocked By**: Tasks 7, 8

  **References**:
  - KITTI trajectory format: one line per frame, 12 space-separated floats (row-major 3x4)
  - minislam main loop pattern for per-frame processing
  - argparse Python standard library
  - matplotlib 2D trajectory plot: `plt.plot(positions[:, 0], positions[:, 2])` for top-down (XZ plane)

  **Acceptance Criteria**:
  - [ ] `python run_vo.py --help` 顯示所有參數
  - [ ] `python run_vo.py --input <test_images> --output <test_dir>` 成功執行
  - [ ] `<test_dir>/trajectory_kitti.txt` 存在，每行 12 個空格分隔的 floats
  - [ ] `<test_dir>/trajectory_plot.png` 存在（有效圖片）
  - [ ] 至少處理 10 張連續圖片不 crash

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: End-to-end prototype runs on synthetic image sequence
    Tool: Bash
    Steps:
      1. Create synthetic image sequence: 20 frames, simple geometric pattern with gradual translation
      2. python run_vo.py --input <synth_dir> --output <out_dir> --fov 63 --device cpu
      3. Check <out_dir>/trajectory_kitti.txt exists and has 19 lines (frame 0 = identity, so 19 relative poses)
      4. Check <out_dir>/trajectory_plot.png exists and is valid image
      5. Each line has exactly 12 space-separated floats
    Expected: Prototype runs without errors, produces output files
    Evidence: .omo/evidence/task-9-e2e-synthetic.txt

  Scenario: CLI argument validation
    Tool: Bash
    Steps:
      1. python run_vo.py --help → verify all 7 args documented
      2. python run_vo.py --input nonexistent/ → handle gracefully (error message, not crash)
      3. python run_vo.py --matcher invalid → handle gracefully
    Expected: Help works, invalid inputs produce clear error messages
    Evidence: .omo/evidence/task-9-cli-validation.txt
  ```

  **Commit**: YES (Phase 1 complete)
  - Message: `feat: working visual odometry prototype with SuperPoint + dual matchers`
  - Files: `run_vo.py`
  - Pre-commit: `python -m pytest tests/ -v`

### Wave 4: Phase 1 Verification (After Wave 3)

- [x] 10. Synthetic Pose Recovery Tests

  **What to do**:
  - 建立 `tests/test_synthetic.py` — 完整的合成場景 end-to-end 測試：
    - **test_synthetic_translation_recovery**: 
      1. 生成 50 個隨機 3D 點（z=5~15m）
      2. 已知 K = K_from_fov(640, 480, 63)
      3. 已知 GT pose: R=identity, t=[1.0, 0, 0]
      4. 投影到兩個視角，加入 0.5px Gaussian 噪音
      5. 用 `estimate_essential()` 恢復 pose
      6. Assert translation direction error < 5°
    - **test_synthetic_rotation_recovery**:
      1. 同樣的 3D 點和 K
      2. GT pose: R=rodrigues([0.02, 0.03, -0.01]), t=[0.3, 0, 0.5]
      3. Assert rotation error < 2°
    - **test_synthetic_trajectory_accumulation**:
      1. 連續 10 幀已知 poses
      2. 用 TrajectoryAccumulator 累積
      3. 驗證最終位置在預期範圍內
    - **test_se3_composition_accuracy**: 驗證 4x4 矩陣相乘 vs 逐步 (R,t) 合成結果一致
  - 輔助函數（在 `tests/helpers.py` 或直接 in-file）：
    - `project_points(points_3d, R, t, K) -> points_2d` — 3D→2D 投影
    - `rotation_error(R1, R2) -> float` — 旋轉誤差（度）
    - `direction_error(t1, t2) -> float` — 平移方向誤差（度）

  **Must NOT do**:
  - 不依賴外部資料集或真實圖片
  - 不測試 SuperPoint 模型本身（只測試幾何 pipeline）
  - 不加入 GPU 測試（所有測試用 CPU）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 合成幾何測試需要仔細的數學實作和 edge case 處理
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 11)
  - **Blocks**: Task 12 (Phase 2 開始需要 Phase 1 測試全 pass)
  - **Blocked By**: Tasks 7, 9

  **References**:
  - 3D projection: `P_2d = K @ (R @ P_3d + t)`, then divide by z
  - Rotation error formula: `arccos((trace(R1 @ R2.T) - 1) / 2)`
  - Direction error: `arccos(dot(t1/||t1||, t2/||t2||))`
  - Rodrigues: `cv2.Rodrigues(rvec)[0]` for small-angle rotation matrices

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_synthetic.py -v` → 4 tests pass
  - [ ] Translation recovery test: error < 5° (with 0.5px noise)
  - [ ] Rotation recovery test: error < 2°
  - [ ] SE3 composition test: matches within 1e-10

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: All synthetic tests pass
    Tool: Bash (pytest)
    Steps:
      1. python -m pytest tests/test_synthetic.py -v
    Expected: 4 tests pass, 0 failures. Each test reports the actual error in degrees.
    Evidence: .omo/evidence/task-10-synthetic-tests.txt

  Scenario: Noise robustness — high noise case
    Tool: Bash (pytest)
    Steps:
      1. Run trajectory accumulation test with 5px noise instead of 0.5px
      2. Verify errors are larger but still bounded (< 15°)
    Expected: Pipeline degrades gracefully with noise, doesn't crash
    Evidence: .omo/evidence/task-10-noise-robustness.txt
  ```

  **Commit**: YES (Phase 1 test)
  - Message: `test: synthetic pose recovery tests for VO pipeline validation`
  - Files: `tests/test_synthetic.py`, `tests/helpers.py`

- [x] 11. Trajectory Visualization + Prototype QA

  **What to do**:
  - 在 `slam_dnn/viz.py` 實作：
    - `plot_trajectory_2d(positions: np.ndarray, title: str = "Camera Trajectory", save_path: str = None, show: bool = True)` — Matplotlib 2D top-down plot (XZ plane for driving, or XY plane as default)
    - 標記起始點（綠色）和終點（紅色）
    - 等比例軸 (equal aspect ratio)
    - 如果 save_path 提供則儲存圖片
  - 對 `run_vo.py` 做完整的 QA 測試：
    - 準備一組簡單的合成圖片序列（用 OpenCV 畫幾何圖形，逐幀平移）
    - 執行 `run_vo.py` 驗證輸出軌跡方向和形狀大致正確
    - 驗證 KITTI 格式正確性（每行 12 欄）
  - 在 `tests/test_viz.py` 寫測試：
    - `plot_trajectory_2d` 生成的圖片檔案存在且大小 > 0
    - matplotlib figure closes properly (no memory leak)

  **Must NOT do**:
  - 不做 3D 視覺化（matplotlib 3D 太複雜，Phase 1 只需要 2D）
  - 不做即時視覺化（不整合進 main loop）
  - 不做互動式 plot（zoom, pan 等）

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: 視覺化任務，關注 matplotlib 的畫面品質
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 10)
  - **Blocks**: Task 12 (Phase 2 需要 visualize 模組)
  - **Blocked By**: Tasks 8, 9

  **References**:
  - matplotlib 2D plot: `plt.plot(x, z, '-')` for top-down view
  - minislam display: top-down XZ plane with colored trajectory
  - Equal aspect: `ax.set_aspect('equal')` or `plt.axis('equal')`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_viz.py -v` → 2+ tests pass
  - [ ] `plot_trajectory_2d(positions, save_path="test.png")` 生成有效圖片
  - [ ] 圖片包含起始點（綠色）和終點（紅色）標記

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Trajectory plot generates valid image
    Tool: Bash (pytest)
    Steps:
      1. positions = np.array([[0,0,0], [1,0,1], [2,0,2], [3,0,3]])
      2. plot_trajectory_2d(positions, save_path="/tmp/test_traj.png", show=False)
      3. Assert os.path.exists("/tmp/test_traj.png")
      4. Assert os.path.getsize("/tmp/test_traj.png") > 1000
    Expected: Valid PNG file generated
    Evidence: .omo/evidence/task-11-plot-generates.txt

  Scenario: Prototype end-to-end visual check
    Tool: Bash
    Steps:
      1. Run prototype on synthetic sequence
      2. Verify output trajectory_plot.png is generated
      3. Read image file and verify it's a valid PNG
    Expected: Prototype produces visual output correctly
    Evidence: .omo/evidence/task-11-prototype-viz.txt
  ```

  **Commit**: YES (Phase 1 visualization)
  - Message: `feat: trajectory visualization and Phase 1 prototype QA`
  - Files: `slam_dnn/viz.py`, `tests/test_viz.py`

### Wave 5: Module Refactor (After Phase 1 Complete, All Parallel)

- [x] 12. Modular Package Refactor + MatcherBase Interface

  **What to do**:
  - 定義 `slam_dnn/matching.py` 中的 abstract base class：
    ```python
    class MatcherBase(ABC):
        @abstractmethod
        def match(self, feats0: dict, feats1: dict) -> dict:
            """回傳 {"points0": (K,2), "points1": (K,2), "scores": (K,), "indices": (K,2)}"""
            ...
    
    class LightGlueMatcher(MatcherBase): ...
    class ClassicMatcher(MatcherBase): ...
    ```
  - 新增 `slam_dnn/vo.py` — `VisualOdometry` 主類別：
    ```python
    class VisualOdometry:
        def __init__(self, camera: PinholeCamera, matcher: MatcherBase | str = 'lightglue', scale=1.0, device='auto', **kwargs)
        def process_frame(self, img: np.ndarray) -> np.ndarray | None  # returns 4x4 pose or None
        def process_sequence(self, images: Iterator[np.ndarray]) -> list[np.ndarray]
        def get_trajectory(self) -> TrajectoryAccumulator
        def reset(self)
    ```
  - 新增 factory function：`create_matcher(method: str, **kwargs) -> MatcherBase`
  - 確保 `run_vo.py` 仍可用（用新的 VisualOdometry class 重寫，保持 CLI 介面不變）
  - 重寫 `slam_dnn/__init__.py` 提供清晰的 public API exports：
    ```python
    from slam_dnn.camera import PinholeCamera, K_from_fov
    from slam_dnn.features import SuperPointExtractor
    from slam_dnn.matching import MatcherBase, LightGlueMatcher, ClassicMatcher, create_matcher
    from slam_dnn.pose import estimate_essential, TrackingLostError
    from slam_dnn.trajectory import TrajectoryAccumulator, pose_Rt, compose_pose
    from slam_dnn.vo import VisualOdometry
    ```

  **Must NOT do**:
  - 不改變現有的輸出格式或行為（Phase 1 tests 必須繼續 pass）
  - 不過度抽象（不要為 camera 或 trajectory 加 abstract base class）
  - 不加入 plugin system 或 dynamic module loading
  - 不改變 SuperPoint 的使用方式（仍然用 LightGlue package 的 SuperPoint）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要細心地將 prototype code 重構成清晰的 package structure
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Tasks 13, 14)
  - **Blocks**: Tasks 15, 16, 17, 18, 19 (everything depends on package structure)
  - **Blocked By**: Tasks 10, 11 (Phase 1 must be complete)

  **References**:
  - Phase 1 code: `run_vo.py` to be refactored into `slam_dnn/vo.py`
  - Python ABC pattern: `from abc import ABC, abstractmethod`
  - Scikit-learn style interfaces: `fit()` / `predict()` metaphor → `match()` here
  - Existing tests: `tests/test_*.py` — all must still pass after refactor

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/ -v` → ALL existing Phase 1 tests still pass (zero regressions)
  - [ ] `MatcherBase` 有 `match` abstract method
  - [ ] `LightGlueMatcher` 和 `ClassicMatcher` 都繼承 `MatcherBase`，interface 一致
  - [ ] `VisualOdometry` class 可透過 `vo = VisualOdometry(camera, matcher='lightglue')` 建立
  - [ ] `run_vo.py` 改用 `VisualOdometry` class 但仍產出相同輸出

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: No regression — existing tests pass
    Tool: Bash (pytest)
    Steps:
      1. python -m pytest tests/ -v
    Expected: All tests from Waves 1-4 pass (test_camera, test_features, test_matching, test_pose, test_trajectory, test_synthetic, test_viz)
    Evidence: .omo/evidence/task-12-no-regression.txt

  Scenario: MatcherBase interface compliance
    Tool: Bash
    Steps:
      1. Instantiate both matchers
      2. Assert isinstance(matcher, MatcherBase) for both
      3. Call match() on same inputs, verify output dict keys match
    Expected: Both matchers implement MatcherBase, same interface
    Evidence: .omo/evidence/task-12-matcher-interface.txt

  Scenario: VisualOdometry class API
    Tool: Bash
    Steps:
      1. cam = PinholeCamera(640, 480, 63)
      2. vo = VisualOdometry(cam, matcher='lightglue', device='cpu')
      3. Assert hasattr(vo, 'process_frame')
      4. Assert hasattr(vo, 'process_sequence')
      5. Assert hasattr(vo, 'get_trajectory')
    Expected: All public methods accessible
    Evidence: .omo/evidence/task-12-vo-api.txt
  ```

  **Commit**: YES
  - Message: `refactor: modular package structure with MatcherBase interface and VisualOdometry class`
  - Files: `slam_dnn/__init__.py`, `slam_dnn/matching.py`, `slam_dnn/vo.py`, `run_vo.py`

- [x] 13. FrameLoader (Image Directories + Video Files)

  **What to do**:
  - 建立 `slam_dnn/io.py`：
    - `FrameLoader` 類別（實作 `__iter__`）：
      ```python
      class FrameLoader:
          def __init__(self, source: str, max_frames: int | None = None, resize: tuple | None = None)
          def __iter__(self) -> Iterator[np.ndarray]  # yields BGR uint8 frames
          def __len__(self) -> int                      # total frames (if known)
      ```
    - **Image directory**: glob PNG/JPG/JPEG/PPM/BMP, sorted by filename, yield each read
    - **Video file**: `cv2.VideoCapture(source)`, yield frames one by one
    - **Auto-detect**: 如果 source 是目錄 → image mode; 如果是檔案 → video mode
    - `resize` 選項: 如果指定則 `cv2.resize` 每幀
    - `max_frames` 選項: 限制最大幀數（testing 用）
  - 工具函數：
    - `to_grayscale(img: np.ndarray) -> np.ndarray` — BGR uint8 → gray uint8
    - `to_float(img: np.ndarray) -> np.ndarray` — uint8 → float32 [0,1]
  - 在 `tests/test_io.py` 寫測試：
    - Image directory loading（建立暫存目錄放測試圖片）
    - Video loading（用 OpenCV 寫一個小影片檔）
    - max_frames 限制
    - resize 行為

  **Must NOT do**:
  - 不支援 webcam / real-time camera stream
  - 不做 frame skipping 或 frame rate control
  - 不做 async / parallel loading
  - 不建立 temp directory 存放 video frames（直接 yield from VideoCapture）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 標準 I/O wrapper，模式清晰
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Tasks 12, 14)
  - **Blocks**: Tasks 17, 18
  - **Blocked By**: Task 12 (package structure)

  **References**:
  - OpenCV VideoCapture: `cap = cv2.VideoCapture(path); ret, frame = cap.read()`
  - Image glob: `sorted(Path(dir).glob('*.png'))`
  - Existing `run_vo.py` 中 `load_images()` 函數需要被取代

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_io.py -v` → 4+ tests pass
  - [ ] FrameLoader 能 auto-detect directory vs video file
  - [ ] max_frames 限制被正確遵守
  - [ ] 不支援的路徑產生清晰的 IndexError/FileNotFoundError

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Image directory loading
    Tool: Bash (pytest)
    Steps:
      1. Create temp dir with 5 test images (different colors)
      2. loader = FrameLoader(temp_dir)
      3. frames = list(loader)
      4. Assert len(frames) == 5
      5. Assert all frames are uint8, shape (H, W, 3)
    Expected: All images loaded in sorted order
    Evidence: .omo/evidence/task-13-image-dir.txt

  Scenario: Video file loading
    Tool: Bash (pytest)
    Steps:
      1. Create synthetic video (10 frames) using cv2.VideoWriter
      2. loader = FrameLoader(video_path)
      3. frames = list(loader)
      4. Assert len(frames) == 10
    Expected: All video frames extracted
    Evidence: .omo/evidence/task-13-video.txt
  ```

  **Commit**: NO (groups with Wave 5)

- [x] 14. Dual Trajectory Export (KITTI + TUM)

  **What to do**:
  - 建立 `slam_dnn/export.py`：
    - `save_trajectory_kitti(poses: list[np.ndarray], filepath: str)` — 保存為 KITTI 格式
      - 每行 12 個 floats: row-major 3x4 matrix (r00 r01 r02 t0 r10 r11 r12 t1 r20 r21 r22 t2)
    - `save_trajectory_tum(poses: list[np.ndarray], timestamps: list[float] | None, filepath: str)` — 保存為 TUM RGB-D 格式
      - 每行: `timestamp tx ty tz qx qy qz qw`
      - 如果未提供 timestamps，使用 frame index (0, 1, 2, ...)
      - 使用 `scipy.spatial.transform.Rotation.from_matrix().as_quat()` 轉換
    - `load_trajectory_kitti(filepath: str) -> list[np.ndarray]` — 載入 KITTI 格式
    - `load_trajectory_tum(filepath: str) -> tuple[list[np.ndarray], list[float]]` — 載入 TUM 格式
  - 更新 `TrajectoryAccumulator`：
    - 增加 `save(self, filepath: str, format: str = 'kitti')` 便捷方法
  - 在 `tests/test_export.py` 寫測試：
    - Round-trip: save → load → compare (within numerical precision)
    - KITTI format: verify each line has exactly 12 floats
    - TUM format: verify each line has timestamp + 3 positions + 4 quaternion values
    - Identity pose round-trip correctness

  **Must NOT do**:
  - 不支援其他格式（e.g., EuRoC, custom）
  - 不做 timestamp 對齊（evaluation 留給 Task 15）
  - 不加入 JSON/CSV 格式

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 標準格式轉換，清晰明確
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Tasks 12, 13)
  - **Blocks**: Tasks 15, 17, 18
  - **Blocked By**: Task 12 (package structure)

  **References**:
  - KITTI format: https://www.cvlibs.net/datasets/kitti/eval_odometry.php — 12 floats per line
  - TUM format: `timestamp tx ty tz qx qy qz qw` (8 values per line)
  - Quaternion from rotation matrix: `scipy.spatial.transform.Rotation.from_matrix(R).as_quat()` gives (x,y,z,w)

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_export.py -v` → 4+ tests pass
  - [ ] KITTI round-trip: save → load → poses match (atol=1e-6)
  - [ ] TUM round-trip: save → load → poses + timestamps match
  - [ ] KITTI output: each line has exactly 12 space-separated floats

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: KITTI format round-trip
    Tool: Bash (pytest)
    Steps:
      1. Generate 5 known poses (identity + 4 random)
      2. save_trajectory_kitti(poses, "/tmp/test_kitti.txt")
      3. loaded = load_trajectory_kitti("/tmp/test_kitti.txt")
      4. For each pair: np.testing.assert_allclose(orig, loaded, atol=1e-6)
    Expected: Exact round-trip preservation
    Evidence: .omo/evidence/task-14-kitti-roundtrip.txt

  Scenario: TUM format validation
    Tool: Bash (pytest)
    Steps:
      1. Generate poses with timestamps
      2. save_trajectory_tum(poses, timestamps, "/tmp/test_tum.txt")
      3. Read file line by line
      4. Each line split() → 8 parts, all parseable as float
    Expected: 8 values per line, all valid floats
    Evidence: .omo/evidence/task-14-tum-format.txt
  ```

  **Commit**: YES (Wave 5 group)
  - Message: `feat: FrameLoader for images+video, dual KITTI+TUM trajectory export`
  - Files: `slam_dnn/io.py`, `slam_dnn/export.py`, `tests/test_io.py`, `tests/test_export.py`

### Wave 6: Evaluation + Robustness (After Wave 5)

- [x] 15. Umeyama Alignment + APE/RTE Evaluation

  **What to do**:
  - 建立 `slam_dnn/eval.py`：
    - `align_umeyama(estimated: np.ndarray, ground_truth: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]`
      - Sim(3) 對齊：回傳 (scale, R, t) 使得 `aligned = s * R @ estimated + t`
      - Based on Umeyama 1991 algorithm
      - `with_scale=False` → rigid alignment (no scale, SE(3))
    - `align_trajectories(est_poses: list[np.ndarray], gt_poses: list[np.ndarray], with_scale=True) -> list[np.ndarray]` — convenience function，對齊整個 pose 列表
    - `compute_ape(estimated_pos: np.ndarray, ground_truth_pos: np.ndarray) -> np.ndarray` — Absolute Pose Error per frame
    - `compute_rte(estimated_pos: np.ndarray, ground_truth_pos: np.ndarray, window: float = 5.0) -> np.ndarray` — Relative Trajectory Error with sliding window
    - `evaluate(estimated: list[np.ndarray], ground_truth: list[np.ndarray], with_scale=True) -> dict` — 回傳完整的 evaluation report：
      ```python
      {"ape_rmse": float, "rte_rmse": float, "scale": float, "aligned_poses": list}
      ```
  - 在 `tests/test_eval.py` 寫測試：
    - Umeyama 對齊驗證：已知 scale=2.0 + rotation，對齊後應完全吻合
    - APE 驗證：零誤差 → APE=0
    - RTE 驗證：constant drift → RTE grows linearly
    - 測試 `with_scale=True` vs `with_scale=False` 的差異

  **Must NOT do**:
  - 不加入 EVO 或第三方 trajectory evaluation library
  - 不處理 timestamp 對齊（假設 pose 列表 1:1 對應）
  - 不做 robust alignment（RANSAC-based outlier rejection）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Umeyama algorithm 需要精確的 SVD 實作
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6 (with Tasks 16, 17)
  - **Blocks**: Task 19
  - **Blocked By**: Task 14 (TUM export format needed for evaluation)

  **References**:
  - Umeyama 1991: "Least-squares estimation of transformation parameters", IEEE PAMI
  - vo-pipeline `estimate_sim3_umeyama()`: SVD-based alignment with reflection fix
  - APE formula: `||p_est - p_gt||` per frame, then RMSE
  - RTE formula: sliding window of fixed distance, compute `||Δp_est - Δp_gt||`
  - Reflection fix: if `det(R) < 0`, flip last singular vector

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_eval.py -v` → 4+ tests pass
  - [ ] 已知 scale=2.0 的對齊：恢復 scale ≈ 2.0 (atol=1e-6)
  - [ ] 完全相同的 trajectory：APE=0, RTE=0
  - [ ] evaluate() 回傳 dict 包含 ape_rmse, rte_rmse, scale

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Umeyama alignment recovers known transform
    Tool: Bash (pytest)
    Steps:
      1. ground_truth = np.array of 10 positions along a straight line
      2. Apply known transform: scale=2.0, R=rodrigues([0,0.2,0]), t=[1,2,3]
      3. estimated = s*R@gt + t for each point
      4. s_est, R_est, t_est = align_umeyama(estimated, ground_truth, with_scale=True)
      5. Assert abs(s_est - 2.0) < 0.01
      6. Assert rotation error < 0.5°
    Expected: Recovers applied transform accurately
    Evidence: .omo/evidence/task-15-umeyama.txt

  Scenario: APE/RTE on zero-error trajectories
    Tool: Bash (pytest)
    Steps:
      1. pos = some trajectory positions
      2. ape = compute_ape(pos, pos)
      3. Assert all(ape < 1e-10)
      4. rte = compute_rte(pos, pos)
      5. Assert all(rte < 1e-10)
    Expected: Zero error for identical trajectories
    Evidence: .omo/evidence/task-15-zero-error.txt
  ```

  **Commit**: NO (groups with Wave 6)

- [x] 16. Edge Case Handling (Pure Rotation, Tracking Lost)

  **What to do**:
  - 新增 edge case handling 到 `slam_dnn/pose.py`：
    - `detect_pure_rotation(points0: np.ndarray, points1: np.ndarray, K: np.ndarray) -> bool`
      - 透過 Homography inlier 比率判斷：`H, mask_H = cv2.findHomography(pts0, pts1, cv2.RANSAC)`
      - 如果 H inlier count >> E inlier count → pure rotation
      - 回傳 True/False
    - `estimate_essential_or_homography(points0, points1, K, ...)` — 改良的 pose estimation：
      - 同時嘗試 E-matrix 和 Homography
      - 如果 pure rotation detected → 只回傳 R（t=zeros）
      - 否則正常 E-matrix recovery
  - 新增 tracking robustness 到 `slam_dnn/vo.py` 的 `VisualOdometry.process_frame`:
    - `len(matches) < MIN_MATCHES (20)` → raise `TrackingLostError` 或 log warning + skip
    - `estimate_essential` returns None → 同上
    - 增加 frame-by-frame 統計記錄：
      ```python
      # Track: num_matches, num_inliers, reprojection_error (if available)
      stats: list[dict]  # per-frame stats
      ```
  - 新增 `slam_dnn/config.py` — `VOConfig` dataclass：
    ```python
    @dataclass
    class VOConfig:
        max_keypoints: int = 2048
        detection_threshold: float = 0.0005
        matcher: str = 'lightglue'
        lightglue_threshold: float = 0.1
        classic_ratio: float = 0.75
        ransac_threshold: float = 1.0
        ransac_confidence: float = 0.999
        min_matches: int = 20
        scale: float = 1.0
        fov_deg: float = 63.0
        handle_pure_rotation: bool = True
        device: str = 'auto'
    ```
  - 在 `tests/test_edge_cases.py` 寫測試：
    - Pure rotation detection（合成 pure rotation 場景）
    - Low match count handling（空圖片對 → TrackingLostError）
    - VOConfig 的 defaults 驗證

  **Must NOT do**:
  - 不做 Homography decomposition（只 detect pure rotation，回傳 t=0 的近似）
  - 不做 feature-level tracking persistence（每幀重新 extract）
  - 不做 sliding window BA 或 pose refinement

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要細心處理 multiple edge cases 和 graceful degradation
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6 (with Tasks 15, 17)
  - **Blocks**: Task 17
  - **Blocked By**: Task 12 (vo.py must exist)

  **References**:
  - Pure rotation detection: compare H inliers vs E inliers (H_ratio > 0.5 → likely pure rotation)
  - Homography estimation: `cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)`
  - minislam features.py: `assert len(good) > 8` pattern
  - vo-from-scratch: `if not success or len(inliers) <= MIN_INLIERS: raise ...`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_edge_cases.py -v` → 3+ tests pass
  - [ ] Pure rotation scene detected correctly (H inliers >> E inliers)
  - [ ] Empty scene (0 matches) → TrackingLostError raised
  - [ ] VOConfig() 的 defaults 匹配計劃中定義的值

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Pure rotation detection
    Tool: Bash (pytest)
    Steps:
      1. Create 3D points at z=5~15
      2. Apply pure rotation (R, t=[0,0,0]) to get pts1, pts2
      3. result = detect_pure_rotation(pts1, pts2, K)
      4. Assert result is True
    Expected: Pure rotation correctly identified
    Evidence: .omo/evidence/task-16-pure-rotation.txt

  Scenario: Tracking lost gracefully handled
    Tool: Bash (pytest)
    Steps:
      1. cam = PinholeCamera(640, 480, 63)
      2. vo = VisualOdometry(cam, device='cpu')
      3. blank_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
      4. vo.process_frame(image1)  # normal image
      5. vo.process_frame(blank_img)  # no features expected
      6. Verify: TrackingLostError is raised or None returned (depending on config)
    Expected: Graceful handling, no crash
    Evidence: .omo/evidence/task-16-tracking-lost.txt
  ```

  **Commit**: NO (groups with Wave 6)

- [x] 17. Comprehensive Unit Tests (Teaching Examples)

  **What to do**:
  - 擴充所有 `tests/test_*.py`，確保每個 module 有 3+ teaching-quality tests：
    - **test_camera.py**: K_from_fov variants (different resolutions, FOV angles), K_inv correctness, PinholeCamera dataclass
    - **test_features.py**: Max keypoints limiting, detection threshold sensitivity, BGR vs grayscale input
    - **test_matching.py**: MatcherBase compliance (both matchers), match count consistency, edge case (0 features)
    - **test_pose.py**: Large rotation test(30°), noisy correspondence test, essential matrix properties verification
    - **test_trajectory.py**: Rotation-only trajectory, circular trajectory, reset() behavior
    - **test_io.py**: Unsupported file types, empty directory, max_frames edge (max=0)
    - **test_export.py**: Large trajectory (100+ poses), TUM with and without timestamps
    - **test_eval.py**: Constant scale offset (no rotation), partial overlap handling
    - **test_vo.py** (new): VisualOdometry full pipeline test, matcher switching, config override
  - 每個 test function 需有：
    - 清晰的 docstring 解釋「這個 test 在驗證什麼概念」
    - 命名規則：`test_{module}_{scenario}_{expected}` 或 `test_{concept_description}`
    - 步驟清晰（適合學生閱讀理解 pipeline 各階段）
  - 建立 `tests/conftest.py` 共享 fixtures：
    - `sample_k_camera` — PinholeCamera(640, 480, 63)
    - `sample_image_pair` — 合成的有特徵的圖片對
    - `sample_3d_points` — 50 個隨機 3D 點

  **Must NOT do**:
  - 不要求 100% code coverage
  - 不 mock SuperPoint 或 LightGlue（用真實 model 跑，接受較慢速度）
  - 不加 GPU-specific tests（所有測試用 CPU）
  - 不做 property-based testing (hypothesis)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要撰寫大量清晰的測試作為教學範例
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (needs all modules from T12-T16)
  - **Parallel Group**: Wave 6 (sequential after T15, T16)
  - **Blocks**: Task 19
  - **Blocked By**: Tasks 12, 14, 15, 16

  **References**:
  - All `tests/test_*.py` files from previous waves
  - pytest fixtures pattern: `@pytest.fixture` in conftest.py
  - numpy.testing: `assert_allclose`, `assert_array_equal`
  - Teaching quality: each test should be readable as a standalone example

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/ -v` → 30+ tests pass (at least 3 per module)
  - [ ] Each test has descriptive docstring
  - [ ] conftest.py has at least 3 shared fixtures
  - [ ] Test names follow consistent pattern

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Full test suite passes
    Tool: Bash (pytest)
    Steps:
      1. python -m pytest tests/ -v --tb=short 2>&1 | tee .omo/evidence/task-17-full-suite.txt
      2. Count total tests (should be 30+)
      3. Verify 0 failures
    Expected: All tests pass, organized output
    Evidence: .omo/evidence/task-17-full-suite.txt

  Scenario: Test docstrings are informative
    Tool: Bash (pytest)
    Steps:
      1. python -m pytest tests/ -v --collect-only
      2. Verify test names are descriptive (not just test_1, test_2)
    Expected: Clear, meaningful test names
    Evidence: .omo/evidence/task-17-test-names.txt
  ```

  **Commit**: YES (Wave 6 group)
  - Message: `feat: Umeyama alignment, APE/RTE evaluation, edge case handling, comprehensive test suite`
  - Files: `slam_dnn/eval.py`, `slam_dnn/config.py`, `slam_dnn/pose.py`, `slam_dnn/vo.py`, `tests/`
  - Pre-commit: `python -m pytest tests/ -v`

### Wave 7: Polish (After Wave 6)

- [x] 18. CLI Entry Point + Logging

  **What to do**:
  - 建立 `slam_dnn/__main__.py` — 讓 `python -m slam_dnn` 可用：
    - 將 `run_vo.py` 的邏輯搬到 `slam_dnn/cli.py`
    - `__main__.py` 只是 `from slam_dnn.cli import main; main()`
    - CLI 參數：
      ```
      --input PATH          Image directory or video file (required)
      --output PATH         Output directory (required)
      --fov FLOAT           Horizontal FOV in degrees (default: 63.0)
      --matcher STR         lightglue | classic (default: lightglue)
      --max-keypoints INT   Max keypoints per frame (default: 2048)
      --scale FLOAT         Translation scale factor (default: 1.0)
      --device STR          auto | cuda | cpu (default: auto)
      --verbose / -v        Enable verbose logging
      --quiet               Suppress all output except errors
      --evaluate GT_PATH    Evaluate against ground truth trajectory (KITTI or TUM format)
      ```
    - 輸出目錄結構：
      ```
      {output}/
      ├── trajectory_kitti.txt
      ├── trajectory_tum.txt
      ├── trajectory_plot.png
      ├── trajectory_aligned.txt  (only if --evaluate)
      └── evaluation_report.txt   (only if --evaluate)
      ```
  - 整合 Python `logging` module：
    - `--verbose`: logging.DEBUG
    - default: logging.INFO（frame progress, match count, etc.）
    - `--quiet`: logging.ERROR only
  - 刪除 `run_vo.py`（已由 `python -m slam_dnn` 取代）
  - 在 `tests/test_cli.py` 寫測試：
    - `python -m slam_dnn --help` 成功
    - 使用 subprocess 呼叫 CLI 處理合成圖片序列
    - 驗證輸出目錄結構

  **Must NOT do**:
  - 不做 GUI 或 web interface
  - 不做 configuration file loading（CLI args 即可）
  - 不做 multi-process or async processing

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 將現有 run_vo.py 包裝成正式的 CLI module
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 7 (with Task 19)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 13, 14

  **References**:
  - Python `__main__.py` pattern for `python -m package` execution
  - `argparse` module for CLI parsing
  - Python `logging` module: `logging.basicConfig(level=...)`
  - Existing `run_vo.py` to be migrated

  **Acceptance Criteria**:
  - [ ] `python -m slam_dnn --help` 顯示完整說明
  - [ ] `python -m slam_dnn --input <dir> --output <dir>` 成功執行
  - [ ] 輸出目錄包含 trajectory_kitti.txt + trajectory_tum.txt + trajectory_plot.png
  - [ ] `--evaluate` 生成 evaluation_report.txt
  - [ ] `--verbose` 和 `--quiet` 正確控制 log level

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: CLI produces all expected outputs
    Tool: Bash
    Steps:
      1. Create synthetic image sequence (10 PNG files in temp dir)
      2. python -m slam_dnn --input <temp_dir> --output <out_dir> --fov 63 --device cpu
      3. Verify: out_dir/trajectory_kitti.txt exists
      4. Verify: out_dir/trajectory_tum.txt exists
      5. Verify: out_dir/trajectory_plot.png exists
      6. Verify: trajectory_kitti.txt has 9 lines (10 frames = 9 relative poses)
    Expected: All output files generated correctly
    Evidence: .omo/evidence/task-18-cli-outputs.txt

  Scenario: CLI --help works
    Tool: Bash
    Steps:
      1. python -m slam_dnn --help
      2. Verify output contains all 10 argument descriptions
    Expected: Help text complete and accurate
    Evidence: .omo/evidence/task-18-cli-help.txt

  Scenario: CLI --evaluate generates report
    Tool: Bash
    Steps:
      1. Generate ground truth trajectory file
      2. python -m slam_dnn --input <images> --output <out> --evaluate <gt_file>
      3. Verify evaluation_report.txt exists and contains APE/RTE metrics
    Expected: Evaluation runs and produces report
    Evidence: .omo/evidence/task-18-cli-evaluate.txt
  ```

  **Commit**: YES
  - Message: `feat: CLI entry point with logging and evaluation support`
  - Files: `slam_dnn/__main__.py`, `slam_dnn/cli.py`, `run_vo.py` (deleted), `tests/test_cli.py`

- [x] 19. README + Docstrings + Usage Examples

  **What to do**:
  - 撰寫 `README.md`：
    ```markdown
    # SLAM-DNN: SuperPoint-based Visual Odometry
    
    A simple, educational visual odometry library using SuperPoint features.
    
    ## Features
    - SuperPoint feature extraction
    - Dual matcher: LightGlue (neural) or Classic (BF + Lowe's ratio)
    - Essential matrix pose estimation
    - SE3 trajectory accumulation
    - KITTI + TUM trajectory export
    - Umeyama alignment + APE/RTE evaluation
    - Pinhole camera from FOV (default: 63° phone wide-angle)
    
    ## Installation
    pip install -r requirements.txt
    pip install -e .
    
    ## Quick Start
    # Image directory
    python -m slam_dnn --input path/to/images --output output/ --fov 63.0
    
    # Video file
    python -m slam_dnn --input path/to/video.mp4 --output output/
    
    # With classic matcher
    python -m slam_dnn --input images/ --output output/ --matcher classic
    
    # Evaluate against ground truth
    python -m slam_dnn --input images/ --output output/ --evaluate gt_poses.txt
    
    ## Usage as Library
    ```python
    from slam_dnn import VisualOdometry, PinholeCamera
    
    camera = PinholeCamera(width=640, height=480, fov_deg=63.0)
    vo = VisualOdometry(camera, matcher='lightglue', device='auto')
    
    # Process frames one by one
    for image in my_images:
        pose = vo.process_frame(image)  # 4x4 matrix or None
        print(f"Current position: {vo.get_trajectory().get_positions()[-1]}")
    
    # Or process entire sequence
    from slam_dnn import FrameLoader
    frames = FrameLoader('path/to/images')
    poses = vo.process_sequence(frames)
    ```
    
    ## Architecture
    (簡短描述模組結構)
    
    ## Running Tests
    python -m pytest tests/ -v
    
    ## Limitations
    - Monocular VO → unit-norm translation (unknown scale)
    - No loop closure, no SLAM, no landmark management
    - Assumes pinhole camera with no distortion
    ```
  - 為所有 public functions/classes 加入 docstrings：
    - 格式：Google style docstring
    - 內容：說明用途、參數、回傳值、raises、example
    - 特別註明數學公式（如 SE3 composition、FOV→K）供教學用
  - 確保 `pyproject.toml` 有完整的 package metadata（description, author, license, classifiers）

  **Must NOT do**:
  - 不做 Sphinx / MkDocs 格式文件
  - 不加入 GIF 或動畫（沒有影片資源）
  - 不做 contribution guide（非開源專案需求）
  - 不寫 CHANGELOG

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 主要是文件撰寫任務
  - **Skills**: none needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 7 (with Task 18)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 15, 17, 18

  **References**:
  - All existing module docstrings to be enhanced
  - Python Google-style docstrings: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
  - Standard README structure: Features, Install, Usage, API, Tests, Limitations

  **Acceptance Criteria**:
  - [ ] `README.md` 存在且包含所有章節（Features, Install, Quick Start, Usage as Library, Tests, Limitations）
  - [ ] 所有 public API（至少 15 個）有 Google-style docstring
  - [ ] CLI usage example 實際可執行（照著 README 打就動）
  - [ ] `pip install -e .` 仍然工作

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: README code examples are valid
    Tool: Bash
    Steps:
      1. Extract Python code block from README "Usage as Library" section
      2. Save to /tmp/test_readme_example.py (with mock images)
      3. python /tmp/test_readme_example.py → no ImportError or AttributeError
    Expected: README code runs without modifications
    Evidence: .omo/evidence/task-19-readme-code.txt

  Scenario: Docstring coverage check
    Tool: Bash
    Steps:
      1. python -c "import slam_dnn; help(slam_dnn)" | head -50
      2. Verify docstrings present for VisualOdometry, PinholeCamera, K_from_fov
    Expected: Public functions have docstrings
    Evidence: .omo/evidence/task-19-docstrings.txt

  Scenario: Full installation from scratch
    Tool: Bash
    Steps:
      1. pip install -e .
      2. python -c "import slam_dnn; print(slam_dnn.__name__)"
      3. python -m slam_dnn --help
      4. python -m pytest tests/ --tb=short -q
    Expected: Clean install, import works, CLI works, tests pass
    Evidence: .omo/evidence/task-19-install-check.txt
  ```

  **Commit**: YES (final Phase 2)
  - Message: `docs: README, docstrings, usage examples, complete documentation`
  - Files: `README.md`, `pyproject.toml`, all `slam_dnn/*.py` (docstring additions)
  - Pre-commit: `python -m pytest tests/ -v`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `python -m pytest tests/ -v` + check for `type: ignore`, `as any` equivalents, empty catches, print statements in non-prototype code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Process a real image sequence through CLI. Verify both KITTI and TUM outputs are generated. Run synthetic tests. Test edge cases (low texture, pure rotation if possible). Save to `.omo/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | CLI [PASS/FAIL] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Phase 1 complete** (T1-T11): `feat: working visual odometry prototype with SuperPoint`
- **Phase 2 modular** (T12-T14): `refactor: modular package structure with switchable matchers`
- **Phase 2 eval** (T15-T17): `feat: Umeyama alignment, APE/RTE evaluation, comprehensive tests`
- **Phase 2 polish** (T18-T19): `docs: CLI, logging, README, complete documentation`

---

## Success Criteria

### Verification Commands
```bash
# Phase 1 verification
python run_vo.py --input data/sample/ --output output/test/   # Expect: trajectory_kitti.txt + plot
python -m pytest tests/test_synthetic.py -v                     # Expect: 3+ tests pass

# Phase 2 verification
python -m slam_dnn --input data/sample/ --output output/test/ --fov 63.0
python -m pytest tests/ -v                                      # Expect: all tests pass
ls output/test/trajectory_kitti.txt output/test/trajectory_tum.txt  # Expect: both exist
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent (no SLAM, no landmarks, no BA, no loop closure)
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] CLI produces both KITTI and TUM trajectory files
- [ ] Synthetic pose recovery: rotation error < 1°, translation direction error < 2°
- [ ] README with installation + working example
- [ ] Both matchers work via `--matcher lightglue` and `--matcher classic`
