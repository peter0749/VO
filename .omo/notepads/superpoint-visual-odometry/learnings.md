# SuperPoint VO Project Learnings

## Session Continuation — Inherited from Waves 1-4 (commit 2f6cb44)

### Platform & Environment
- macOS, zsh shell, Python 3.14.5 via homebrew (`python3`, NOT `python`)
- Apple Silicon (MPS available but tests should work on CPU)
- LightGlue installed via `pip install git+https://github.com/cvg/LightGlue`
- pyproject.toml exists, package is editable-installed via `pip install -e .`

### CRITICAL INTERFACES (verified as of T11)
1. `SuperPointExtractor.extract(image)` → dict with:
   - `keypoints`: (N, 2) float32 — pixel coords (x, y)
   - `descriptors`: (N, 256) float32 — L2-normalized ✓
   - `scores`: (N,) float32
2. `LightGlueMatcher.match(feats0, feats1)` returns:
   - `points0`, `points1` (K, 2), `scores` (K,), `indices` (K, 2)
3. `ClassicMatcher` — same interface, scores normalized [0,1] higher=better
4. `estimate_essential(points0, points1, K)` → (R, t, inlier_mask) or None
5. `TrajectoryAccumulator`:
   - `add_pose(R, t)` — normalizes t, applies scale
   - `get_poses()` — list of 4x4 SE3, starts with identity
   - `get_positions()` — (N, 3) camera centers = -R.T @ t

### THE DESCRIPTOR SHAPE BUG (T12 MUST FIX)
- LightGlue library expects descriptors shape (B, N, D=256) [see LightGlue source line 510]
- SuperPointExtractor returns (N, 256) which is CORRECT
- LightGlueMatcher currently does `.T.unsqueeze(0)` expecting (N, 256) input
- Result: `(256, N, 1)` shape after transpose+unsqueeeze — WRONG
- WORKAROUND exists in `run_vo.py`: transpose descriptors before passing to matcher
- FIX: Remove `.T` in LightGlueMatcher (just `.unsqueeze(0)`), remove workaround in run_vo.py

### Test Infrastructure
- pytest with `numpy.testing` as `npt`
- `tests/fixtures/kitti_05_subset/` — 50 synthetic PNGs (640x360)
- Current test count: 97 (67 baseline + T10: 8 + T11: 22)
- Full suite > 5min runtime
- Run with: `python3 -m pytest tests/ -x -q`

### Subagent Lessons
- ALWAYS verify file existence + line counts + test output after subagent claims "done"
- T7 and T8 subagents reported complete but delivered NO code — caught by verification
- Redo tasks require explicit listing: "Previous attempt delivered nothing. Here's proof."
- Use `python3` consistently — `python` symlink doesn't exist on macOS

### Git State
- 5 commits on master (scaffolding, features+matching, pose+trajectory, prototype, wave4-validation)
- Next commit should be: T12 (refactor) or combine with T13+T14 after Wave 5

### Upcoming Wave 5 Tasks
- **T12**: MatcherBase ABC + VisualOdometry facade + fix descriptor bug (dispatched, in progress)
- **T13**: FrameLoader (image dirs + videos) — depends on T12
- **T14**: KITTI + TUM dual export on TrajectoryAccumulator — depends on T12

## Session Continuation — Wave 5 Lessons (commit 72b9745)

### Parallel Subagent Conflict: __init__.py race
- T13 (FrameLoader) and T14 (export.py) both tried to modify `slam_dnn/__init__.py`
- T13's write happened AFTER T14's → T14's 4 export imports were LOST
- Result: `ImportError: cannot import name 'export_kitti_format' from 'slam_dnn'`
- Fix: orchestrator had to manually merge the missing imports after verification

**TAKEAWAY for future waves**: When multiple subagents need to modify a shared file like `__init__.py`:
- Either instruct them to SKIP that file and have orchestrator do a final merge pass
- Or explicitly mark one task as owning the shared file (others append to a notepad list)

### Current Public API (after Wave 5, 13 exports in __init__.py)
```python
K_from_fov, PinholeCamera, TrackingLostError,
SuperPointExtractor, ClassicMatcher, LightGlueMatcher, MatcherBase, create_matcher,
estimate_essential, TrajectoryAccumulator, VisualOdometry,
FrameLoader, to_grayscale, to_float,
export_kitti_format, export_tum_format, load_kitti_format, load_tum_format,
visualization (submodule)
```

### Test Count After Wave 5
- Total core unit tests (excluding slow synthetic): ~98
- Core test suite runtime: ~12s (fast)
- Full suite (including synthetic): ~90s

### Upcoming Wave 6 Tasks
- **T15**: Umeyama alignment + APE/RTE evaluation in slam_dnn/eval.py
- **T16**: Edge case handling — pure rotation (homography fallback), tracking lost recovery
- **T17**: Comprehensive test polish + conftest fixtures

### File Layout After Wave 5
```
slam_dnn/
├── __init__.py (18 exports)
├── camera.py
├── exceptions.py
├── export.py (NEW - T14)
├── features.py
├── io.py (NEW - T13)
├── matching.py (MatcherBase added - T12)
├── pose.py
├── trajectory.py (.save() method added - T14)
├── visualization.py
└── vo.py (NEW - T12)
```
