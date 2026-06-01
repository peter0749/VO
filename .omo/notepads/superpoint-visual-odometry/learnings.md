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
