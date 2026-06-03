"""VO configuration dataclass with validated defaults."""
from dataclasses import dataclass


@dataclass
class VOConfig:
    """Configuration for VisualOdometry pipeline.

    All parameters have sensible defaults for a typical mobile phone
    camera (e.g., iPhone wide-angle ~63° FOV).
    """
    max_keypoints: int = 2048
    detection_threshold: float = 0.0005
    extractor: str = 'superpoint'     # 'superpoint' or 'xfeat'
    matcher: str = 'lightglue'       # 'lightglue' or 'classic' or 'xfeat'
    lightglue_threshold: float = 0.1
    classic_ratio: float = 0.75
    xfeat_min_cossim: float = 0.82    # minimum cosine similarity for xfeat matching
    ransac_threshold: float = 1.0
    ransac_confidence: float = 0.999
    min_matches: int = 20
    scale: float = 1.0
    fov_deg: float = 63.0
    handle_pure_rotation: bool = True
    device: str = 'auto'              # 'auto' | 'cuda' | 'mps' | 'cpu'
    
    # Keyframe Selection Heuristics
    use_keyframe_selection: bool = True
    min_parallax: float = 8.0         # minimum median parallax (pixels) to trigger keyframe
    max_overlap: float = 0.85         # maximum overlap ratio before keyframe is forced
    max_keyframe_interval: int = 10   # maximum consecutive frames before forcing keyframe
    
    # Constant Velocity Motion Model
    use_motion_model: bool = True
    motion_model_alpha: float = 0.5   # velocity smoothing EMA factor

    # Speed Optimizations
    target_resolution: int | None = None  # if not None, image is resized so max(H, W) <= target_resolution
