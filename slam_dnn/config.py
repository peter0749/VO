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
    matcher: str = 'lightglue'       # 'lightglue' or 'classic'
    lightglue_threshold: float = 0.1
    classic_ratio: float = 0.75
    ransac_threshold: float = 1.0
    ransac_confidence: float = 0.999
    min_matches: int = 20
    scale: float = 1.0
    fov_deg: float = 63.0
    handle_pure_rotation: bool = True
    device: str = 'auto'              # 'auto' | 'cuda' | 'mps' | 'cpu'
