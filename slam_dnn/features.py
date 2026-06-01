"""Feature extraction (SuperPoint wrapper)."""

import numpy as np
import torch
from lightglue import SuperPoint


class SuperPointExtractor:
    """Wraps LightGlue's SuperPoint for feature extraction."""

    def __init__(
        self,
        nms_radius: int = 4,
        max_keypoints: int = 1024,
        conf_thresh: float = 0.005,
        device: str = "auto",
    ):
        """
        Args:
            nms_radius: Non-maximum suppression radius (default 4).
            max_keypoints: Maximum keypoints per image (default 1024).
            conf_thresh: Keypoint confidence threshold (default 0.005).
            device: "auto" | "cuda" | "cpu".
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.superpoint = SuperPoint(
            max_num_keypoints=max_keypoints,
            detection_threshold=conf_thresh,
            nms_radius=nms_radius,
        ).to(self.device).eval()

    def extract(self, image: np.ndarray) -> dict:
        """Extract SuperPoint features from an image.

        Args:
            image: numpy array, shape (H, W) or (H, W, 3), dtype uint8.

        Returns:
            dict with:
                - "keypoints": (N, 2) float32, pixel coordinates (x, y).
                - "descriptors": (N, 256) float32, L2-normalized.
                - "scores": (N,) float32, confidence scores.
        """
        # Convert uint8 → float32 in [0, 1]
        img = image.astype(np.float32) / 255.0

        # Build (1, C, H, W) tensor
        if img.ndim == 2:
            # Grayscale: (H, W) → (1, 1, H, W)
            tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        elif img.ndim == 3:
            # Color: (H, W, 3) → (1, 3, H, W)
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

        tensor = tensor.to(self.device)

        with torch.no_grad():
            pred = self.superpoint({"image": tensor})

        # Remove batch dimension
        keypoints = pred["keypoints"][0].cpu().numpy()        # (N, 2)
        descriptors = pred["descriptors"][0].cpu().numpy()    # (N, 256)
        scores = pred["keypoint_scores"][0].cpu().numpy()    # (N,)

        # Ensure float32
        keypoints = keypoints.astype(np.float32)
        descriptors = descriptors.astype(np.float32)
        scores = scores.astype(np.float32)

        # Re-normalize descriptors (safety: model normalizes internally,
        # but float32 casting can introduce tiny drift)
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        descriptors = descriptors / norms

        return {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "scores": scores,
        }
