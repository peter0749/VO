"""Feature extraction (SuperPoint wrapper)."""

import numpy as np
import torch
import cv2
from lightglue import SuperPoint


class SuperPointExtractor:
    """Wrapper around LightGlue's SuperPoint model for feature extraction.

    SuperPoint detects keypoints and computes 256-dimensional descriptors
    using a VGG backbone trained in a self-supervised manner. The wrapper
    handles device placement, dtype normalization, and batch dimension
    management.

    Attributes:
        device: Torch device where the model runs ('cuda', 'mps', or 'cpu').

    Args:
        nms_radius: Non-maximum suppression radius in pixels. Larger values
            produce fewer, more spread-out keypoints. Default 4.
        max_keypoints: Cap on total keypoints per image. Default 1024.
        conf_thresh: Minimum detection confidence. Lower values detect more
            keypoints at the cost of noise. Default 0.005.
        device: Device string. 'auto' selects cuda if available, else cpu.

    Example:
        >>> extractor = SuperPointExtractor(max_keypoints=512, device='cpu')
        >>> feats = extractor.extract(gray_image)
        >>> feats['keypoints'].shape  # (N, 2)
        >>> feats['descriptors'].shape  # (N, 256)
    """

    def __init__(
        self,
        nms_radius: int = 4,
        max_keypoints: int = 1024,
        conf_thresh: float = 0.005,
        device: str = "auto",
        target_resolution: int | None = None,
    ):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.target_resolution = target_resolution

        self.superpoint = SuperPoint(
            max_num_keypoints=max_keypoints,
            detection_threshold=conf_thresh,
            nms_radius=nms_radius,
        ).to(self.device).eval()

    def extract(self, image: np.ndarray) -> dict:
        """Extract SuperPoint features from an image.

        Converts uint8 input to float32 in [0, 1], and runs the model in
        a single forward pass. Keypoints are returned in pixel coordinates
        (x, y) with descriptors L2-normalized to unit length.

        Args:
            image: Grayscale or color image as uint8 ndarray. Shape (H, W)
                for grayscale or (H, W, 3) for BGR/RGB.

        Returns:
            Dictionary with:
                - "keypoints": ndarray (N, 2) float32, pixel coordinates (x, y).
                - "descriptors": ndarray (N, 256) float32, L2-normalized.
                - "scores": ndarray (N,) float32, per-keypoint confidence.

        Raises:
            ValueError: If image is not 2D or 3D.

        Example:
            >>> extractor = SuperPointExtractor(max_keypoints=512, device='cpu')
            >>> img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            >>> feats = extractor.extract(img)
            >>> print(feats['keypoints'].shape, feats['descriptors'].shape)
            (N, 2) (N, 256)
        """
        h, w = image.shape[:2]
        scale_factor = 1.0
        if self.target_resolution is not None and max(h, w) > self.target_resolution:
            scale_factor = self.target_resolution / max(h, w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            resized_img = cv2.resize(image, (new_w, new_h))
            img = resized_img.astype(np.float32) / 255.0
        else:
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

        # Rescale keypoints back to original resolution if scaled
        if scale_factor != 1.0:
            keypoints = keypoints / scale_factor

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


class XFeatExtractor:
    """Wrapper around XFeat model for feature extraction.

    XFeat is a lightweight CNN that detects keypoints, scores, and extracts
    64-dimensional descriptors in real-time.
    """

    def __init__(
        self,
        max_keypoints: int = 2048,
        conf_thresh: float = 0.05,
        device: str = "auto",
        target_resolution: int | None = None,
    ):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.target_resolution = target_resolution

        # Dynamic import of XFeat
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        xfeat_path = project_root / "slam_dnn" / "thirdparty" / "accelerated_features"
        if str(xfeat_path) not in sys.path:
            sys.path.append(str(xfeat_path))
        
        from modules.xfeat import XFeat
        
        self.xfeat = XFeat(top_k=max_keypoints, detection_threshold=conf_thresh)
        self.xfeat.dev = torch.device(self.device)
        self.xfeat.net.to(self.device)

    def extract(self, image: np.ndarray) -> dict:
        """Extract XFeat features from an image.

        Converts uint8 input to float32 in [0, 1], and runs the model.
        """
        h, w = image.shape[:2]
        scale_factor = 1.0
        if self.target_resolution is not None and max(h, w) > self.target_resolution:
            scale_factor = self.target_resolution / max(h, w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            resized_img = cv2.resize(image, (new_w, new_h))
            img = resized_img.astype(np.float32) / 255.0
        else:
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
            # XFeat returns a list of dicts, take the first one (batch size = 1)
            pred = self.xfeat.detectAndCompute(tensor)[0]

        # Remove batch dimension
        keypoints = pred["keypoints"].cpu().numpy()        # (N, 2)
        descriptors = pred["descriptors"].cpu().numpy()    # (N, 64)
        scores = pred["scores"].cpu().numpy()              # (N,)

        # Rescale keypoints back to original resolution if scaled
        if scale_factor != 1.0:
            keypoints = keypoints / scale_factor

        # Ensure float32
        keypoints = keypoints.astype(np.float32)
        descriptors = descriptors.astype(np.float32)
        scores = scores.astype(np.float32)

        # Re-normalize descriptors (safety check)
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        descriptors = descriptors / norms

        return {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "scores": scores,
        }
