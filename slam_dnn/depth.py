"""Helper to load pre-computed depth maps."""
import os
import cv2
import numpy as np


class DepthMapLoader:
    """Loads and scales depth maps for depth-prior Visual Odometry."""

    def __init__(self, directory: str, scale_factor: float = 256.0):
        """Initialize depth map loader.

        Args:
            directory: Path to folder containing depth maps.
            scale_factor: Scaling factor to convert depth map pixel values to meters.
        """
        self.directory = directory
        self.scale_factor = scale_factor
        self.depth_files = []
        if os.path.exists(directory) and os.path.isdir(directory):
            valid_exts = ('.png', '.npz', '.npy')
            self.depth_files = sorted([
                os.path.join(directory, f) for f in os.listdir(directory)
                if f.lower().endswith(valid_exts)
            ])

    def get_depth(self, frame_idx: int) -> np.ndarray:
        """Get depth map for the specified frame index.

        Args:
            frame_idx: Index of the frame.

        Returns:
            np.ndarray: 2D array of depth values in meters.
        """
        if not self.depth_files:
            raise FileNotFoundError(f"No depth maps found in directory: {self.directory}")
        if frame_idx < 0 or frame_idx >= len(self.depth_files):
            raise IndexError(f"Depth map index {frame_idx} out of range (total {len(self.depth_files)})")

        file_path = self.depth_files[frame_idx]
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.png':
            # Load 16-bit PNG (IMREAD_UNCHANGED)
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"Failed to load depth image: {file_path}")
            depth = img.astype(np.float32) / self.scale_factor
        elif ext == '.npy':
            depth = np.load(file_path).astype(np.float32)
        elif ext == '.npz':
            with np.load(file_path) as data:
                keys = list(data.keys())
                if 'depth' in keys:
                    depth = data['depth'].astype(np.float32)
                elif keys:
                    depth = data[keys[0]].astype(np.float32)
                else:
                    raise IOError(f"Empty npz file: {file_path}")
        else:
            raise ValueError(f"Unsupported depth file extension: {ext}")

        return depth


class DepthAnythingEstimator:
    """Estimates monocular depth maps using pre-trained Depth Anything (v1/v2) models."""

    def __init__(
        self,
        model_name: str = "LiheYoung/depth-anything-small-hf",
        target_resolution: tuple[int, int] = (320, 192),
        device: str = "auto"
    ):
        """Initialize estimator.

        Args:
            model_name: Hugging Face model repository ID.
            target_resolution: (width, height) at which to run inference.
            device: PyTorch device ('cpu', 'mps', 'cuda', or 'auto').
        """
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.target_resolution = target_resolution
        self.is_metric = "metric" in model_name.lower()

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load processor and model from Hugging Face
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate metric depth from a single image.

        Args:
            image: Input frame as a numpy array.

        Returns:
            np.ndarray: Predicted depth map in meters.
        """
        import torch

        # Ensure image is RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image.copy()

        h_orig, w_orig = image_rgb.shape[:2]
        w_target, h_target = self.target_resolution

        # Resize image for fast inference (downscaling)
        img_resized = cv2.resize(image_rgb, (w_target, h_target), interpolation=cv2.INTER_LINEAR)

        # Run processor
        inputs = self.processor(images=img_resized, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Output is disparity-like for relative models, direct depth for metric models
            pred_val = outputs.predicted_depth.squeeze().cpu().numpy()

        # Scale back to original resolution
        pred_resized = cv2.resize(pred_val, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        if self.is_metric:
            depth = np.maximum(pred_resized, 0.0)
        else:
            # Convert disparity to depth: depth = 1.0 / (disparity + epsilon)
            # Avoid division by zero
            depth = 1.0 / (np.maximum(pred_resized, 0.0) + 1e-6)

        return depth.astype(np.float32)


