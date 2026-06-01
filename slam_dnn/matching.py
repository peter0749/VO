"""Feature matching (LightGlue + Classic)."""
import cv2
import numpy as np
import torch
from abc import ABC, abstractmethod
from lightglue import LightGlue


class MatcherBase(ABC):
    """Abstract base class for feature matchers.

    All matchers must implement this interface so they can be swapped
    in/out without code changes. This enables the strategy pattern:
    VisualOdometry doesn't care which matcher it uses.
    """

    @abstractmethod
    def match(self, feats0: dict, feats1: dict, image_size: tuple | None = None) -> dict:
        """Match features between two frames.

        Args:
            feats0: Dict with keys "keypoints" (N, 2) float32,
                   "descriptors" (N, 256) float32, "scores" (N,) float32.
            feats1: Same format.
            image_size: Optional (H, W) for better matching (LightGlue only).

        Returns:
            Dict with:
                - "points0": (K, 2) float32 — matched points in image 0
                - "points1": (K, 2) float32 — matched points in image 1
                - "scores": (K,) float32 — match confidence (higher = better)
                - "indices": (K, 2) int32 — index pairs in original features
        """
        pass


def create_matcher(method: str = "lightglue", **kwargs) -> MatcherBase:
    """Factory: create matcher by name.

    Args:
        method: "lightglue" or "classic"
        **kwargs: forwarded to matcher constructor

    Returns:
        MatcherBase instance
    """
    if method == "lightglue":
        return LightGlueMatcher(**kwargs)
    elif method == "classic":
        kwargs.pop("device", None)
        return ClassicMatcher(**kwargs)
    else:
        raise ValueError(
            f"Unknown matcher method: {method!r}. Use 'lightglue' or 'classic'."
        )


class LightGlueMatcher(MatcherBase):
    """Wraps LightGlue's neural matcher for SuperPoint features."""

    def __init__(self, filter_threshold: float = 0.1, device: str = "auto"):
        """
        Initialize LightGlue matcher.

        Args:
            filter_threshold: confidence threshold for matches (default 0.1)
            device: "auto" | "cuda" | "cpu"
        """
        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize matcher with SuperPoint features
        self.matcher = LightGlue(
            features='superpoint', filter_threshold=filter_threshold
        )
        self.matcher.eval()
        self.matcher.to(self.device)

    def match(
        self, feats0: dict, feats1: dict, image_size: tuple = None
    ) -> dict:
        """Match features between two images using LightGlue.

        Args:
            feats0: Dict with keys "keypoints" (N,2), "descriptors" (N,256),
                    "scores" (N,) — as returned by SuperPointExtractor.
            feats1: Same format as feats0 with M keypoints.
            image_size: Optional (H, W) tuple for better matching quality.

        Returns:
            Dict with keys:
                - points0: (K,2) matched points in image 0 (float32)
                - points1: (K,2) matched points in image 1 (float32)
                - scores: (K,) match confidence scores (float32)
                - indices: (K,2) [idx_in_feats0, idx_in_feats1] (int32)
        """
        # 1. Convert keypoints to torch tensors, add batch dim: (N,2) → (1,N,2)
        kpts0 = (
            torch.from_numpy(feats0["keypoints"]).float().unsqueeze(0).to(self.device)
        )
        kpts1 = (
            torch.from_numpy(feats1["keypoints"]).float().unsqueeze(0).to(self.device)
        )

        # 2. Add batch dim to descriptors: (N,256) → (1,N,256)
        desc0 = (
            torch.from_numpy(feats0["descriptors"]).float().unsqueeze(0).to(self.device)
        )
        desc1 = (
            torch.from_numpy(feats1["descriptors"]).float().unsqueeze(0).to(self.device)
        )

        # 3. Build input dicts for LightGlue
        input0 = {"keypoints": kpts0, "descriptors": desc0}
        input1 = {"keypoints": kpts1, "descriptors": desc1}

        if image_size is not None:
            h, w = image_size
            img_size_tensor = torch.tensor([[h, w]]).to(self.device)
            input0["image_size"] = img_size_tensor
            input1["image_size"] = img_size_tensor

        # 4. Run matcher in inference mode
        with torch.no_grad():
            matches_dict = self.matcher(
                {"image0": input0, "image1": input1}
            )

        # 5. Parse output: matches is (1, K, 2) — [idx_in_feats0, idx_in_feats1]
        matches = matches_dict["matches"][0]  # (K, 2)

        # 6. Filter out unmatched keypoints (indicated by -1 index)
        valid = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
        matches = matches[valid]

        # 7. Gather matched keypoints by index
        idx0 = matches[:, 0].cpu().numpy()
        idx1 = matches[:, 1].cpu().numpy()

        points0 = feats0["keypoints"][idx0]  # (K, 2)
        points1 = feats1["keypoints"][idx1]  # (K, 2)

        # 8. Extract confidence scores for matched keypoints
        # matching_scores0: (1, N) — score per keypoint in feats0
        scores0 = matches_dict["matching_scores0"][0].cpu().numpy()
        scores = scores0[idx0]  # (K,)

        # 9. Return as float32 numpy arrays
        return {
            "points0": points0.astype(np.float32),
            "points1": points1.astype(np.float32),
            "scores": scores.astype(np.float32),
            "indices": matches.cpu().numpy().astype(np.int32),
        }


class ClassicMatcher(MatcherBase):
    """Classic BF matcher with Lowe's ratio test."""

    def __init__(
        self,
        ratio: float = 0.75,
        ransac_reproj_threshold: float = 3.0,
        method: str = "bf",
        device: str = "cpu",
    ):
        """
        Args:
            ratio: Lowe's ratio test parameter (default 0.75, stricter = lower)
            ransac_reproj_threshold: RANSAC reprojection threshold
            method: 'bf' (brute-force) or 'flann'
            device: Ignored (kept for MatcherBase interface compatibility);
                    ClassicMatcher is always CPU-only since cv2 matchers are CPU.
        """
        self.ratio = ratio
        self.ransac_reproj_threshold = ransac_reproj_threshold

        if method == "bf":
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif method == "flann":
            FLANN_INDEX_L2 = 1
            index_params = dict(algorithm=FLANN_INDEX_L2, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(
                index_params, search_params
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def match(self, feats0: dict, feats1: dict, image_size: tuple | None = None) -> dict:
        """Match features between two images using classic BF + ratio test.

        Args:
            feats0: Dict with keys "keypoints" (N,2) and "descriptors" (N,D).
            feats1: Dict with keys "keypoints" (M,2) and "descriptors" (M,D).

        Returns:
            Dict with keys:
                - points0: (K,2) matched points in image 0
                - points1: (K,2) matched points in image 1
                - scores: (K,) normalized scores, higher = better
                - indices: (K,2) corresponding indices in original features
        """
        desc0 = feats0["descriptors"].astype(np.float32)
        desc1 = feats1["descriptors"].astype(np.float32)

        # k-NN matching with k=2 for ratio test
        matches = self.matcher.knnMatch(desc0, desc1, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio * n.distance:
                good_matches.append(m)

        # Extract matched indices
        idx0 = [m.queryIdx for m in good_matches]
        idx1 = [m.trainIdx for m in good_matches]

        # Gather keypoints
        points0 = feats0["keypoints"][idx0].astype(np.float32)
        points1 = feats1["keypoints"][idx1].astype(np.float32)

        # Scores: BFMatcher gives distances (lower = better), invert to
        # "higher = better" format to match LightGlue interface.
        distances = np.array([m.distance for m in good_matches])
        # Normalize to [0, 1] where higher = better
        max_dist = distances.max() if distances.size > 0 else 1.0
        if max_dist == 0:
            scores = np.ones_like(distances, dtype=np.float32)
        else:
            scores = 1.0 - (distances / max_dist)
        scores = scores.astype(np.float32)

        indices = np.stack([idx0, idx1], axis=1).astype(np.int32)

        return {
            "points0": points0,
            "points1": points1,
            "scores": scores,
            "indices": indices,
        }
