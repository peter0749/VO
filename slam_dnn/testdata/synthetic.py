"""Synthetic visual odometry dataset generator with known ground truth.

Generates synthetic VO datasets with configurable camera trajectories,
3D point clouds, and rendered feature images. Designed for end-to-end
pipeline testing with deterministic, reproducible output.

Three scenarios are supported:
- ``"translation"``: Camera moves along X-axis with identity rotation.
- ``"rotation"``: Camera rotates around Y-axis with zero translation.
- ``"mixed"``: Camera follows a circular trajectory looking at scene center.

All rendered images contain a mix of point features (circles), line features,
and checkerboard patches to provide diverse, SuperPoint-detectable features.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from slam_dnn.camera import K_from_fov


class SyntheticVODataset:
    """Generate synthetic VO datasets with known ground truth.

    Creates a 3D scene with random points and renders grayscale images
    from a moving camera with known intrinsics and extrinsics. The output
    is suitable for testing visual odometry pipelines end-to-end.

    Args:
        scenario: Camera motion type. One of ``"translation"``,
            ``"rotation"``, or ``"mixed"``.
        n_frames: Total number of frames in the dataset.
        n_points: Number of random 3D points in the scene.
        image_size: ``(W, H)`` tuple for rendered image dimensions.
        fov_deg: Horizontal field of view in degrees for the synthetic camera.
        noise_px: Gaussian noise standard deviation (in pixels) applied
            to projected point positions for realism.
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If ``scenario`` is not one of the three supported values.

    Example:
        >>> ds = SyntheticVODataset(scenario="mixed", n_frames=10)
        >>> data = ds.generate()
        >>> len(data["images"])
        10
        >>> data["gt_poses"][0].shape
        (4, 4)
    """

    VALID_SCENARIOS = ("translation", "rotation", "mixed")

    def __init__(
        self,
        scenario: str = "mixed",
        n_frames: int = 50,
        n_points: int = 300,
        image_size: tuple = (640, 480),
        fov_deg: float = 63.0,
        noise_px: float = 0.5,
        seed: int = 42,
    ):
        if scenario not in self.VALID_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Must be one of {self.VALID_SCENARIOS}"
            )
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2, got {n_frames}")
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1, got {n_points}")

        self.scenario = scenario
        self.n_frames = n_frames
        self.n_points = n_points
        self.image_size = image_size  # (W, H)
        self.fov_deg = fov_deg
        self.noise_px = noise_px
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Intrinsic matrix
    # ------------------------------------------------------------------

    def _build_K(self) -> np.ndarray:
        """Build 3x3 intrinsic matrix from FOV and image size."""
        w, h = self.image_size
        return K_from_fov(w, h, self.fov_deg)

    # ------------------------------------------------------------------
    # 3D Point generation
    # ------------------------------------------------------------------

    def _generate_points_3d(self) -> np.ndarray:
        """Generate random 3D points placed for visibility in the scenario.

        Returns:
            ``(n_points, 3)`` float64 array of 3D point positions in the
            world coordinate frame.
        """
        rng = self._rng
        n = self.n_points

        if self.scenario == "translation":
            # Camera moves along X from 0 to total_motion.
            # Points spread across the full motion range at depth.
            total_motion = self._translation_total()
            x = rng.uniform(-2.0, total_motion + 4.0, n)
            y = rng.uniform(-4.0, 4.0, n)
            z = rng.uniform(5.0, 15.0, n)

        elif self.scenario == "rotation":
            # Camera stays at origin and rotates. Points in front at z > 0.
            x = rng.uniform(-8.0, 8.0, n)
            y = rng.uniform(-5.0, 5.0, n)
            z = rng.uniform(4.0, 15.0, n)

        else:  # mixed — circular trajectory
            # Points distributed in a shell around origin, outside camera path
            # Camera path: radius=3 circle in XZ plane
            # Points at radius 5–12, uniformly in sphere (excluding inner region)
            theta = rng.uniform(0, 2 * np.pi, n)
            phi = rng.uniform(0, np.pi, n)
            r = rng.uniform(5.0, 12.0, n)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.cos(phi)
            z = r * np.sin(phi) * np.sin(theta)

        return np.stack([x, y, z], axis=1).astype(np.float64)

    def _translation_total(self) -> float:
        """Total translation distance for the translation scenario."""
        return min(5.0, 0.4 * (self.n_frames - 1))

    # ------------------------------------------------------------------
    # Pose generation
    # ------------------------------------------------------------------

    def _generate_poses(self) -> list[np.ndarray]:
        """Generate 4x4 world-to-camera pose matrices for all frames.

        Returns:
            List of ``n_frames`` 4x4 float64 matrices representing the
            world-to-camera transform ``[R | t]`` for each frame.
        """
        if self.scenario == "translation":
            return self._poses_translation()
        elif self.scenario == "rotation":
            return self._poses_rotation()
        else:
            return self._poses_mixed()

    def _poses_translation(self) -> list[np.ndarray]:
        """Pure X-axis translation with identity rotation.

        Camera center in world frame moves along +X.
        World-to-cam: R=I, t=[-cam_x, 0, 0].
        """
        total = self._translation_total()
        step = total / max(self.n_frames - 1, 1)
        poses = []
        for i in range(self.n_frames):
            cam_x = i * step
            T = np.eye(4, dtype=np.float64)
            T[0, 3] = -cam_x  # world-to-cam translation
            poses.append(T)
        return poses

    def _poses_rotation(self) -> list[np.ndarray]:
        """Pure Y-axis rotation with zero translation.

        Camera stays at world origin and rotates around Y.
        """
        total_angle_deg = min(60.0, 5.0 * (self.n_frames - 1))
        angle_step_deg = total_angle_deg / max(self.n_frames - 1, 1)
        poses = []
        for i in range(self.n_frames):
            angle = np.deg2rad(i * angle_step_deg)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            # Rotation around Y-axis (world-to-cam)
            R = np.array([
                [cos_a, 0.0, sin_a],
                [0.0, 1.0, 0.0],
                [-sin_a, 0.0, cos_a],
            ], dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            # t = [0, 0, 0] already
            poses.append(T)
        return poses

    def _poses_mixed(self) -> list[np.ndarray]:
        """Circular trajectory looking at scene center.

        Camera orbits the origin at a fixed radius in the XZ plane,
        always looking at the origin (0, 0, 0).
        """
        radius = 3.0
        target = np.array([0.0, 0.0, 0.0])
        n = self.n_frames
        # Orbit angle range: up to ~180° for good spread
        max_angle = np.pi  # half orbit
        poses = []
        for i in range(n):
            if n == 1:
                theta = 0.0
            else:
                theta = max_angle * i / (n - 1)
            cam_pos = radius * np.array([np.sin(theta), 0.0, np.cos(theta)])
            R, t = self._look_at(cam_pos, target)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(T)
        return poses

    @staticmethod
    def _look_at(
        cam_pos: np.ndarray,
        target: np.ndarray,
        world_up: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute world-to-camera (R, t) for a camera looking at a target.

        Camera frame convention: X right, Y down, Z forward (OpenCV).

        Args:
            cam_pos: (3,) camera position in world frame.
            target: (3,) point the camera is looking at.
            world_up: (3,) world up direction. Default: (0, -1, 0).

        Returns:
            R: (3, 3) rotation matrix (world-to-cam).
            t: (3,) translation vector (world-to-cam).
        """
        if world_up is None:
            world_up = np.array([0.0, -1.0, 0.0])

        forward = target - cam_pos
        fn = np.linalg.norm(forward)
        if fn < 1e-10:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        forward = forward / fn

        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            # Degenerate — forward parallel to world_up, pick alternate up
            world_up = np.array([1.0, 0.0, 0.0])
            right = np.cross(forward, world_up)
            rn = np.linalg.norm(right)
        right = right / rn
        down = np.cross(forward, right)

        R = np.vstack([right, down, forward]).astype(np.float64)
        t = -R @ cam_pos
        return R, t

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(
        self,
        T_w2c: np.ndarray,
        points_3d: np.ndarray,
        K: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        """Render a single grayscale image with projected 3D features.

        Args:
            T_w2c: (4, 4) world-to-camera transform.
            points_3d: (N, 3) 3D points in world coordinates.
            K: (3, 3) intrinsic matrix.
            frame_idx: Frame index (used for per-frame RNG seeding).

        Returns:
            (H, W) uint8 grayscale image with rendered features.
        """
        w, h = self.image_size
        R = T_w2c[:3, :3]
        t = T_w2c[:3, 3]

        # Per-frame RNG for reproducible noise/occlusion
        frame_rng = np.random.RandomState(self.seed + frame_idx * 1000)

        # Project all points to camera frame
        pts_cam = (R @ points_3d.T).T + t  # (N, 3)
        z = pts_cam[:, 2]

        # Filter: in front of camera
        in_front = z > 0.5
        n_visible = int(in_front.sum())

        # Random occlusion: drop 20% of visible points
        if n_visible > 0:
            occlusion_mask = frame_rng.random(n_visible) > 0.2
            visible_indices = np.where(in_front)[0]
            keep = visible_indices[occlusion_mask]
        else:
            keep = np.array([], dtype=int)

        # Project to pixel coordinates
        if len(keep) > 0:
            pts_visible = pts_cam[keep]
            z_visible = z[keep]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u = fx * pts_visible[:, 0] / z_visible + cx
            v = fy * pts_visible[:, 1] / z_visible + cy

            # Add Gaussian noise
            u += frame_rng.normal(0, self.noise_px, len(u))
            v += frame_rng.normal(0, self.noise_px, len(v))

            # Filter to image bounds (with margin)
            margin = 5
            in_bounds = (
                (u >= margin) & (u < w - margin)
                & (v >= margin) & (v < h - margin)
            )
            u = u[in_bounds]
            v = v[in_bounds]
            z_visible = z_visible[in_bounds]
        else:
            u = np.array([])
            v = np.array([])
            z_visible = np.array([])

        # --- Render image ---
        img = self._draw_features(u, v, z_visible, w, h, frame_rng)
        return img

    def _draw_features(
        self,
        u: np.ndarray,
        v: np.ndarray,
        depths: np.ndarray,
        w: int,
        h: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Draw mixed feature types on a dark background.

        Features include:
        - Point features (white circles with varying size by depth)
        - Line features (connecting nearby projected points)
        - Checkerboard patches (small textured squares at some points)

        Args:
            u: (N,) pixel x-coordinates.
            v: (N,) pixel y-coordinates.
            depths: (N,) depth values for size scaling.
            w: Image width.
            h: Image height.
            rng: Per-frame random state.

        Returns:
            (H, W) uint8 grayscale image.
        """
        # Dark gray background with slight texture
        img = np.full((h, w), 30, dtype=np.uint8)

        # Add subtle background noise for texture
        bg_noise = rng.randint(0, 15, (h, w), dtype=np.uint8)
        img = np.clip(img.astype(np.int16) + bg_noise.astype(np.int16),
                      0, 255).astype(np.uint8)

        n_pts = len(u)
        if n_pts == 0:
            return img

        # Assign feature types: 0=point, 1=line, 2=checkerboard
        feature_types = rng.randint(0, 3, n_pts)

        # Sort by depth (draw far points first, near on top)
        order = np.argsort(-depths)

        for idx in order:
            px = int(round(u[idx]))
            py = int(round(v[idx]))
            depth = depths[idx]
            ftype = feature_types[idx]

            # Size scales inversely with depth
            radius = max(2, int(8.0 / max(depth, 1.0)))
            intensity = rng.randint(160, 255)

            if ftype == 0:
                # Point feature: filled circle with outline
                cv2.circle(img, (px, py), radius, int(intensity), -1)
                cv2.circle(img, (px, py), radius + 1,
                           max(int(intensity) - 60, 0), 1)

            elif ftype == 1:
                # Line feature: cross pattern + circle center
                r = radius + 1
                cv2.line(img, (px - r, py), (px + r, py),
                         int(intensity), 1)
                cv2.line(img, (px, py - r), (px, py + r),
                         int(intensity), 1)
                cv2.circle(img, (px, py), max(1, radius // 2),
                           int(intensity), -1)

            else:
                # Checkerboard patch: small textured square
                patch_size = max(6, radius * 3)
                half = patch_size // 2
                x0 = max(0, px - half)
                y0 = max(0, py - half)
                x1 = min(w, px + half)
                y1 = min(h, py + half)
                pw, ph = x1 - x0, y1 - y0
                if pw < 3 or ph < 3:
                    continue
                # Create checkerboard pattern
                patch = np.zeros((ph, pw), dtype=np.uint8)
                cell = max(2, patch_size // 4)
                for ci in range(0, ph, cell):
                    for cj in range(0, pw, cell):
                        if ((ci // cell) + (cj // cell)) % 2 == 0:
                            patch[ci:ci + cell, cj:cj + cell] = int(intensity)
                        else:
                            patch[ci:ci + cell, cj:cj + cell] = max(
                                int(intensity) - 100, 20)
                img[y0:y1, x0:x1] = patch

        # Draw lines connecting some adjacent points (for line features)
        if n_pts > 2:
            # Connect each point to its 2 nearest neighbors
            pts_2d = np.stack([u, v], axis=1)
            n_lines = min(n_pts * 2, 200)
            sampled = rng.choice(n_pts, min(n_pts, 50), replace=False)
            for idx in sampled:
                distances = np.linalg.norm(pts_2d - pts_2d[idx], axis=1)
                nearest = np.argsort(distances)[:3]  # skip self (index 0)
                for j in nearest[1:3]:
                    if distances[j] < 40:  # only connect nearby points
                        p1 = (int(round(u[idx])), int(round(v[idx])))
                        p2 = (int(round(u[j])), int(round(v[j])))
                        cv2.line(img, p1, p2, rng.randint(80, 160), 1)

        return img

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> dict:
        """Generate a complete synthetic VO dataset.

        Returns:
            Dictionary with keys:
            - ``"images"``: list of ``(H, W)`` uint8 grayscale ndarrays.
            - ``"gt_poses"``: list of ``n_frames`` 4x4 float64 world-to-camera
              transformation matrices.
            - ``"K"``: (3, 3) float64 intrinsic matrix.
            - ``"points_3d"``: ``(n_points, 3)`` float64 3D point positions
              in world coordinates.

        Example:
            >>> ds = SyntheticVODataset(scenario="translation", n_frames=5)
            >>> data = ds.generate()
            >>> data["images"][0].shape
            (480, 640)
            >>> data["K"].shape
            (3, 3)
        """
        K = self._build_K()
        points_3d = self._generate_points_3d()
        gt_poses = self._generate_poses()

        images = []
        for i, T in enumerate(gt_poses):
            img = self._render_frame(T, points_3d, K, frame_idx=i)
            images.append(img)

        return {
            "images": images,
            "gt_poses": gt_poses,
            "K": K,
            "points_3d": points_3d,
        }

    def save(self, output_dir: str, format: str = "kitti") -> None:
        """Save the generated dataset to disk in KITTI-compatible format.

        Creates the following directory structure::

            output_dir/
            image_0/
                000000.png
                000001.png
                ...
                poses.txt        # KITTI format: 12 floats per line
                calib.txt        # Camera intrinsics (P0 matrix)

        The image directory is loadable by :class:`slam_dnn.io.FrameLoader`.
        The poses file is loadable by :func:`slam_dnn.export.load_kitti_format`.

        Args:
            output_dir: Directory path to write outputs. Created if needed.
            format: Output format. Currently only ``"kitti"`` is supported.

        Raises:
            ValueError: If ``format`` is not ``"kitti"``.

        Example:
            >>> ds = SyntheticVODataset(scenario="mixed", n_frames=5)
            >>> ds.save("/tmp/synth_test")
        """
        if format != "kitti":
            raise ValueError(f"Unsupported format '{format}'. Use 'kitti'.")

        data = self.generate()
        out = Path(output_dir)
        img_dir = out / "image_0"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Save images as PNG (grayscale → BGR for cv2.imwrite)
        for i, img in enumerate(data["images"]):
            # Convert grayscale to BGR for cv2.imwrite to produce PNG
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(img_dir / f"{i:06d}.png"), bgr)

        # Save poses in KITTI format (12 floats per line, 3x4 row-major)
        poses_path = out / "poses.txt"
        with open(poses_path, "w") as f:
            for T in data["gt_poses"]:
                T_3x4 = T[:3, :]
                f.write(" ".join(f"{x:.6f}" for x in T_3x4.flatten()) + "\n")

        # Save calibration (KITTI calib.txt format)
        K = data["K"]
        calib_path = out / "calib.txt"
        with open(calib_path, "w") as f:
            # P0/P1/P2/P3: 3x4 projection matrices (P = K @ [R|t])
            # For a monocular setup, P0 = P1 = P2 = P3 = K @ [I|0]
            p_row = np.zeros(12)
            p_row[:9] = K.flatten()
            # Also add R_rect and Tr_velo_to_cam as identity (standard KITTI)
            row_str = " ".join(f"{x:.12e}" for x in p_row)
            f.write(f"P0: {row_str}\n")
            f.write(f"P1: {row_str}\n")
            f.write(f"P2: {row_str}\n")
            f.write(f"P3: {row_str}\n")
            # R_rect: 3x3 identity (rectification rotation)
            r_rect = np.eye(3).flatten()
            r_str = " ".join(f"{x:.12e}" for x in r_rect)
            f.write(f"R0_rect: {r_str}\n")
            # Tr_velo_to_cam: 3x4 identity (velodyne to camera transform)
            tr = np.eye(4)[:3, :].flatten()
            tr_str = " ".join(f"{x:.12e}" for x in tr)
            f.write(f"Tr_velo_to_cam: {tr_str}\n")
