"""Synthetic scene generation for visual odometry validation.

Generates 3D scenes with known ground truth poses and renders synthetic
images for end-to-end pipeline testing.  All rendered textures come from
3D planes so that every feature has correct parallax under camera motion.
"""

import cv2
import numpy as np


class SyntheticScene:
    """Random 3D scene with textured features for synthetic VO testing.

    The scene consists of:
    - 5 textured 3D walls forming a room (back, left, right, floor, ceiling)
    - Random 3D points scattered through the scene volume

    All textures are rendered via homography warping so that every feature
    in the synthetic image has geometrically correct parallax.
    """

    def __init__(self, n_points: int = 500, scene_size: float = 5.0, seed: int = 42):
        """
        Args:
            n_points: Number of 3D points in the scene.
            scene_size: Half-extent of the XY bounding box.
            seed: Random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
        self.n_points = n_points
        self.scene_size = scene_size

        # --- 3D feature points ---
        self.points_3d = np.empty((n_points, 3), dtype=np.float64)
        self.points_3d[:, 0] = rng.uniform(-scene_size / 2, scene_size / 2, n_points)
        self.points_3d[:, 1] = rng.uniform(-scene_size / 2, scene_size / 2, n_points)
        self.points_3d[:, 2] = rng.uniform(3.0, 8.0, n_points)

        # Per-point marker properties
        self._marker_intensity = rng.randint(60, 230, size=n_points).astype(np.uint8)
        self._marker_type = rng.randint(0, 4, size=n_points)

        # --- Room planes (for textured background with correct parallax) ---
        # Each plane: 4 corners in world coords (3D) + texture image
        # Room: X ∈ [-5, 5], Y ∈ [-3, 3], Z ∈ [0, 10]
        self._planes = self._make_room_planes(rng)
        self._tex_size = 256  # texture image resolution per plane

    # ------------------------------------------------------------------
    # Room plane setup
    # ------------------------------------------------------------------

    @staticmethod
    def _make_room_planes(rng) -> list[dict]:
        """Create 5 textured planes forming a room.

        Returns list of dicts with 'corners' (4,3) and 'texture' (H,W) arrays.
        """
        tex_size = 256

        def _random_texture(s=256):
            """Generate a rich, feature-rich random texture."""
            tex = np.zeros((s, s), dtype=np.uint8)
            # Base: random noise
            tex[:, :] = rng.randint(40, 200, (s, s), dtype=np.uint8)
            # Add grid lines for strong features
            for i in range(0, s, 32):
                tex[i, :] = rng.randint(0, 30, dtype=np.uint8)
                tex[:, i] = rng.randint(0, 30, dtype=np.uint8)
            # Add random circles/blobs
            for _ in range(30):
                cx, cy = rng.randint(10, s - 10, 2)
                r = rng.randint(3, 15)
                val = rng.randint(20, 240, dtype=np.uint8)
                cv2.circle(tex, (int(cx), int(cy)), int(r), int(val), -1)
            # Add random rectangles
            for _ in range(20):
                x1, y1 = rng.randint(0, s - 30, 2)
                w, h = rng.randint(5, 30, 2)
                val = rng.randint(20, 240, dtype=np.uint8)
                cv2.rectangle(tex, (int(x1), int(y1)),
                              (int(x1 + w), int(y1 + h)), int(val), -1)
            # Blur slightly for realism
            tex = cv2.GaussianBlur(tex, (3, 3), 0.8)
            return tex

        planes = []
        room_x, room_y, room_z = 5.0, 3.0, 10.0

        # Back wall (Z = room_z, facing -Z)
        planes.append({
            "corners": np.array([
                [-room_x, -room_y, room_z],
                [room_x, -room_y, room_z],
                [room_x, room_y, room_z],
                [-room_x, room_y, room_z],
            ]),
            "texture": _random_texture(tex_size),
        })
        # Left wall (X = -room_x)
        planes.append({
            "corners": np.array([
                [-room_x, -room_y, 0.0],
                [-room_x, -room_y, room_z],
                [-room_x, room_y, room_z],
                [-room_x, room_y, 0.0],
            ]),
            "texture": _random_texture(tex_size),
        })
        # Right wall (X = room_x)
        planes.append({
            "corners": np.array([
                [room_x, -room_y, room_z],
                [room_x, -room_y, 0.0],
                [room_x, room_y, 0.0],
                [room_x, room_y, room_z],
            ]),
            "texture": _random_texture(tex_size),
        })
        # Floor (Y = room_y)
        planes.append({
            "corners": np.array([
                [-room_x, room_y, room_z],
                [room_x, room_y, room_z],
                [room_x, room_y, 0.0],
                [-room_x, room_y, 0.0],
            ]),
            "texture": _random_texture(tex_size),
        })
        # Ceiling (Y = -room_y)
        planes.append({
            "corners": np.array([
                [-room_x, -room_y, 0.0],
                [room_x, -room_y, 0.0],
                [room_x, -room_y, room_z],
                [-room_x, -room_y, room_z],
            ]),
            "texture": _random_texture(tex_size),
        })

        return planes

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project_points(
        self, K: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D points to camera frame.

        Returns:
            pixels: (N, 2) projected pixel coordinates.
            in_front: (N,) boolean mask.
            depths: (N,) depth values.
        """
        pts_cam = (R @ self.points_3d.T).T + t
        z = pts_cam[:, 2]
        in_front = z > 0.1
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = fx * pts_cam[:, 0] / np.where(z > 1e-8, z, 1.0) + cx
        v = fy * pts_cam[:, 1] / np.where(z > 1e-8, z, 1.0) + cy
        return np.stack([u, v], axis=1), in_front, z

    def _project_3d(
        self, pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Project arbitrary 3D points, returning (N, 2) pixel coords and (N,) z."""
        pts_cam = (R @ pts.T).T + t
        z = pts_cam[:, 2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        safe_z = np.where(z > 1e-8, z, 1.0)
        u = fx * pts_cam[:, 0] / safe_z + cx
        v = fy * pts_cam[:, 1] / safe_z + cy
        return np.stack([u, v], axis=1), z

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_from_pose(
        self,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        """Render a synthetic image from a given camera pose.

        Renders textured 3D planes via homography warping and overlays
        projected 3D feature markers, producing an image with dense
        features that all have correct parallax.

        Args:
            K: (3, 3) camera intrinsic matrix.
            R: (3, 3) rotation matrix (world-to-camera).
            t: (3,) translation vector (world-to-camera).
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            (height, width) uint8 grayscale image.
        """
        img = np.full((height, width), 128, dtype=np.uint8)

        # --- Render textured planes ---
        ts = self._tex_size
        # Texture corner coords: (0,0), (ts,0), (ts,ts), (0,ts)
        tex_corners = np.array(
            [[0, 0], [ts, 0], [ts, ts], [0, ts]], dtype=np.float32
        )

        for plane in self._planes:
            corners_3d = plane["corners"]
            # Check all corners are in front of camera
            pts_cam = (R @ corners_3d.T).T + t
            z_corners = pts_cam[:, 2]
            if np.any(z_corners < 0.2):
                continue  # plane partially behind camera, skip

            # Project corners to pixel coordinates
            proj, _ = self._project_3d(corners_3d, K, R, t)
            dst_corners = proj.astype(np.float32)

            # Compute homography from texture to image
            H = cv2.getPerspectiveTransform(tex_corners, dst_corners)
            if H is None:
                continue

            # Warp texture into image
            warped = cv2.warpPerspective(
                plane["texture"], H, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            # Create mask for the warped texture
            tex_mask = np.ones((ts, ts), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(
                tex_mask, H, (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            # Composite: blend warped texture into image
            blend = warped_mask.astype(np.float32) / 255.0
            img = (img * (1 - blend) + warped.astype(np.float32) * blend).astype(
                np.uint8
            )

        # --- Overlay 3D feature markers ---
        pixels, in_front, depths = self.project_points(K, R, t)
        margin = 15

        for i in range(self.n_points):
            if not in_front[i]:
                continue
            u, v = int(round(pixels[i, 0])), int(round(pixels[i, 1]))
            if u < margin or u >= width - margin or v < margin or v >= height - margin:
                continue

            depth = depths[i]
            radius = max(2, int(10.0 / depth * 4.0))
            intensity = int(self._marker_intensity[i])
            mtype = self._marker_type[i]

            if mtype == 0:
                cv2.circle(img, (u, v), radius, intensity, -1)
                cv2.circle(img, (u, v), radius + 1, max(intensity - 80, 0), 1)
            elif mtype == 1:
                half = radius
                cv2.rectangle(
                    img, (u - half, v - half), (u + half, v + half), intensity, -1
                )
            elif mtype == 2:
                cv2.line(img, (u - radius, v), (u + radius, v), intensity, 2)
                cv2.line(img, (u, v - radius), (u, v + radius), intensity, 2)
                cv2.circle(img, (u, v), max(1, radius // 2), intensity, -1)
            else:
                # Diamond shape
                pts_d = np.array(
                    [[u, v - radius], [u + radius, v],
                     [u, v + radius], [u - radius, v]],
                    dtype=np.int32,
                )
                cv2.fillPoly(img, [pts_d], int(intensity))

        # --- Deterministic Gaussian noise (seeded per-pose for stability) ---
        noise_rng = np.random.RandomState(
            abs(hash((R.tobytes(), t.tobytes()))) % (2**31)
        )
        noise = noise_rng.normal(0, 3, (height, width)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def generate_trajectory(
        self, n_frames: int = 20, motion_type: str = "circular"
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate a smooth camera trajectory with known ground-truth poses.

        Args:
            n_frames: Number of frames.
            motion_type: One of "circular", "linear", "pure_translation_x",
                "pure_rotation_yaw", "figure_eight".

        Returns:
            List of (R, t) tuples — world-to-camera transforms.
        """
        generators = {
            "circular": self._gen_circular,
            "linear": self._gen_linear_forward,
            "pure_translation_x": self._gen_pure_translation_x,
            "pure_rotation_yaw": self._gen_pure_rotation_yaw,
            "figure_eight": self._gen_figure_eight,
        }
        if motion_type not in generators:
            raise ValueError(
                f"Unknown motion_type '{motion_type}'. "
                f"Options: {list(generators.keys())}"
            )
        return generators[motion_type](n_frames)

    @staticmethod
    def _look_at_Rt(
        cam_pos: np.ndarray, target: np.ndarray, world_up: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute world-to-camera (R, t) for a camera looking at a target.

        Camera frame: X right, Y down, Z forward (OpenCV).
        """
        if world_up is None:
            world_up = np.array([0.0, -1.0, 0.0])

        forward = target - cam_pos
        fn = np.linalg.norm(forward)
        if fn < 1e-10:
            return np.eye(3), np.zeros(3)
        forward = forward / fn

        right = np.cross(world_up, forward)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            world_up = np.array([1.0, 0.0, 0.0])
            right = np.cross(world_up, forward)
            rn = np.linalg.norm(right)
        right = right / rn
        down = np.cross(forward, right)

        R = np.vstack([right, down, forward])
        t = -R @ cam_pos
        return R, t

    def _gen_circular(self, n_frames: int, radius: float = 1.0) -> list:
        """Circular orbit around scene centre in the XZ plane."""
        target = np.array([0.0, 0.0, 5.0])
        poses = []
        for i in range(n_frames):
            theta = 2.0 * np.pi * i / n_frames
            cam_pos = target - radius * np.array(
                [np.sin(theta), 0.0, np.cos(theta)]
            )
            R, t = self._look_at_Rt(cam_pos, target)
            poses.append((R, t))
        return poses

    def _gen_linear_forward(self, n_frames: int, step: float = 0.3) -> list:
        """Linear forward motion along Z, identity rotation."""
        poses = []
        for i in range(n_frames):
            R = np.eye(3)
            t = np.array([0.0, 0.0, -i * step])
            poses.append((R, t))
        return poses

    def _gen_pure_translation_x(self, n_frames: int, step: float = 0.3) -> list:
        """Pure translation along X axis, no rotation."""
        poses = []
        for i in range(n_frames):
            R = np.eye(3)
            t = np.array([-i * step, 0.0, 0.0])
            poses.append((R, t))
        return poses

    def _gen_pure_rotation_yaw(
        self, n_frames: int, angle_step_deg: float = 10.0
    ) -> list:
        """Pure yaw rotation in place (camera at origin).

        Known limitation: essential matrix cannot recover translation
        from pure rotation — this test documents the failure mode.
        """
        poses = []
        for i in range(n_frames):
            angle = np.deg2rad(i * angle_step_deg)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array(
                [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]]
            )
            t = np.zeros(3)
            poses.append((R, t))
        return poses

    def _gen_figure_eight(self, n_frames: int, radius: float = 1.0) -> list:
        """Figure-eight (lemniscate) trajectory."""
        target = np.array([0.0, 0.0, 5.0])
        poses = []
        for i in range(n_frames):
            theta = 2.0 * np.pi * i / n_frames
            offset = radius * np.array(
                [np.sin(theta), 0.0, np.sin(theta) * np.cos(theta)]
            )
            cam_pos = target - np.array([0.0, 0.0, radius]) + offset
            R, t = self._look_at_Rt(cam_pos, target)
            poses.append((R, t))
        return poses


# ======================================================================
# Error metrics and alignment utilities
# ======================================================================


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Rotation error in degrees between two rotation matrices."""
    cos_angle = (np.trace(R_est @ R_gt.T) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def translation_direction_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Translation direction error in degrees (scale-invariant)."""
    ne = np.linalg.norm(t_est)
    ng = np.linalg.norm(t_gt)
    if ne < 1e-8 or ng < 1e-8:
        return 180.0
    t_e = t_est / ne
    t_g = t_gt / ng
    dot = np.clip(np.dot(t_e, t_g), -1.0, 1.0)
    return float(np.degrees(np.arccos(np.abs(dot))))


def umeyama_alignment(
    X: np.ndarray, Y: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama alignment: find (s, R, t) minimising ||Y - (s*R@X + t)||^2.

    Args:
        X: (N, 3) source points.
        Y: (N, 3) target points.

    Returns:
        s: Uniform scale factor.
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
        Aligned points: ``Y_aligned = s * X @ R.T + t``
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    n = X.shape[0]
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    H = Xc.T @ Yc / n
    U, S, Vt = np.linalg.svd(H)

    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1.0

    R = Vt.T @ D @ U.T

    var_x = np.sum(Xc ** 2) / n
    if var_x < 1e-12:
        s = 1.0
    else:
        s = float(np.sum(S * np.diag(D)) / var_x)

    t = mu_y - s * R @ mu_x
    return s, R, t


def align_trajectory(
    est_pos: np.ndarray, gt_pos: np.ndarray
) -> np.ndarray:
    """Align estimated positions to ground truth via Umeyama.

    Args:
        est_pos: (N, 3) estimated positions.
        gt_pos: (N, 3) ground truth positions.

    Returns:
        (N, 3) aligned estimated positions.
    """
    s, R, t = umeyama_alignment(est_pos, gt_pos)
    return s * est_pos @ R.T + t
