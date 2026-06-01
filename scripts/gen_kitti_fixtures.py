#!/usr/bin/env python3
"""Generate synthetic KITTI-like image sequence for VO testing.

Creates a set of images with a synthetic 3D scene viewed from slightly
different camera positions, simulating a forward-moving vehicle.
"""

import cv2
import numpy as np
from pathlib import Path
import sys


def generate_kitti_subset(output_dir: str, n_frames: int = 50, seed: int = 42):
    """Generate synthetic road-scene-like image sequence."""
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Image dimensions (KITTI-like: 1241x376, but use smaller for speed)
    W, H = 640, 360
    fx = fy = 500.0
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Create a synthetic 3D scene: random textured points in front of camera
    # Spread points across a ground plane and some vertical structures
    n_points = 800

    # Ground plane points (y = -1.5, like road surface)
    n_ground = n_points // 2
    gx = np.random.uniform(-8, 8, n_ground)
    gy = np.full(n_ground, -1.5)
    gz = np.random.uniform(3, 30, n_ground)
    ground_pts = np.stack([gx, gy, gz], axis=1)

    # Building/pole points (vertical structures on sides)
    n_struct = n_points - n_ground
    sx = np.random.choice([-1, 1], n_struct) * np.random.uniform(3, 8, n_struct)
    sy = np.random.uniform(-2, 3, n_struct)
    sz = np.random.uniform(5, 25, n_struct)
    struct_pts = np.stack([sx, sy, sz], axis=1)

    all_pts_3d = np.vstack([ground_pts, struct_pts]).astype(np.float64)

    # Assign random colors to 3D points
    colors = np.random.randint(50, 255, (len(all_pts_3d), 3)).astype(np.uint8)

    # Camera trajectory: gentle forward motion with slight curve
    # Each frame: small forward translation + tiny rotation
    for i in range(n_frames):
        # Camera position: moving forward (z increases) with slight lateral drift
        t_cam_x = 0.05 * i + 0.002 * np.sin(0.1 * i)
        t_cam_y = 0.0  # constant height
        t_cam_z = 0.15 * i  # forward motion

        # Small rotation around y-axis (yaw)
        yaw = 0.003 * np.sin(0.05 * i)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R_cam = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y],
        ])

        t_cam = np.array([t_cam_x, t_cam_y, t_cam_z])

        # Transform 3D points to camera frame
        # p_cam = R_cam @ (p_world - t_cam) = R_cam @ p_world - R_cam @ t_cam
        pts_cam = (all_pts_3d - t_cam) @ R_cam.T  # world to camera

        # Keep only points in front of camera (z > 0.5)
        mask = pts_cam[:, 2] > 0.5
        pts_visible = pts_cam[mask]
        colors_visible = colors[mask]

        if len(pts_visible) < 50:
            # Fallback: render blank image with some noise
            img = np.random.randint(100, 200, (H, W), dtype=np.uint8)
        else:
            # Project to image
            pts_2d_hom = pts_visible @ K.T  # (N, 3) homogeneous
            pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]

            # Render as colored dots on a gradient background
            # Sky gradient (top half lighter, bottom half darker)
            img = np.zeros((H, W, 3), dtype=np.uint8)
            for row in range(H):
                val = int(180 - 80 * row / H)
                img[row, :] = [val, val, val]

            # Add some texture to the background (road noise)
            noise = np.random.randint(-10, 10, (H, W, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Draw projected points with slight size based on depth
            for j in range(len(pts_2d)):
                x, y = int(pts_2d[j, 0]), int(pts_2d[j, 1])
                if 0 <= x < W and 0 <= y < H:
                    depth = pts_visible[j, 2]
                    radius = max(1, int(4.0 / depth))
                    color = colors_visible[j].tolist()
                    cv2.circle(img, (x, y), radius, color, -1)

            # Convert to grayscale for consistency
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Add some edge structures (lane markings, building edges)
            # Horizontal lines at various y positions
            for y_line in [int(H * 0.6), int(H * 0.65), int(H * 0.7)]:
                offset = int(t_cam_z * 2) % 10
                x_start = offset
                cv2.line(img, (x_start, y_line), (W, y_line), 220, 1)

        # Save as PNG
        filepath = out / f"{i:06d}.png"
        cv2.imwrite(str(filepath), img)

    print(f"Generated {n_frames} synthetic images in {output_dir}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/kitti_05_subset"
    generate_kitti_subset(output)
