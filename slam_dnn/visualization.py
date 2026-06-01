"""Trajectory and VO visualization utilities.

Publication-quality plots for camera trajectories, feature matches,
and VO pipeline diagnostics.
"""
import os

# Headless backend selection for CI / headless environments
if os.environ.get("DISPLAY", "") == "":
    import matplotlib

    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poses_to_positions(poses) -> np.ndarray:
    """Convert poses to world-frame camera centers.

    Accepts:
      - list/array of 4x4 SE3 matrices
      - (N, 3) array of positions (returned as-is)
      - (N, 4, 4) array of SE3 matrices
      - empty array / empty list -> (0, 3) array

    Returns:
        (N, 3) array of camera center positions in world frame.
    """
    if poses is None:
        return np.zeros((0, 3), dtype=np.float64)

    poses_arr = np.asarray(poses, dtype=np.float64)

    # Empty input
    if poses_arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Already positions (N, 3)
    if poses_arr.ndim == 2 and poses_arr.shape[1] == 3:
        return poses_arr

    # (N, 4, 4) SE3 matrices
    if poses_arr.ndim == 3 and poses_arr.shape[1:] == (4, 4):
        positions = []
        for T in poses_arr:
            R = T[:3, :3]
            t = T[:3, 3]
            c_world = -R.T @ t
            positions.append(c_world)
        return np.array(positions, dtype=np.float64)

    # List of 4x4 matrices (heterogeneous input)
    if isinstance(poses, list) and len(poses) > 0:
        first = np.asarray(poses[0])
        if first.shape == (4, 4):
            positions = []
            for T in poses:
                T = np.asarray(T, dtype=np.float64)
                R = T[:3, :3]
                t = T[:3, 3]
                c_world = -R.T @ t
                positions.append(c_world)
            return np.array(positions, dtype=np.float64)
        elif first.shape == (3,):
            return np.array(poses, dtype=np.float64)

    return np.zeros((0, 3), dtype=np.float64)


# ---------------------------------------------------------------------------
# 2D Trajectory Comparison
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(
    estimated: np.ndarray,
    ground_truth: np.ndarray | None = None,
    title: str = "Camera Trajectory",
    save_path: str | None = None,
    show: bool = False,
) -> Figure:
    """Plot estimated versus ground truth trajectory in a 2D top-down view.

    Renders both trajectories on axes with equal aspect ratio for true
    spatial accuracy. Start and end markers are added to the estimated
    trajectory; ground truth is shown in dashed gray.

    Args:
        estimated: Estimated trajectory as (N, 3) positions, (N, 4, 4) SE(3)
            matrices, or a list of 4x4 matrices.
        ground_truth: Optional ground truth in the same format.
        title: Plot title. Defaults to "Camera Trajectory".
        save_path: If provided, save the figure as a PNG to this path.
            Example: save_path="results/trajectory.png".
        show: If True, display the plot interactively via plt.show().

    Returns:
        matplotlib Figure object.
    """
    est_pos = _poses_to_positions(estimated)
    gt_pos = _poses_to_positions(ground_truth) if ground_truth is not None else None

    fig, ax = plt.subplots(figsize=(10, 10))

    # Handle empty estimated trajectory gracefully
    if est_pos.shape[0] == 0:
        ax.set_title(title)
        ax.text(
            0.5, 0.5, "No trajectory data", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="gray",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # Estimated trajectory
    ex, ez = est_pos[:, 0], est_pos[:, 2]
    ax.plot(ex, ez, "b-", linewidth=1.5, label="Estimated")
    ax.plot(ex[0], ez[0], "go", markersize=12, label="Start", zorder=10)
    ax.plot(ex[-1], ez[-1], "r^", markersize=12, label="End", zorder=10)

    # Ground truth trajectory
    if gt_pos is not None and gt_pos.shape[0] > 0:
        gx, gz = gt_pos[:, 0], gt_pos[:, 2]
        ax.plot(gx, gz, color="gray", linestyle="--", linewidth=1.5, label="Ground Truth")

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Z (m)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# 3D Trajectory
# ---------------------------------------------------------------------------

def plot_trajectory_3d(
    poses,
    title: str = "3D Trajectory",
    save_path: str | None = None,
    show: bool = False,
) -> Figure:
    """Plot a trajectory in 3D space using matplotlib's 3D projection.

    The trajectory line is colored by frame number using a viridis gradient,
    making it easy to track temporal progression along the path.

    Args:
        poses: List of 4x4 SE(3) matrices or (N, 3) positions.
        title: Plot title. Defaults to "3D Trajectory".
        save_path: If provided, save the figure as a PNG to this path.
        show: If True, display the plot interactively via plt.show().

    Returns:
        matplotlib Figure with 3D axes.
    """
    positions = _poses_to_positions(poses)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    if positions.shape[0] == 0:
        ax.set_title(title)
        ax.text2D(
            0.5, 0.5, "No trajectory data", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="gray",
        )
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    n = len(x)

    # Color gradient by frame number
    colors = cm.viridis(np.linspace(0, 1, n))

    # Draw line segments with gradient color
    for i in range(n - 1):
        ax.plot(
            x[i:i + 2], y[i:i + 2], z[i:i + 2],
            color=colors[i], linewidth=2,
        )

    # Start / end markers
    ax.scatter(x[0], y[0], z[0], c="green", s=100, marker="o", label="Start", zorder=10)
    ax.scatter(x[-1], y[-1], z[-1], c="red", s=100, marker="^", label="End", zorder=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=13)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Feature Matches
# ---------------------------------------------------------------------------

def plot_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    matches: dict,
    n_show: int = 50,
    title: str = "Feature Matches",
    save_path: str | None = None,
) -> Figure:
    """Visualize feature matches between two images side by side.

    Each image is shown with its keypoints overlaid, and lines connect
    matching pairs across the two images. When match scores or distances
    are available, the lines are colored by confidence.

    Args:
        img0: First image, shape (H, W) or (H, W, 3).
        img1: Second image, same format as img0.
        matches: Dictionary with keys 'keypoints0' and 'keypoints1' (each
            shape (K, 2) in pixel coordinates). Optionally includes
            'scores' or 'distances' for color-coded lines.
        n_show: Maximum number of match lines to draw. Default 50.
        title: Plot title. Defaults to "Feature Matches".
        save_path: If provided, save the figure as a PNG to this path.

    Returns:
        matplotlib Figure object.
    """
    # Convert grayscale to RGB for consistent display
    if img0.ndim == 2:
        img0_disp = np.stack([img0] * 3, axis=-1)
    else:
        img0_disp = img0
    if img1.ndim == 2:
        img1_disp = np.stack([img1] * 3, axis=-1)
    else:
        img1_disp = img1

    kp0 = np.asarray(matches.get("keypoints0", []))
    kp1 = np.asarray(matches.get("keypoints1", []))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(img0_disp)
    axes[0].set_title("Frame 0")
    axes[0].axis("off")
    axes[1].imshow(img1_disp)
    axes[1].set_title("Frame 1")
    axes[1].axis("off")

    if len(kp0) == 0 or len(kp1) == 0:
        fig.suptitle(f"{title} (no matches)", fontsize=13)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # Limit number of matches shown
    n_matches = min(n_show, len(kp0), len(kp1))
    kp0_show = kp0[:n_matches]
    kp1_show = kp1[:n_matches]

    # Score-based coloring if available
    scores = matches.get("scores", matches.get("distances", None))
    if scores is not None:
        scores = np.asarray(scores)[:n_matches]
        # Normalize to [0, 1] for colormap
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores) * 0.5
        line_colors = cm.coolwarm(norm_scores)
    else:
        # Fallback: gradient by index
        colors_t = np.linspace(0.2, 0.9, n_matches)
        line_colors = cm.viridis(colors_t)

    # Draw keypoints and match lines
    # We overlay lines as offsets on the second axes image
    h, w = img0_disp.shape[:2]
    for i in range(n_matches):
        x0, y0 = kp0_show[i]
        x1, y1 = kp1_show[i]
        color = line_colors[i]
        # Keypoint on image 0
        axes[0].plot(x0, y0, ".", color=color, markersize=4)
        # Keypoint on image 1
        axes[1].plot(x1, y1, ".", color=color, markersize=4)
        # Connecting line (we add a small annotation on the right image)
        # Use ConnectionPatch between the two images for cross-image lines
        from matplotlib.patches import ConnectionPatch

        con = ConnectionPatch(
            xyA=(x0, y0), coordsA=axes[0].transData,
            xyB=(x1, y1), coordsB=axes[1].transData,
            color=color, linewidth=0.8, alpha=0.6,
        )
        fig.add_artist(con)

    fig.suptitle(f"{title} ({n_matches}/{len(kp0)} shown)", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Trajectory Video
# ---------------------------------------------------------------------------

def save_trajectory_video(
    poses,
    images: list,
    output_path: str,
    fps: int = 10,
) -> None:
    """Generate an MP4 video showing the camera trajectory overlaid on each frame.

    Each output frame has the camera image on the left and an up-to-date
    top-down trajectory map on the right. The current position is highlighted
    in red and the start position in green.

    Args:
        poses: List of 4x4 SE(3) matrices or (N, 3) positions.
        images: List of camera images, one per pose (same length as poses).
        output_path: Path for the output MP4 file.
        fps: Frames per second in the output video. Default 10.

    Raises:
        RuntimeError: If opencv-python is not available or the VideoWriter
            cannot be opened.
        ValueError: If no frames are provided (poses or images are empty).
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("opencv-python required for video output") from e

    positions = _poses_to_positions(poses)
    n_frames = min(len(positions), len(images))
    if n_frames == 0:
        raise ValueError("No frames to render")

    # Use a temp file approach with cv2.VideoWriter
    frame_h, frame_w = 480, 960  # Combined frame dimensions
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    try:
        for i in range(n_frames):
            combined = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

            # --- Left half: camera image ---
            img = images[i]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_resized = cv2.resize(img, (frame_w // 2, frame_h))
            combined[:, :frame_w // 2] = img_resized

            # Frame counter overlay
            cv2.putText(
                combined, f"Frame {i}/{n_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            # --- Right half: top-down trajectory ---
            traj_panel = np.ones((frame_h, frame_w // 2, 3), dtype=np.uint8) * 240

            # Build 2D plot of trajectory up to current frame
            traj_so_far = positions[:i + 1]
            if len(traj_so_far) >= 2:
                xs = traj_so_far[:, 0]
                zs = traj_so_far[:, 2]

                # Scale to fit panel
                x_range = xs.max() - xs.min() or 1.0
                z_range = zs.max() - zs.min() or 1.0
                scale = min((frame_w // 2 - 60) / x_range, (frame_h - 60) / z_range)

                # Center
                cx = (frame_w // 2) / 2
                cy = frame_h / 2
                mean_x = (xs.max() + xs.min()) / 2
                mean_z = (zs.max() + zs.min()) / 2

                # Draw trajectory line
                pts = []
                for j in range(len(xs)):
                    px = int(cx + (xs[j] - mean_x) * scale)
                    py = int(cy + (zs[j] - mean_z) * scale)
                    px = max(0, min(frame_w // 2 - 1, px))
                    py = max(0, min(frame_h - 1, py))
                    pts.append((px, py))

                for j in range(len(pts) - 1):
                    cv2.line(traj_panel, pts[j], pts[j + 1], (200, 150, 50), 2)

                # Start marker (green)
                cv2.circle(traj_panel, pts[0], 6, (0, 200, 0), -1)
                # Current position (red)
                cv2.circle(traj_panel, pts[-1], 6, (0, 0, 230), -1)

            # Title
            cv2.putText(
                traj_panel, "Trajectory (top-down)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1,
            )

            combined[:, frame_w // 2:] = traj_panel
            writer.write(combined)
    finally:
        writer.release()
