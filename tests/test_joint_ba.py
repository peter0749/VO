import numpy as np
import pytest
import cv2
from slam_dnn.pose import triangulate_points
from slam_dnn.local_ba import LocalBundleAdjuster
from slam_dnn.vo import VisualOdometry
from slam_dnn.config import VOConfig
from slam_dnn.camera import PinholeCamera


def test_triangulate_points_noiseless():
    """Verify that triangulate_points works with zero error under noiseless conditions."""
    P1 = np.eye(4)[:3, :]
    
    # Pose 2: translated by 0.2 along X, 1.2 along Z (ensuring non-zero parallax)
    T2 = np.eye(4)
    T2[0, 3] = 0.2
    T2[2, 3] = 1.2
    P2 = T2[:3, :]
    
    # Ground truth 3D points
    true_pts = np.array([
        [-0.5, 0.2, 5.0],
        [0.5, -0.2, 4.0],
        [0.2, 0.1, 6.0],
    ])
    
    # Project to normalized camera coordinates
    kpn1 = true_pts[:, :2] / true_pts[:, 2:3]
    
    pts_cam2 = (T2[:3, :3] @ true_pts.T).T + T2[:3, 3]
    kpn2 = pts_cam2[:, :2] / pts_cam2[:, 2:3]
    
    # Triangulate
    est_pts = triangulate_points(P1, P2, kpn1, kpn2)
    
    # Verify correctness
    assert np.allclose(true_pts, est_pts, atol=1e-5)


def test_local_ba_noiseless():
    """Verify that LocalBundleAdjuster converges and refines poses and structure."""
    # Intrinsics
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # True poses: 3 frames
    poses = []
    
    # Frame 1
    poses.append(np.eye(4))
    
    # Frame 2
    T2 = np.eye(4)
    T2[0, 3] = 0.1
    T2[2, 3] = 1.0
    poses.append(T2)
    
    # Frame 3
    T3 = np.eye(4)
    T3[0, 3] = 0.2
    T3[2, 3] = 2.0
    poses.append(T3)
    
    # Generate 15 points
    np.random.seed(42)
    points_3d = np.zeros((15, 3))
    points_3d[:, 0] = np.random.uniform(-0.5, 0.5, 15)
    points_3d[:, 1] = np.random.uniform(-0.5, 0.5, 15)
    points_3d[:, 2] = np.random.uniform(3.0, 6.0, 15)
    
    # Project and construct observations for all 3 frames
    observations = []
    for c_idx, T in enumerate(poses):
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3]
        pts_proj, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
        uvs = pts_proj.reshape(-1, 2)
        for pt_idx, uv in enumerate(uvs):
            observations.append({
                "cam_idx": c_idx,
                "pt_idx": pt_idx,
                "uv": uv
            })
            
    # Add noise to initial poses and points to see if optimizer refines them
    # Keep first two poses (Frame 1 and Frame 2) exact to lock scale
    noisy_poses = [poses[0].copy(), poses[1].copy()]
    
    # Add noise to Frame 3 pose
    T3_noisy = T3.copy()
    T3_noisy[0, 3] += 0.05
    T3_noisy[2, 3] -= 0.04
    noisy_poses.append(T3_noisy)
    
    # Add noise to points
    noisy_points_3d = points_3d.copy()
    noisy_points_3d[:, 0] += 0.02
    noisy_points_3d[:, 2] += 0.06
    
    # Optimize (first two poses are fixed, third is optimized, all points optimized)
    ba = LocalBundleAdjuster(window_size=4)
    opt_poses, opt_pts = ba.optimize(
        noisy_poses, noisy_points_3d, observations, K, fix_first_two=True
    )
    
    # Verify optimization reduced Frame 3 translation error
    init_err = np.linalg.norm(T3[:3, 3] - T3_noisy[:3, 3])
    opt_err = np.linalg.norm(T3[:3, 3] - opt_poses[2][:3, 3])
    assert opt_err < init_err
    
    # Verify points error reduced
    init_pt_err = np.mean(np.linalg.norm(points_3d - noisy_points_3d, axis=1))
    opt_pt_err = np.mean(np.linalg.norm(points_3d - opt_pts, axis=1))
    assert opt_pt_err < init_pt_err


def test_visual_odometry_config():
    """Verify that VisualOdometry can be initialized with use_joint_ba config."""
    camera = PinholeCamera(width=640, height=480, fov_deg=63.0)
    config = VOConfig(use_joint_ba=True, extractor='xfeat', matcher='classic')
    
    vo = VisualOdometry(camera=camera, config=config)
    assert vo.config.use_joint_ba is True
    assert len(vo._keyframe_poses) == 0
    assert vo._map_points_3d.shape == (0, 3)
