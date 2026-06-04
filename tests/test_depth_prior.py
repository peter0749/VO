import os
import tempfile
import shutil
import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock

from slam_dnn.depth import DepthMapLoader, DepthAnythingEstimator
from slam_dnn.config import VOConfig
from slam_dnn.camera import PinholeCamera
from slam_dnn.vo import VisualOdometry


@pytest.fixture
def temp_depth_dir():
    """Create a temporary directory with mock depth files."""
    tmpdir = tempfile.mkdtemp()
    
    # 1. Write a 16-bit PNG depth map (values scaled by 256.0)
    # Let's create a depth of 5.0m -> value = 5.0 * 256.0 = 1280
    png_depth = np.full((100, 100), 1280, dtype=np.uint16)
    cv2.imwrite(os.path.join(tmpdir, "000000.png"), png_depth)
    
    # 2. Write a .npy file (metric values directly)
    npy_depth = np.full((100, 100), 4.2, dtype=np.float32)
    np.save(os.path.join(tmpdir, "000001.npy"), npy_depth)
    
    # 3. Write a .npz file
    npz_depth = np.full((100, 100), 3.5, dtype=np.float32)
    np.savez(os.path.join(tmpdir, "000002.npz"), depth=npz_depth)
    
    yield tmpdir
    
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_depth_map_loader(temp_depth_dir):
    """Verify DepthMapLoader loads and scales depth maps for PNG, npy, and npz formats."""
    loader = DepthMapLoader(temp_depth_dir, scale_factor=256.0)
    
    assert len(loader.depth_files) == 3
    
    # Check PNG (Frame 0)
    d0 = loader.get_depth(0)
    assert d0.shape == (100, 100)
    assert np.allclose(d0, 5.0)
    
    # Check NPY (Frame 1)
    d1 = loader.get_depth(1)
    assert d1.shape == (100, 100)
    assert np.allclose(d1, 4.2)
    
    # Check NPZ (Frame 2)
    d2 = loader.get_depth(2)
    assert d2.shape == (100, 100)
    assert np.allclose(d2, 3.5)
    
    # Index error check
    with pytest.raises(IndexError):
        loader.get_depth(3)


def test_depth_prior_vo_noiseless():
    """Verify that the depth-prior VO tracker estimates correct poses in noise-free conditions."""
    # Camera settings
    w, h = 640, 480
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    camera = PinholeCamera(width=w, height=h, fov_deg=63.0)
    camera.K = K
    camera.K_inv = np.linalg.inv(K)
    
    # Ground truth camera trajectory
    # Frame 0 (Keyframe 0): Identity
    # Frame 1: Translate along X and Z
    T0 = np.eye(4)
    
    T1 = np.eye(4)
    T1[0, 3] = 0.25  # translation along X
    T1[2, 3] = 1.0   # translation along Z
    
    # PnP poses are world-to-camera, so we invert camera position: T_c_w
    # Wait, T1 above represents camera translation in world coordinates, i.e., T_w_c.
    # The world-to-camera pose T_c_w = T_w_c^-1
    T0_c_w = np.eye(4)
    T1_c_w = np.linalg.inv(T1)
    
    # Generate 50 points in front of the camera (world coordinates)
    np.random.seed(42)
    pts_w = np.zeros((50, 3))
    pts_w[:, 0] = np.random.uniform(-1.0, 1.0, 50)
    pts_w[:, 1] = np.random.uniform(-1.0, 1.0, 50)
    pts_w[:, 2] = np.random.uniform(4.0, 6.0, 50)  # depths between 4.0 and 6.0m
    
    # Project to Frame 0
    pts_c0 = pts_w  # since T0_c_w = I
    kps0_proj = (K @ pts_c0.T).T
    kps0 = kps0_proj[:, :2] / kps0_proj[:, 2:3]
    
    # Project to Frame 1
    pts_c1 = (T1_c_w[:3, :3] @ pts_w.T).T + T1_c_w[:3, 3]
    kps1_proj = (K @ pts_c1.T).T
    kps1 = kps1_proj[:, :2] / kps1_proj[:, 2:3]
    
    # Setup mock depth maps
    # In Frame 0, the depths at the keypoints are exactly pts_w[:, 2]
    # We create a depth map of size h x w and fill it with these values at nearest pixel coords
    depth0 = np.zeros((h, w), dtype=np.float32)
    u0 = np.clip(np.round(kps0[:, 0]).astype(np.int32), 0, w - 1)
    v0 = np.clip(np.round(kps0[:, 1]).astype(np.int32), 0, h - 1)
    depth0[v0, u0] = pts_w[:, 2]
    
    # In Frame 1, the depths are pts_c1[:, 2]
    depth1 = np.zeros((h, w), dtype=np.float32)
    u1 = np.clip(np.round(kps1[:, 0]).astype(np.int32), 0, w - 1)
    v1 = np.clip(np.round(kps1[:, 1]).astype(np.int32), 0, h - 1)
    depth1[v1, u1] = pts_c1[:, 2]
    
    # Initialize config
    config = VOConfig(
        use_depth_prior=True,
        min_inliers_pnp=10,
        min_matches=10,
        ransac_threshold=1.5,
    )
    
    # Create VisualOdometry
    vo = VisualOdometry(camera=camera, config=config)
    
    # Mock the DepthMapLoader
    mock_depth_loader = MagicMock()
    mock_depth_loader.get_depth.side_effect = [depth0, depth1]
    vo.depth_loader = mock_depth_loader
    
    # Mock the Extractor to return our projected keypoints
    mock_extractor = MagicMock()
    # First call (Frame 0), second call (Frame 1)
    mock_extractor.extract.side_effect = [
        {"keypoints": kps0, "descriptors": np.zeros((50, 128))},
        {"keypoints": kps1, "descriptors": np.zeros((50, 128))},
    ]
    vo.extractor = mock_extractor
    
    # Mock the Matcher to return identity matching
    mock_matcher = MagicMock()
    indices = np.arange(50)[:, None]
    match_result = {
        "points0": kps0,
        "points1": kps1,
        "indices": np.hstack([indices, indices]),
    }
    mock_matcher.match.return_value = match_result
    vo.matcher = mock_matcher
    
    # Process Frame 0 (first keyframe)
    dummy_img0 = np.zeros((h, w, 3), dtype=np.uint8)
    pose0 = vo.process_frame(dummy_img0)
    assert pose0 is None
    assert len(vo._keyframe_poses) == 1
    assert np.allclose(vo._keyframe_poses[0], T0_c_w)
    
    # Process Frame 1
    dummy_img1 = np.zeros((h, w, 3), dtype=np.uint8)
    pose1 = vo.process_frame(dummy_img1)
    
    assert pose1 is not None
    # pose1 is 3x4 relative pose of Frame 1 w.r.t. Keyframe 0.
    # T_curr_kf = T_curr_w @ T_kf_w_inv
    # Since T_kf_w = T0_c_w = I, relative pose = T1_c_w
    R_rel = pose1[:, :3]
    t_rel = pose1[:, 3]
    
    # Compare with ground truth relative pose
    assert np.allclose(R_rel, T1_c_w[:3, :3], atol=1e-3)
    assert np.allclose(t_rel, T1_c_w[:3, 3], atol=1e-3)
    
    # Check that scale-drift-free metric trajectory is kept
    poses = vo.get_trajectory().get_poses()
    assert len(poses) == 2
    
    # poses[1] should be T1_c_w or its equivalent in world frame
    assert np.allclose(poses[1], T1_c_w, atol=1e-3)


def test_depth_anything_estimator_mock():
    """Verify DepthAnythingEstimator initialization and inference using mocked Hugging Face models."""
    from unittest.mock import patch, MagicMock
    import torch
    
    with patch("transformers.AutoImageProcessor.from_pretrained") as mock_processor_load, \
         patch("transformers.AutoModelForDepthEstimation.from_pretrained") as mock_model_load:
         
        # Setup mock processor and model
        mock_processor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs
        mock_processor_load.return_value = mock_processor
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_output = MagicMock()
        # Mock predicted depth output tensor shape (1, 192, 320)
        mock_output.predicted_depth = torch.ones((1, 192, 320))
        mock_model.return_value = mock_output
        mock_model_load.return_value = mock_model
        
        # Instantiate estimator (force device to 'cpu' for testing)
        estimator = DepthAnythingEstimator(
            model_name="dummy-repo/depth-anything-small-hf",
            target_resolution=(320, 192),
            device="cpu"
        )
        
        assert estimator.device == torch.device("cpu")
        
        # Estimate depth on dummy image
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = estimator.estimate_depth(dummy_img)
        
        # Verify shape matches original image, and values are non-zero
        assert depth.shape == (480, 640)
        assert np.all(depth > 0.0)


def test_depth_prior_vo_model_mode_noiseless():
    """Verify that Depth-Prior VO estimates poses and scales dynamically using model depth predictions."""
    # Camera settings
    w, h = 640, 480
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    camera = PinholeCamera(width=w, height=h, fov_deg=63.0)
    camera.K = K
    camera.K_inv = np.linalg.inv(K)
    
    # GT Trajectory (Translate along X and Z)
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 0.3
    T1[2, 3] = 1.5
    
    T0_c_w = np.eye(4)
    T1_c_w = np.linalg.inv(T1)
    
    # Generate 50 points
    np.random.seed(42)
    pts_w = np.zeros((50, 3))
    pts_w[:, 0] = np.random.uniform(-1.0, 1.0, 50)
    pts_w[:, 1] = np.random.uniform(-1.0, 1.0, 50)
    pts_w[:, 2] = np.random.uniform(4.0, 6.0, 50)
    
    # Project to Frame 0 and Frame 1
    pts_c0 = pts_w
    kps0_proj = (K @ pts_c0.T).T
    kps0 = kps0_proj[:, :2] / kps0_proj[:, 2:3]
    
    pts_c1 = (T1_c_w[:3, :3] @ pts_w.T).T + T1_c_w[:3, 3]
    kps1_proj = (K @ pts_c1.T).T
    kps1 = kps1_proj[:, :2] / kps1_proj[:, 2:3]
    
    # Generate predicted relative depth maps (say, they are scaled by 0.5 compared to metric depth)
    scale_offset = 0.5
    
    depth0_rel = np.zeros((h, w), dtype=np.float32)
    u0 = np.clip(np.round(kps0[:, 0]).astype(np.int32), 0, w - 1)
    v0 = np.clip(np.round(kps0[:, 1]).astype(np.int32), 0, h - 1)
    depth0_rel[v0, u0] = pts_w[:, 2] * scale_offset
    
    depth1_rel = np.zeros((h, w), dtype=np.float32)
    u1 = np.clip(np.round(kps1[:, 0]).astype(np.int32), 0, w - 1)
    v1 = np.clip(np.round(kps1[:, 1]).astype(np.int32), 0, h - 1)
    depth1_rel[v1, u1] = pts_c1[:, 2] * scale_offset
    
    # Config
    config = VOConfig(
        use_depth_prior=True,
        depth_source="model",
        depth_scale_mode="median_ratio",
        depth_scale_factor=1.0,
        min_inliers_pnp=10,
        min_matches=10,
        ransac_threshold=1.5,
    )
    
    from unittest.mock import patch
    with patch("slam_dnn.depth.DepthAnythingEstimator") as mock_estimator_cls:
        mock_estimator = MagicMock()
        mock_estimator.estimate_depth.side_effect = [depth0_rel, depth1_rel]
        mock_estimator_cls.return_value = mock_estimator
        
        # VisualOdometry
        vo = VisualOdometry(camera=camera, config=config)
        
        # Mock extractor
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = [
            {"keypoints": kps0, "descriptors": np.zeros((50, 128))},
            {"keypoints": kps1, "descriptors": np.zeros((50, 128))},
        ]
        vo.extractor = mock_extractor
        
        # Mock matcher
        mock_matcher = MagicMock()
        indices = np.arange(50)[:, None]
        match_result = {
            "points0": kps0,
            "points1": kps1,
            "indices": np.hstack([indices, indices]),
        }
        mock_matcher.match.return_value = match_result
        vo.matcher = mock_matcher
        
        # Run Frame 0
        dummy_img0 = np.zeros((h, w, 3), dtype=np.uint8)
        pose0 = vo.process_frame(dummy_img0)
        assert pose0 is None
        
        # Check that map points in Frame 0 are scaled by depth_scale_factor (1.0)
        assert np.allclose(vo._map_points_3d[:, 2], pts_w[:, 2] * scale_offset, atol=1e-3)
        
        # Run Frame 1 (which will trigger median ratio scale calibration!)
        dummy_img1 = np.zeros((h, w, 3), dtype=np.uint8)
        pose1 = vo.process_frame(dummy_img1)
        assert pose1 is not None

