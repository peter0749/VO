"""High-level Visual Odometry orchestrator."""
import numpy as np
import cv2
import logging
import time
from typing import Iterator

from .camera import PinholeCamera
from .config import VOConfig
from .features import SuperPointExtractor, XFeatExtractor
from .matching import MatcherBase, create_matcher
from .pose import estimate_essential, estimate_essential_or_homography, triangulate_points
from .trajectory import TrajectoryAccumulator
from .keyframe import KeyframeSelector
from .motion_model import MotionModel
from .local_ba import LocalBundleAdjuster


logger = logging.getLogger(__name__)


class VisualOdometry:
    """Facade that orchestrates the full VO pipeline.

    Example:
        >>> from slam_dnn import VisualOdometry, PinholeCamera
        >>> camera = PinholeCamera(width=640, height=480, fov_deg=63)
        >>> vo = VisualOdometry(camera, matcher='lightglue', device='cpu')
        >>> for frame in video_frames:
        ...     pose = vo.process_frame(frame)
    """

    def __init__(
        self,
        camera: PinholeCamera,
        matcher: str | MatcherBase = "lightglue",
        max_keypoints: int = 1024,
        scale: float = 1.0,
        device: str = "auto",
        min_matches: int = 20,
        config: VOConfig | None = None,
    ):
        """
        Args:
            camera: PinholeCamera with intrinsics.
            matcher: "lightglue", "classic", or MatcherBase instance.
            max_keypoints: SuperPoint max keypoints per frame.
            scale: Trajectory scale factor.
            device: "auto" | "cuda" | "mps" | "cpu".
            min_matches: Minimum matches required for pose estimation.
            config: Optional VOConfig; overrides other kwargs when provided.
        """
        if config is None:
            config = VOConfig()
            # Preserve overrides if custom values were explicitly passed as kwargs
            if max_keypoints != 1024:
                config.max_keypoints = max_keypoints
            if scale != 1.0:
                config.scale = scale
            if device != "auto":
                config.device = device
            if min_matches != 20:
                config.min_matches = min_matches
            if isinstance(matcher, str):
                if matcher == "lightglue":
                    matcher = config.matcher
                else:
                    config.matcher = matcher
        else:
            # Respect overrides when config is provided but explicitly overridden in kwargs
            if max_keypoints != 1024:
                config.max_keypoints = max_keypoints
            if scale != 1.0:
                config.scale = scale
            if device != "auto":
                config.device = device
            if min_matches != 20:
                config.min_matches = min_matches
            if isinstance(matcher, str) and matcher == "lightglue":
                matcher = config.matcher

        self.camera = camera
        self.min_matches = config.min_matches
        self._handle_pure_rotation = config.handle_pure_rotation
        self._ransac_threshold = config.ransac_threshold
        self._ransac_confidence = config.ransac_confidence

        if config.extractor == 'superpoint':
            self.extractor = SuperPointExtractor(
                max_keypoints=config.max_keypoints,
                conf_thresh=config.detection_threshold,
                device=config.device,
                target_resolution=config.target_resolution,
            )
        elif config.extractor == 'xfeat':
            self.extractor = XFeatExtractor(
                max_keypoints=config.max_keypoints,
                conf_thresh=config.detection_threshold,
                device=config.device,
                target_resolution=config.target_resolution,
            )
        else:
            raise ValueError(f"Unknown extractor: {config.extractor}")

        if isinstance(matcher, MatcherBase):
            self.matcher = matcher
        else:
            if matcher == "lightglue":
                self.matcher = create_matcher(
                    matcher,
                    filter_threshold=config.lightglue_threshold,
                    device=config.device,
                )
            elif matcher == "classic":
                self.matcher = create_matcher(
                    matcher,
                    ratio=config.classic_ratio,
                    device=config.device,
                )
            elif matcher == "xfeat":
                self.matcher = create_matcher(
                    matcher,
                    min_cossim=config.xfeat_min_cossim,
                    device=config.device,
                )
            else:
                self.matcher = create_matcher(matcher, device=config.device)

        self.trajectory = TrajectoryAccumulator(scale=scale)
        self._prev_feats = None
        self._frame_idx = 0
        
        # Keyframe & Motion Model Heuristics
        self._use_keyframe_selection = config.use_keyframe_selection
        self.keyframe_selector = KeyframeSelector(
            min_parallax=config.min_parallax,
            max_overlap=config.max_overlap,
            max_interval=config.max_keyframe_interval,
        )
        
        self._use_motion_model = config.use_motion_model
        self.motion_model = MotionModel(ema_alpha=config.motion_model_alpha)
        self.config = config

        # Map / Sliding Window BA States
        self._keyframe_poses: list[np.ndarray] = []  # world-to-camera keyframe poses
        self._keyframe_feats: list[dict] = []        # features for keyframes
        self._keyframe_kp_to_3d: list[list[int | None]] = []  # kp_idx -> global 3d point index
        self._map_points_3d: np.ndarray = np.zeros((0, 3))    # N x 3 global 3D points
        self.local_ba = LocalBundleAdjuster(window_size=config.ba_window_size)

        # Depth Prior Loader
        if config.use_depth_prior:
            if config.depth_source == 'directory':
                from .depth import DepthMapLoader
                self.depth_loader = DepthMapLoader(
                    directory=config.depth_directory,
                    scale_factor=config.depth_scale_factor
                )
                self.depth_estimator = None
            elif config.depth_source == 'model':
                if 'moge' in config.depth_model_name.lower():
                    from .depth import MoGeEstimator
                    self.depth_estimator = MoGeEstimator(
                        model_name=config.depth_model_name,
                        device=config.device
                    )
                else:
                    from .depth import DepthAnythingEstimator
                    self.depth_estimator = DepthAnythingEstimator(
                        model_name=config.depth_model_name,
                        target_resolution=config.depth_target_resolution,
                        device=config.device
                    )
                self.depth_loader = None
            else:
                raise ValueError(f"Unknown depth_source: {config.depth_source}")
        else:
            self.depth_loader = None
            self.depth_estimator = None

        # Scale calibration states
        self._calib_scale_factor = 1.0
        self._calib_ratios = []
        self._calib_finished = False
        self._calib_frames = config.depth_calib_frames
        
        # Ground plane camera height
        self._camera_height = config.camera_height
        
        # Sliding window history of scale factors
        self._sliding_window_scales = []
        self._sliding_window_size = config.depth_sliding_window_size

        self._per_frame_stats: list[dict] = []
        self._stats = {
            "total": 0,
            "successful": 0,
            "tracking_lost": 0,
            "pose_failed": 0,
            "keyframes": 0,
            "motion_model_fallbacks": 0,
        }
        self._timings = {
            "extraction": 0.0,
            "matching": 0.0,
            "pose_estimation": 0.0,
            "total": 0.0,
        }

    def process_frame(self, image: np.ndarray) -> np.ndarray | None:
        """Process one frame, return relative pose (3x4) if successful, else None.

        The first frame always returns None (no previous frame to compare against).
        """
        if self.config.use_depth_prior:
            return self._process_frame_depth_prior(image)

        if self.config.use_joint_ba:
            return self._process_frame_joint_ba(image)

        t_start = time.perf_counter()
        self._stats["total"] += 1
        
        t_extract_start = time.perf_counter()
        feats = self.extractor.extract(image)
        self._timings["extraction"] += time.perf_counter() - t_extract_start

        if self._prev_feats is None:
            self._prev_feats = feats
            self._frame_idx += 1
            self._stats["keyframes"] += 1
            self._per_frame_stats.append({
                "frame_idx": self._frame_idx - 1,
                "num_matches": 0,
                "num_inliers": 0,
                "tracking_lost": False,
                "pose_failed": False,
                "is_keyframe": True,
                "motion_model_fallback": False,
            })
            self._timings["total"] += time.perf_counter() - t_start
            return None

        t_match_start = time.perf_counter()
        match_result = self.matcher.match(
            self._prev_feats, feats, image_size=image.shape[:2]
        )
        self._timings["matching"] += time.perf_counter() - t_match_start
        n_matches = len(match_result["points0"])

        # 1. Tracking lost handling
        if n_matches < self.min_matches:
            self._stats["tracking_lost"] += 1
            logger.warning(
                f"Frame {self._frame_idx}: tracking lost "
                f"({n_matches} < {self.min_matches})"
            )
            
            if self._use_motion_model:
                R, t = self.motion_model.predict()
                self.trajectory.add_pose_relative_to_keyframe(R, t, is_keyframe=False)
                self._stats["motion_model_fallbacks"] += 1
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": True,
                    "pose_failed": False,
                    "is_keyframe": False,
                    "motion_model_fallback": True,
                })
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                pose_3x4 = np.hstack([R, t.reshape(3, 1)])
                return pose_3x4
            else:
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": True,
                    "pose_failed": False,
                    "is_keyframe": False,
                    "motion_model_fallback": False,
                })
                self._prev_feats = feats
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                return None

        # 2. Estimate relative pose
        t_pose_start = time.perf_counter()
        pose_fn = estimate_essential_or_homography if self._handle_pure_rotation else estimate_essential
        result = pose_fn(
            match_result["points0"],
            match_result["points1"],
            self.camera.K,
            ransac_thresh=self._ransac_threshold,
            conf=self._ransac_confidence,
        )
        self._timings["pose_estimation"] += time.perf_counter() - t_pose_start

        # 3. Pose estimation failed handling
        if result is None:
            self._stats["pose_failed"] += 1
            logger.warning(f"Frame {self._frame_idx}: pose estimation failed")
            
            if self._use_motion_model:
                R, t = self.motion_model.predict()
                self.trajectory.add_pose_relative_to_keyframe(R, t, is_keyframe=False)
                self._stats["motion_model_fallbacks"] += 1
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": False,
                    "pose_failed": True,
                    "is_keyframe": False,
                    "motion_model_fallback": True,
                })
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                pose_3x4 = np.hstack([R, t.reshape(3, 1)])
                return pose_3x4
            else:
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": False,
                    "pose_failed": True,
                    "is_keyframe": False,
                    "motion_model_fallback": False,
                })
                self._prev_feats = feats
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                return None

        # 4. Successful pose estimation: decide if we insert a keyframe
        R, t, inlier_mask = result
        
        is_kf = True
        if self._use_keyframe_selection:
            is_kf = self.keyframe_selector.should_insert(
                match_result["points0"],
                match_result["points1"],
                len(feats["keypoints"]),
            )

        self.trajectory.add_pose_relative_to_keyframe(R, t, is_keyframe=is_kf)
        self.motion_model.update(R, t)
        
        if is_kf:
            self._prev_feats = feats
            self._stats["keyframes"] += 1

        self._frame_idx += 1
        self._stats["successful"] += 1
        self._per_frame_stats.append({
            "frame_idx": self._frame_idx - 1,
            "num_matches": n_matches,
            "num_inliers": int(inlier_mask.sum()),
            "tracking_lost": False,
            "pose_failed": False,
            "is_keyframe": is_kf,
            "motion_model_fallback": False,
        })

        self._timings["total"] += time.perf_counter() - t_start
        pose_3x4 = np.hstack([R, t.reshape(3, 1)])
        return pose_3x4

    def _process_frame_depth_prior(self, image: np.ndarray) -> np.ndarray | None:
        t_start = time.perf_counter()
        self._stats["total"] += 1

        t_extract_start = time.perf_counter()
        feats = self.extractor.extract(image)
        self._timings["extraction"] += time.perf_counter() - t_extract_start

        # Initialization: Frame 0 (first keyframe)
        if len(self._keyframe_poses) == 0:
            T0 = np.eye(4)
            self._keyframe_poses.append(T0)
            self._keyframe_feats.append(feats)
            
            # Load depth map
            if self.config.depth_source == 'directory':
                depth = self.depth_loader.get_depth(0)
            elif self.config.depth_source == 'model':
                from .depth import MoGeEstimator
                if isinstance(self.depth_estimator, MoGeEstimator):
                    K = self.camera.K
                    W = self.camera.width
                    fov_x_deg = float(2.0 * np.arctan(W / (2.0 * K[0, 0])) * 180.0 / np.pi)
                    depth = self.depth_estimator.estimate_depth(image, fov_x=fov_x_deg)
                else:
                    depth = self.depth_estimator.estimate_depth(image)
                
                # Default scale factor
                s = self.config.depth_scale_factor
                
                if self.config.depth_scale_mode in ['ground_plane', 'calibrate']:
                    # Try to fit ground plane to the first frame to establish a metric scale anchor
                    K_inv = np.linalg.inv(self.camera.K)
                    kps = feats["keypoints"]
                    N_kps = len(kps)
                    if N_kps > 0:
                        u_kps = kps[:, 0]
                        v_kps = kps[:, 1]
                        u_idx_kps = np.clip(np.round(u_kps).astype(np.int32), 0, depth.shape[1] - 1)
                        v_idx_kps = np.clip(np.round(v_kps).astype(np.int32), 0, depth.shape[0] - 1)
                        d_kps = depth[v_idx_kps, u_idx_kps]
                        
                        valid_mask = (d_kps > 0.1) & (d_kps < 150.0)
                        kps_hom = np.hstack([kps, np.ones((N_kps, 1))])
                        p_c = (K_inv @ kps_hom.T).T
                        pts_3d_c = p_c * d_kps.reshape(-1, 1)
                        
                        candidates = pts_3d_c[valid_mask]
                        candidates_mask = (kps[valid_mask, 1] > image.shape[0] * 0.5) & (candidates[:, 1] > 0.1) & (candidates[:, 2] < 40.0)
                        plane_pts = candidates[candidates_mask]
                        
                        plane_fit = fit_ground_plane_ransac(plane_pts, camera_height_prior=self._camera_height)
                        if plane_fit is not None:
                            _, d_plane, _ = plane_fit
                            s = self._camera_height / abs(d_plane)
                            logger.info(f"Frame 0: Calibrated initial scale factor from ground plane = {s:.4f} (height = {abs(d_plane):.2f}m)")
                            if self.config.depth_scale_mode == 'calibrate':
                                self._calib_ratios.append(s)
                
                depth = depth * s
            else:
                raise ValueError(f"Unknown depth_source: {self.config.depth_source}")

            if depth.shape[:2] != image.shape[:2]:
                depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            # Backproject 2D keypoints to 3D
            kps = feats["keypoints"]
            N = len(kps)
            kp_to_3d = [None] * N
            
            if N > 0:
                K_inv = np.linalg.inv(self.camera.K)
                
                u = kps[:, 0]
                v = kps[:, 1]
                u_idx = np.clip(np.round(u).astype(np.int32), 0, depth.shape[1] - 1)
                v_idx = np.clip(np.round(v).astype(np.int32), 0, depth.shape[0] - 1)
                d = depth[v_idx, u_idx]
                
                valid_depth_mask = (d > 0.1) & (d < 150.0) & (~np.isnan(d)) & (~np.isinf(d))
                kps_hom = np.hstack([kps, np.ones((N, 1))])
                p_c = (K_inv @ kps_hom.T).T
                pts_3d = p_c * d.reshape(-1, 1)
                
                valid_indices = np.where(valid_depth_mask)[0]
                if len(valid_indices) > 0:
                    self._map_points_3d = pts_3d[valid_indices]
                    for i, idx in enumerate(valid_indices):
                        kp_to_3d[idx] = i
            
            self._keyframe_kp_to_3d.append(kp_to_3d)
            self._prev_feats = feats
            self._frame_idx += 1
            self._stats["keyframes"] += 1
            self._per_frame_stats.append({
                "frame_idx": self._frame_idx - 1,
                "num_matches": 0,
                "num_inliers": 0,
                "tracking_lost": False,
                "pose_failed": False,
                "is_keyframe": True,
                "motion_model_fallback": False,
            })
            self._timings["total"] += time.perf_counter() - t_start
            return None

        # Subsequent Frames: 3D-2D Tracking against previous keyframe
        feats_kf = self._keyframe_feats[-1]
        t_match_start = time.perf_counter()
        match_result = self.matcher.match(
            feats_kf, feats, image_size=image.shape[:2]
        )
        self._timings["matching"] += time.perf_counter() - t_match_start
        n_matches = len(match_result["points0"])

        # Gather 3D-2D correspondences
        points_3d_pnp = []
        points_2d_pnp = []
        match_indices_pnp = []

        for j in range(n_matches):
            idx_kf = match_result["indices"][j, 0]
            pt_idx = self._keyframe_kp_to_3d[-1][idx_kf]
            if pt_idx is not None:
                points_3d_pnp.append(self._map_points_3d[pt_idx])
                points_2d_pnp.append(match_result["points1"][j])
                match_indices_pnp.append(j)

        n_pnp = len(points_3d_pnp)

        # Check tracking status
        if n_pnp < self.config.min_inliers_pnp:
            self._stats["tracking_lost"] += 1
            logger.warning(
                f"Frame {self._frame_idx}: 3D-2D depth-prior tracking lost "
                f"({n_pnp} < {self.config.min_inliers_pnp})"
            )
            
            if self._use_motion_model:
                R_pred, t_pred = self.motion_model.predict()
                self.trajectory.add_pose_metric(R_pred, t_pred, is_keyframe=False)
                self._stats["motion_model_fallbacks"] += 1
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": True,
                    "pose_failed": False,
                    "is_keyframe": False,
                    "motion_model_fallback": True,
                })
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                pose_3x4 = np.hstack([R_pred, t_pred.reshape(3, 1)])
                return pose_3x4
            else:
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": True,
                    "pose_failed": False,
                    "is_keyframe": False,
                    "motion_model_fallback": False,
                })
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                return None

        # Run PnP with initial guess from motion model
        t_pose_start = time.perf_counter()
        T_kf_w = self._keyframe_poses[-1]
        R_pred, t_pred = self.motion_model.predict()
        T_curr_kf = np.eye(4)
        T_curr_kf[:3, :3] = R_pred
        T_curr_kf[:3, 3] = t_pred
        T_pred_w = T_curr_kf @ T_kf_w

        rvec_pred, _ = cv2.Rodrigues(T_pred_w[:3, :3])
        tvec_pred = T_pred_w[:3, 3]

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(points_3d_pnp, dtype=np.float64),
            np.array(points_2d_pnp, dtype=np.float64),
            self.camera.K,
            distCoeffs=None,
            rvec=rvec_pred.copy(),
            tvec=tvec_pred.copy(),
            useExtrinsicGuess=True,
            iterationsCount=150,
            reprojectionError=self._ransac_threshold,
            confidence=self._ransac_confidence
        )
        self._timings["pose_estimation"] += time.perf_counter() - t_pose_start

        if not success or inliers is None or len(inliers) < self.config.min_inliers_pnp:
            self._stats["pose_failed"] += 1
            logger.warning(f"Frame {self._frame_idx}: PnP tracking failed")
            
            if self._use_motion_model:
                R_pred, t_pred = self.motion_model.predict()
                self.trajectory.add_pose_metric(R_pred, t_pred, is_keyframe=False)
                self._stats["motion_model_fallbacks"] += 1
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": False,
                    "pose_failed": True,
                    "is_keyframe": False,
                    "motion_model_fallback": True,
                })
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                pose_3x4 = np.hstack([R_pred, t_pred.reshape(3, 1)])
                return pose_3x4
            else:
                self._per_frame_stats.append({
                    "frame_idx": self._frame_idx,
                    "num_matches": n_matches,
                    "num_inliers": 0,
                    "tracking_lost": False,
                    "pose_failed": True,
                    "is_keyframe": False,
                    "motion_model_fallback": False,
                })
                self._frame_idx += 1
                self._timings["total"] += time.perf_counter() - t_start
                return None

        # PnP tracking success
        t_curr = tvec.flatten()
        R_curr, _ = cv2.Rodrigues(rvec)
        T_curr_w = np.eye(4)
        T_curr_w[:3, :3] = R_curr
        T_curr_w[:3, 3] = t_curr

        R_kf = T_kf_w[:3, :3]
        t_kf = T_kf_w[:3, 3]
        T_kf_w_inv = np.eye(4)
        T_kf_w_inv[:3, :3] = R_kf.T
        T_kf_w_inv[:3, 3] = -R_kf.T @ t_kf

        T_curr_kf = T_curr_w @ T_kf_w_inv
        R_rel = T_curr_kf[:3, :3]
        t_rel = T_curr_kf[:3, 3]

        is_kf = False
        if self._use_keyframe_selection:
            is_kf = self.keyframe_selector.should_insert(
                match_result["points0"],
                match_result["points1"],
                len(feats["keypoints"]),
            )

        if is_kf:
            self._keyframe_poses.append(T_curr_w)
            self._keyframe_feats.append(feats)
            self._keyframe_kp_to_3d.append([None] * len(feats["keypoints"]))

            # Propagate 3D points for inliers
            for idx in inliers.flatten():
                j = match_indices_pnp[idx]
                idx_curr = match_result["indices"][j, 1]
                idx_kf = match_result["indices"][j, 0]
                map_pt_idx = self._keyframe_kp_to_3d[-2][idx_kf]
                self._keyframe_kp_to_3d[-1][idx_curr] = map_pt_idx

            # Load current depth map and project new keypoints
            if self.config.depth_source == 'directory':
                depth = self.depth_loader.get_depth(self._frame_idx)
            elif self.config.depth_source == 'model':
                from .depth import MoGeEstimator
                if isinstance(self.depth_estimator, MoGeEstimator):
                    K = self.camera.K
                    W = self.camera.width
                    fov_x_deg = float(2.0 * np.arctan(W / (2.0 * K[0, 0])) * 180.0 / np.pi)
                    depth = self.depth_estimator.estimate_depth(image, fov_x=fov_x_deg)
                else:
                    depth = self.depth_estimator.estimate_depth(image)
                
                # Default scale factor
                s = self.config.depth_scale_factor
                
                if self.config.depth_scale_mode == 'median_ratio':
                    if len(inliers) > 0:
                        ratios = []
                        for idx in inliers.flatten():
                            j = match_indices_pnp[idx]
                            idx_kf = match_result["indices"][j, 0]
                            map_pt_idx = self._keyframe_kp_to_3d[-2][idx_kf]
                            if map_pt_idx is not None:
                                pt_w = self._map_points_3d[map_pt_idx]
                                pt_c = R_curr @ pt_w + t_curr
                                d_true = pt_c[2]
                                uv = match_result["points1"][j]
                                u_px = np.clip(np.round(uv[0]).astype(np.int32), 0, depth.shape[1] - 1)
                                v_px = np.clip(np.round(uv[1]).astype(np.int32), 0, depth.shape[0] - 1)
                                d_pred = depth[v_px, u_px]
                                if d_pred > 0.01 and d_true > 0.1:
                                    ratios.append(d_true / d_pred)
                        if len(ratios) >= 5:
                            s = np.median(ratios)
                            logger.info(f"Frame {self._frame_idx}: Calibrated monocular depth scale = {s:.4f}")
                
                elif self.config.depth_scale_mode == 'calibrate':
                    if not self._calib_finished:
                        # Collect ratios using ground plane or median ratio to calibrate
                        K_inv = np.linalg.inv(self.camera.K)
                        kps = feats["keypoints"]
                        N_kps = len(kps)
                        s_frame = None
                        if N_kps > 0:
                            u_kps = kps[:, 0]
                            v_kps = kps[:, 1]
                            u_idx_kps = np.clip(np.round(u_kps).astype(np.int32), 0, depth.shape[1] - 1)
                            v_idx_kps = np.clip(np.round(v_kps).astype(np.int32), 0, depth.shape[0] - 1)
                            d_kps = depth[v_idx_kps, u_idx_kps]
                            
                            valid_mask = (d_kps > 0.1) & (d_kps < 150.0)
                            kps_hom = np.hstack([kps, np.ones((N_kps, 1))])
                            p_c = (K_inv @ kps_hom.T).T
                            pts_3d_c = p_c * d_kps.reshape(-1, 1)
                            
                            # Fit ground plane
                            candidates = pts_3d_c[valid_mask]
                            candidates_mask = (kps[valid_mask, 1] > image.shape[0] * 0.5) & (candidates[:, 1] > 0.1) & (candidates[:, 2] < 40.0)
                            plane_pts = candidates[candidates_mask]
                            
                            plane_fit = fit_ground_plane_ransac(plane_pts, camera_height_prior=self._camera_height)
                            if plane_fit is not None:
                                _, d_plane, _ = plane_fit
                                s_frame = self._camera_height / abs(d_plane)
                                self._calib_ratios.append(s_frame)
                                logger.info(f"Frame {self._frame_idx} (Calibrating): Ground plane scale = {s_frame:.4f} (height = {abs(d_plane):.2f}m)")
                            else:
                                # Fallback to median ratio of matched features
                                if len(inliers) >= 5:
                                    ratios = []
                                    for idx in inliers.flatten():
                                        j = match_indices_pnp[idx]
                                        idx_kf = match_result["indices"][j, 0]
                                        map_pt_idx = self._keyframe_kp_to_3d[-2][idx_kf]
                                        if map_pt_idx is not None:
                                            pt_w = self._map_points_3d[map_pt_idx]
                                            pt_c = R_curr @ pt_w + t_curr
                                            d_true = pt_c[2]
                                            uv = match_result["points1"][j]
                                            u_px = np.clip(np.round(uv[0]).astype(np.int32), 0, depth.shape[1] - 1)
                                            v_px = np.clip(np.round(uv[1]).astype(np.int32), 0, depth.shape[0] - 1)
                                            d_pred = depth[v_px, u_px]
                                            if d_pred > 0.01 and d_true > 0.1:
                                                ratios.append(d_true / d_pred)
                                    if len(ratios) >= 5:
                                        s_frame = np.median(ratios)
                                        self._calib_ratios.append(s_frame)
                                        logger.info(f"Frame {self._frame_idx} (Calibrating): Tracker median ratio = {s_frame:.4f}")
                        
                        if len(self._calib_ratios) >= self._calib_frames or self._frame_idx >= self._calib_frames:
                            if len(self._calib_ratios) > 0:
                                self._calib_scale_factor = np.median(self._calib_ratios)
                            else:
                                self._calib_scale_factor = self.config.depth_scale_factor
                            self._calib_finished = True
                            logger.info(f"=== SCALE CALIBRATION FINISHED ===")
                            logger.info(f"Locked Scale Factor = {self._calib_scale_factor:.4f} (from {len(self._calib_ratios)} samples)")
                            logger.info(f"===================================")
                        
                        # During calibration, use the latest valid calibration sample or default
                        if len(self._calib_ratios) > 0:
                            s = self._calib_ratios[-1]
                        else:
                            s = self.config.depth_scale_factor
                    else:
                        s = self._calib_scale_factor
                
                elif self.config.depth_scale_mode == 'ground_plane':
                    K_inv = np.linalg.inv(self.camera.K)
                    kps = feats["keypoints"]
                    N_kps = len(kps)
                    s_fit = None
                    if N_kps > 0:
                        u_kps = kps[:, 0]
                        v_kps = kps[:, 1]
                        u_idx_kps = np.clip(np.round(u_kps).astype(np.int32), 0, depth.shape[1] - 1)
                        v_idx_kps = np.clip(np.round(v_kps).astype(np.int32), 0, depth.shape[0] - 1)
                        d_kps = depth[v_idx_kps, u_idx_kps]
                        
                        valid_mask = (d_kps > 0.1) & (d_kps < 150.0)
                        kps_hom = np.hstack([kps, np.ones((N_kps, 1))])
                        p_c = (K_inv @ kps_hom.T).T
                        pts_3d_c = p_c * d_kps.reshape(-1, 1)
                        
                        candidates = pts_3d_c[valid_mask]
                        candidates_mask = (kps[valid_mask, 1] > image.shape[0] * 0.5) & (candidates[:, 1] > 0.1) & (candidates[:, 2] < 40.0)
                        plane_pts = candidates[candidates_mask]
                        
                        plane_fit = fit_ground_plane_ransac(plane_pts, camera_height_prior=self._camera_height)
                        if plane_fit is not None:
                            _, d_plane, _ = plane_fit
                            s_fit = self._camera_height / abs(d_plane)
                            logger.info(f"Frame {self._frame_idx}: Ground plane scale factor = {s_fit:.4f} (height = {abs(d_plane):.2f}m)")
                    
                    if s_fit is not None:
                        s = s_fit
                    else:
                        s = self.config.depth_scale_factor
                
                elif self.config.depth_scale_mode == 'sliding_window':
                    s_frame = self.config.depth_scale_factor
                    if len(inliers) >= 5:
                        ratios = []
                        for idx in inliers.flatten():
                            j = match_indices_pnp[idx]
                            idx_kf = match_result["indices"][j, 0]
                            map_pt_idx = self._keyframe_kp_to_3d[-2][idx_kf]
                            if map_pt_idx is not None:
                                pt_w = self._map_points_3d[map_pt_idx]
                                pt_c = R_curr @ pt_w + t_curr
                                d_true = pt_c[2]
                                uv = match_result["points1"][j]
                                u_px = np.clip(np.round(uv[0]).astype(np.int32), 0, depth.shape[1] - 1)
                                v_px = np.clip(np.round(uv[1]).astype(np.int32), 0, depth.shape[0] - 1)
                                d_pred = depth[v_px, u_px]
                                if d_pred > 0.01 and d_true > 0.1:
                                    ratios.append(d_true / d_pred)
                        if len(ratios) >= 5:
                            s_frame = np.median(ratios)
                            
                    self._sliding_window_scales.append(s_frame)
                    if len(self._sliding_window_scales) > self._sliding_window_size:
                        self._sliding_window_scales.pop(0)
                    
                    # Compute rolling median scale
                    s_sliding = np.median(self._sliding_window_scales)
                    
                    # Regularize rolling scale towards global prior scale (blend 80/20)
                    s = 0.8 * s_sliding + 0.2 * self.config.depth_scale_factor
                    logger.info(f"Frame {self._frame_idx}: Sliding-window scale = {s:.4f} (raw_frame = {s_frame:.4f})")
                
                else:
                    s = self.config.depth_scale_factor
                
                depth = depth * s
            else:
                raise ValueError(f"Unknown depth_source: {self.config.depth_source}")

            if depth.shape[:2] != image.shape[:2]:
                depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            kps = feats["keypoints"]
            N = len(kps)
            if N > 0:
                K_inv = np.linalg.inv(self.camera.K)

                u = kps[:, 0]
                v = kps[:, 1]
                u_idx = np.clip(np.round(u).astype(np.int32), 0, depth.shape[1] - 1)
                v_idx = np.clip(np.round(v).astype(np.int32), 0, depth.shape[0] - 1)
                d = depth[v_idx, u_idx]

                valid_depth_mask = (d > 0.1) & (d < 150.0) & (~np.isnan(d)) & (~np.isinf(d))
                kps_hom = np.hstack([kps, np.ones((N, 1))])
                p_c = (K_inv @ kps_hom.T).T
                pts_3d_c = p_c * d.reshape(-1, 1)

                pts_3d_w = (R_curr.T @ pts_3d_c.T).T - (R_curr.T @ t_curr).reshape(1, 3)

                new_pts_list = []
                for idx in range(N):
                    if self._keyframe_kp_to_3d[-1][idx] is None and valid_depth_mask[idx]:
                        new_pts_list.append(pts_3d_w[idx])
                        self._keyframe_kp_to_3d[-1][idx] = len(self._map_points_3d) + len(new_pts_list) - 1

                if len(new_pts_list) > 0:
                    self._map_points_3d = np.vstack([self._map_points_3d, np.array(new_pts_list)])

            if self.config.use_joint_ba:
                W = min(self.config.ba_window_size, len(self._keyframe_poses))
                window_poses = self._keyframe_poses[-W:]
                
                pt_observation_counts = {}
                for global_kf_idx in range(len(self._keyframe_poses) - W, len(self._keyframe_poses)):
                    for pt_idx in self._keyframe_kp_to_3d[global_kf_idx]:
                        if pt_idx is not None:
                            pt_observation_counts[pt_idx] = pt_observation_counts.get(pt_idx, 0) + 1
                
                sorted_pts = sorted(pt_observation_counts.keys(), key=lambda x: pt_observation_counts[x], reverse=True)
                observed_global_pts = sorted_pts[:100]
                
                if len(observed_global_pts) > 0:
                    global_to_local_pt = {pt_idx: i for i, pt_idx in enumerate(observed_global_pts)}
                    window_pts_3d = self._map_points_3d[observed_global_pts]
                    
                    observations = []
                    for local_c_idx, global_kf_idx in enumerate(range(len(self._keyframe_poses) - W, len(self._keyframe_poses))):
                        kf_feats = self._keyframe_feats[global_kf_idx]
                        for kp_idx, pt_idx in enumerate(self._keyframe_kp_to_3d[global_kf_idx]):
                            if pt_idx is not None and pt_idx in global_to_local_pt:
                                observations.append({
                                    "cam_idx": local_c_idx,
                                    "pt_idx": global_to_local_pt[pt_idx],
                                    "uv": kf_feats["keypoints"][kp_idx]
                                })
                                
                    opt_poses, opt_pts_3d = self.local_ba.optimize(
                        window_poses, window_pts_3d, observations, self.camera.K, fix_first_two=True
                    )
                    
                    for local_c_idx, global_kf_idx in enumerate(range(len(self._keyframe_poses) - W, len(self._keyframe_poses))):
                        self._keyframe_poses[global_kf_idx] = opt_poses[local_c_idx]
                        
                    self._map_points_3d[observed_global_pts] = opt_pts_3d
                    
                    # Compute optimized relative pose relative to the optimized previous keyframe
                    if len(self._keyframe_poses) >= 2:
                        T_kf_w = self._keyframe_poses[-2]
                        R_kf = T_kf_w[:3, :3]
                        t_kf = T_kf_w[:3, 3]
                        T_kf_w_inv = np.eye(4)
                        T_kf_w_inv[:3, :3] = R_kf.T
                        T_kf_w_inv[:3, 3] = -R_kf.T @ t_kf

                        T_curr_w = self._keyframe_poses[-1]
                        T_curr_kf = T_curr_w @ T_kf_w_inv
                        R_rel = T_curr_kf[:3, :3]
                        t_rel = T_curr_kf[:3, 3]
                        
                        # Sync trajectory accumulator's keyframe base to the optimized previous keyframe
                        self.trajectory._kf_R = T_kf_w[:3, :3].copy()
                        self.trajectory._kf_t = T_kf_w[:3, 3].copy()

            self._prev_feats = feats
            self._stats["keyframes"] += 1

        self.trajectory.add_pose_metric(R_rel, t_rel, is_keyframe=is_kf)
        self.motion_model.update(R_rel, t_rel)

        self._frame_idx += 1
        self._stats["successful"] += 1
        self._per_frame_stats.append({
            "frame_idx": self._frame_idx - 1,
            "num_matches": n_matches,
            "num_inliers": len(inliers),
            "tracking_lost": False,
            "pose_failed": False,
            "is_keyframe": is_kf,
            "motion_model_fallback": False,
        })
        self._timings["total"] += time.perf_counter() - t_start
        pose_3x4 = np.hstack([R_rel, t_rel.reshape(3, 1)])
        return pose_3x4

    def _process_frame_joint_ba(self, image: np.ndarray) -> np.ndarray | None:
        t_start = time.perf_counter()
        self._stats["total"] += 1
        
        t_extract_start = time.perf_counter()
        feats = self.extractor.extract(image)
        self._timings["extraction"] += time.perf_counter() - t_extract_start

        # Initialization: Frame 0 (first keyframe)
        if len(self._keyframe_poses) == 0:
            self._keyframe_poses.append(np.eye(4))
            self._keyframe_feats.append(feats)
            self._keyframe_kp_to_3d.append([None] * len(feats["keypoints"]))
            self._prev_feats = feats
            self._frame_idx += 1
            self._stats["keyframes"] += 1
            self._per_frame_stats.append({
                "frame_idx": self._frame_idx - 1,
                "num_matches": 0,
                "num_inliers": 0,
                "tracking_lost": False,
                "pose_failed": False,
                "is_keyframe": True,
                "motion_model_fallback": False,
            })
            self._timings["total"] += time.perf_counter() - t_start
            return None

        # Initialization: Frame 1 (second keyframe)
        if len(self._keyframe_poses) == 1:
            t_match_start = time.perf_counter()
            match_result = self.matcher.match(
                self._keyframe_feats[0], feats, image_size=image.shape[:2]
            )
            self._timings["matching"] += time.perf_counter() - t_match_start
            n_matches = len(match_result["points0"])
            
            if n_matches < self.min_matches:
                # Initialization failed, wait for next frame
                self._timings["total"] += time.perf_counter() - t_start
                return None
                
            t_pose_start = time.perf_counter()
            pose_fn = estimate_essential_or_homography if self._handle_pure_rotation else estimate_essential
            result = pose_fn(
                match_result["points0"],
                match_result["points1"],
                self.camera.K,
                ransac_thresh=self._ransac_threshold,
                conf=self._ransac_confidence,
            )
            self._timings["pose_estimation"] += time.perf_counter() - t_pose_start
            
            if result is None:
                self._timings["total"] += time.perf_counter() - t_start
                return None
                
            R, t, inlier_mask = result
            t_scaled = t * self.trajectory.scale
            
            T1 = np.eye(4)
            T1[:3, :3] = R
            T1[:3, 3] = t_scaled
            
            self._keyframe_poses.append(T1)
            self._keyframe_feats.append(feats)
            self._keyframe_kp_to_3d.append([None] * len(feats["keypoints"]))
            
            # Triangulate initial points
            fx, fy = self.camera.K[0, 0], self.camera.K[1, 1]
            cx, cy = self.camera.K[0, 2], self.camera.K[1, 2]
            kpn0 = (match_result["points0"] - np.array([cx, cy])) / np.array([fx, fy])
            kpn1 = (match_result["points1"] - np.array([cx, cy])) / np.array([fx, fy])
            
            P0 = np.eye(4)[:3, :]
            P1 = T1[:3, :]
            pts_3d = triangulate_points(P0, P1, kpn0[inlier_mask], kpn1[inlier_mask])
            
            # Add to map
            M = len(pts_3d)
            if M > 0:
                self._map_points_3d = np.vstack([self._map_points_3d, pts_3d])
                
                # Update keypoint associations
                inlier_indices = np.where(inlier_mask)[0]
                for inlier_idx, j in enumerate(inlier_indices):
                    idx0 = match_result["indices"][j, 0]
                    idx1 = match_result["indices"][j, 1]
                    pt_idx = len(self._map_points_3d) - M + inlier_idx
                    self._keyframe_kp_to_3d[0][idx0] = pt_idx
                    self._keyframe_kp_to_3d[1][idx1] = pt_idx
                    
            self.trajectory.add_pose_relative_to_keyframe(R, t_scaled, is_keyframe=True)
            self.motion_model.update(R, t_scaled)
            self._prev_feats = feats
            self._frame_idx += 1
            self._stats["successful"] += 1
            self._stats["keyframes"] += 1
            self._per_frame_stats.append({
                "frame_idx": self._frame_idx - 1,
                "num_matches": n_matches,
                "num_inliers": int(inlier_mask.sum()),
                "tracking_lost": False,
                "pose_failed": False,
                "is_keyframe": True,
                "motion_model_fallback": False,
            })
            self._timings["total"] += time.perf_counter() - t_start
            pose_3x4 = np.hstack([R, t_scaled.reshape(3, 1)])
            return pose_3x4

        # Frame 2+: 3D-2D tracking relative to previous keyframe
        feats_kf = self._keyframe_feats[-1]
        t_match_start = time.perf_counter()
        match_result = self.matcher.match(
            feats_kf, feats, image_size=image.shape[:2]
        )
        self._timings["matching"] += time.perf_counter() - t_match_start
        n_matches = len(match_result["points0"])
        
        # Gather 3D-2D correspondences
        points_3d_pnp = []
        points_2d_pnp = []
        match_indices_pnp = []
        
        for j in range(n_matches):
            idx_kf = match_result["indices"][j, 0]
            pt_idx = self._keyframe_kp_to_3d[-1][idx_kf]
            if pt_idx is not None:
                points_3d_pnp.append(self._map_points_3d[pt_idx])
                points_2d_pnp.append(match_result["points1"][j])
                match_indices_pnp.append(j)
                
        n_pnp = len(points_3d_pnp)
        
        # Check tracking status
        if n_pnp < self.config.min_inliers_pnp:
            self._stats["tracking_lost"] += 1
            logger.warning(f"Frame {self._frame_idx}: 3D-2D tracking lost ({n_pnp} < {self.config.min_inliers_pnp})")
            
            # Motion model fallback
            R_pred, t_pred = self.motion_model.predict()
            self.trajectory.add_pose_relative_to_keyframe(R_pred, t_pred, is_keyframe=False)
            self._stats["motion_model_fallbacks"] += 1
            self._per_frame_stats.append({
                "frame_idx": self._frame_idx,
                "num_matches": n_matches,
                "num_inliers": 0,
                "tracking_lost": True,
                "pose_failed": False,
                "is_keyframe": False,
                "motion_model_fallback": True,
            })
            self._frame_idx += 1
            self._timings["total"] += time.perf_counter() - t_start
            pose_3x4 = np.hstack([R_pred, t_pred.reshape(3, 1)])
            return pose_3x4

        # Run PnP with initial guess from motion model
        t_pose_start = time.perf_counter()
        T_kf_w = self._keyframe_poses[-1]
        R_pred, t_pred = self.motion_model.predict()
        T_curr_kf = np.eye(4)
        T_curr_kf[:3, :3] = R_pred
        T_curr_kf[:3, 3] = t_pred
        T_pred_w = T_curr_kf @ T_kf_w
        
        rvec_pred, _ = cv2.Rodrigues(T_pred_w[:3, :3])
        tvec_pred = T_pred_w[:3, 3]
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(points_3d_pnp, dtype=np.float64),
            np.array(points_2d_pnp, dtype=np.float64),
            self.camera.K,
            distCoeffs=None,
            rvec=rvec_pred.copy(),
            tvec=tvec_pred.copy(),
            useExtrinsicGuess=True,
            iterationsCount=150,
            reprojectionError=2.0,
            confidence=0.99
        )
        self._timings["pose_estimation"] += time.perf_counter() - t_pose_start

        if not success or inliers is None or len(inliers) < self.config.min_inliers_pnp:
            self._stats["pose_failed"] += 1
            logger.warning(f"Frame {self._frame_idx}: PnP tracking failed")
            
            # Motion model fallback
            R_pred, t_pred = self.motion_model.predict()
            self.trajectory.add_pose_relative_to_keyframe(R_pred, t_pred, is_keyframe=False)
            self._stats["motion_model_fallbacks"] += 1
            self._per_frame_stats.append({
                "frame_idx": self._frame_idx,
                "num_matches": n_matches,
                "num_inliers": 0,
                "tracking_lost": False,
                "pose_failed": True,
                "is_keyframe": False,
                "motion_model_fallback": True,
            })
            self._frame_idx += 1
            self._timings["total"] += time.perf_counter() - t_start
            pose_3x4 = np.hstack([R_pred, t_pred.reshape(3, 1)])
            return pose_3x4

        # PnP tracking success: compute pose
        t_curr = tvec.flatten()
        R_curr, _ = cv2.Rodrigues(rvec)
        T_curr_w = np.eye(4)
        T_curr_w[:3, :3] = R_curr
        T_curr_w[:3, 3] = t_curr
        
        # Relative pose with respect to last keyframe
        T_curr_kf = T_curr_w @ np.linalg.inv(T_kf_w)
        R_rel = T_curr_kf[:3, :3]
        t_rel = T_curr_kf[:3, 3]
        
        # Decide if new keyframe
        is_kf = False
        if self._use_keyframe_selection:
            is_kf = self.keyframe_selector.should_insert(
                match_result["points0"],
                match_result["points1"],
                len(feats["keypoints"]),
            )

        if is_kf:
            # Add to keyframe history
            self._keyframe_poses.append(T_curr_w)
            self._keyframe_feats.append(feats)
            self._keyframe_kp_to_3d.append([None] * len(feats["keypoints"]))
            
            # 1. Propagate 3D points
            for idx in inliers.flatten():
                j = match_indices_pnp[idx]
                idx_prev = match_result["indices"][j, 0]
                idx_curr = match_result["indices"][j, 1]
                pt_idx = self._keyframe_kp_to_3d[-2][idx_prev]
                self._keyframe_kp_to_3d[-1][idx_curr] = pt_idx

            # 2. Triangulate new 3D points between Keyframe k and Keyframe k+1
            fx, fy = self.camera.K[0, 0], self.camera.K[1, 1]
            cx, cy = self.camera.K[0, 2], self.camera.K[1, 2]
            kpn_prev = (match_result["points0"] - np.array([cx, cy])) / np.array([fx, fy])
            kpn_curr = (match_result["points1"] - np.array([cx, cy])) / np.array([fx, fy])
            
            P_prev = T_kf_w[:3, :]
            P_curr = T_curr_w[:3, :]
            
            new_pts_3d = triangulate_points(P_prev, P_curr, kpn_prev, kpn_curr)
            
            # Filter triangulated points and add valid ones to the map
            for j in range(len(new_pts_3d)):
                idx_prev = match_result["indices"][j, 0]
                idx_curr = match_result["indices"][j, 1]
                
                # Check if it already has a 3D point
                if self._keyframe_kp_to_3d[-1][idx_curr] is not None:
                    continue
                    
                pt_w = new_pts_3d[j]
                # Check positive depth in both cameras
                pt_c_prev = P_prev[:, :3] @ pt_w + P_prev[:, 3]
                pt_c_curr = P_curr[:, :3] @ pt_w + P_curr[:, 3]
                if pt_c_prev[2] < 0.2 or pt_c_curr[2] < 0.2:
                    continue
                    
                # Check reprojection error
                uv_proj_prev, _ = cv2.projectPoints(pt_w.reshape(1, 3), P_prev[:, :3], P_prev[:, 3], self.camera.K, None)
                uv_proj_curr, _ = cv2.projectPoints(pt_w.reshape(1, 3), P_curr[:, :3], P_curr[:, 3], self.camera.K, None)
                
                err_prev = np.linalg.norm(match_result["points0"][j] - uv_proj_prev.ravel())
                err_curr = np.linalg.norm(match_result["points1"][j] - uv_proj_curr.ravel())
                
                if err_prev < 3.0 and err_curr < 3.0:
                    self._map_points_3d = np.vstack([self._map_points_3d, pt_w])
                    pt_idx = len(self._map_points_3d) - 1
                    self._keyframe_kp_to_3d[-2][idx_prev] = pt_idx
                    self._keyframe_kp_to_3d[-1][idx_curr] = pt_idx
                    
            # 3. Sliding Window Local Joint BA
            W = min(self.config.ba_window_size, len(self._keyframe_poses))
            window_poses = self._keyframe_poses[-W:]
            
            # Map observed global 3D point indices to local indices, limiting to top 100 most observed points
            pt_observation_counts = {}
            for global_kf_idx in range(len(self._keyframe_poses) - W, len(self._keyframe_poses)):
                for pt_idx in self._keyframe_kp_to_3d[global_kf_idx]:
                    if pt_idx is not None:
                        pt_observation_counts[pt_idx] = pt_observation_counts.get(pt_idx, 0) + 1
            
            # Sort points by observation counts (descending) and keep top 100
            sorted_pts = sorted(pt_observation_counts.keys(), key=lambda x: pt_observation_counts[x], reverse=True)
            observed_global_pts = sorted_pts[:100]
            
            if len(observed_global_pts) > 0:
                global_to_local_pt = {pt_idx: i for i, pt_idx in enumerate(observed_global_pts)}
                window_pts_3d = self._map_points_3d[observed_global_pts]
                
                # Build observations
                observations = []
                for local_c_idx, global_kf_idx in enumerate(range(len(self._keyframe_poses) - W, len(self._keyframe_poses))):
                    kf_feats = self._keyframe_feats[global_kf_idx]
                    for kp_idx, pt_idx in enumerate(self._keyframe_kp_to_3d[global_kf_idx]):
                        if pt_idx is not None and pt_idx in global_to_local_pt:
                            observations.append({
                                "cam_idx": local_c_idx,
                                "pt_idx": global_to_local_pt[pt_idx],
                                "uv": kf_feats["keypoints"][kp_idx]
                            })
                            
                # Run BA optimizer
                opt_poses, opt_pts_3d = self.local_ba.optimize(
                    window_poses, window_pts_3d, observations, self.camera.K, fix_first_two=True
                )
                
                # Write back optimized poses
                for local_c_idx, global_kf_idx in enumerate(range(len(self._keyframe_poses) - W, len(self._keyframe_poses))):
                    self._keyframe_poses[global_kf_idx] = opt_poses[local_c_idx]
                    
                # Write back optimized 3D points
                self._map_points_3d[observed_global_pts] = opt_pts_3d
                
                # Update trajectory accumulator with optimized current keyframe pose
                T_curr_w = self._keyframe_poses[-1]
                self.trajectory._kf_R = T_curr_w[:3, :3].copy()
                self.trajectory._kf_t = T_curr_w[:3, 3].copy()
                self.trajectory._current_R = T_curr_w[:3, :3].copy()
                self.trajectory._current_t = T_curr_w[:3, 3].copy()
            
            self._prev_feats = feats
            self._stats["keyframes"] += 1

        self.trajectory.add_pose_relative_to_keyframe(R_rel, t_rel, is_keyframe=is_kf)
        self.motion_model.update(R_rel, t_rel)
        
        self._frame_idx += 1
        self._stats["successful"] += 1
        self._per_frame_stats.append({
            "frame_idx": self._frame_idx - 1,
            "num_matches": n_matches,
            "num_inliers": len(inliers),
            "tracking_lost": False,
            "pose_failed": False,
            "is_keyframe": is_kf,
            "motion_model_fallback": False,
        })
        self._timings["total"] += time.perf_counter() - t_start
        pose_3x4 = np.hstack([R_rel, t_rel.reshape(3, 1)])
        return pose_3x4

    def process_sequence(self, images: Iterator[np.ndarray]) -> list[np.ndarray]:
        """Process a sequence of images, return list of 4x4 poses."""
        poses = []
        for img in images:
            pose = self.process_frame(img)
            if pose is not None:
                T = np.eye(4)
                T[:3, :] = pose
                poses.append(T)
        return poses

    def get_trajectory(self) -> TrajectoryAccumulator:
        """Return the trajectory accumulator."""
        return self.trajectory

    def get_stats(self) -> dict:
        """Return statistics dict."""
        return self._stats.copy()

    def get_timings(self) -> dict:
        """Return accumulated pipeline timings (in seconds) per stage."""
        return self._timings.copy()

    def get_per_frame_stats(self) -> list[dict]:
        """Return copy of per-frame statistics list."""
        return [s.copy() for s in self._per_frame_stats]

    def reset(self) -> None:
        """Reset internal state (trajectory + frame counter)."""
        self.trajectory.reset()
        self._prev_feats = None
        self._frame_idx = 0
        self.keyframe_selector.reset()
        self.motion_model.reset()
        self._keyframe_poses = []
        self._keyframe_feats = []
        self._keyframe_kp_to_3d = []
        self._map_points_3d = np.zeros((0, 3))
        self._per_frame_stats = []
        self._stats = {
            "total": 0,
            "successful": 0,
            "tracking_lost": 0,
            "pose_failed": 0,
            "keyframes": 0,
            "motion_model_fallbacks": 0,
        }
        self._timings = {
            "extraction": 0.0,
            "matching": 0.0,
            "pose_estimation": 0.0,
            "total": 0.0,
        }


def fit_ground_plane_ransac(
    pts_3d: np.ndarray,
    camera_height_prior: float = 1.65,
    max_iterations: int = 150,
    inlier_threshold: float = 0.05,
    angle_threshold_deg: float = 20.0,
) -> tuple[np.ndarray, float, int] | None:
    """Fit a ground plane to 3D points in camera coordinates.

    Camera coordinate system: X right, Y down, Z forward.
    The ground plane normal vector should be close to [0, -1, 0] or [0, 1, 0].
    We normalize the normal to point upwards (Y is negative, i.e., normal is [0, -1, 0] approx).
    Since Y is down, a point on the ground has Y > 0.
    Let plane equation be: n_x * x + n_y * y + n_z * z + d = 0.
    Since ground is below the camera, y is positive, so if n_y is positive (e.g. ~1.0),
    then d must be negative (e.g., -camera_height).
    Thus, camera height is |d|.

    Returns:
        tuple (normal_vector, plane_distance, num_inliers) or None if fit fails.
    """
    n_pts = len(pts_3d)
    if n_pts < 3:
        return None

    best_inliers = []
    best_plane = None

    # Angle constraint: normal must be close to vertical [0, 1, 0] in camera frame
    min_ny = np.cos(np.radians(angle_threshold_deg))

    # Run RANSAC (deterministic RNG for reproducibility)
    rng = np.random.default_rng(42)

    for _ in range(max_iterations):
        # Sample 3 random points
        idx = rng.choice(n_pts, size=3, replace=False)
        p1, p2, p3 = pts_3d[idx]

        # Calculate normal vector
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-6:
            continue
        normal = normal / norm_val

        # Enforce normal orientation: Y axis should be dominant
        if abs(normal[1]) < min_ny:
            continue

        # Plane distance parameter
        d = -np.dot(normal, p1)

        # Skip degenerate planes
        if abs(d) < 0.1 or abs(d) > 10.0 * camera_height_prior:
            continue

        # Compute inliers
        distances = np.abs(np.dot(pts_3d, normal) + d)
        inliers = np.where(distances < inlier_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    if best_plane is None or len(best_inliers) < 10:
        return None

    # Refit plane using all inliers (Least Squares)
    inlier_pts = pts_3d[best_inliers]
    centroid = np.mean(inlier_pts, axis=0)
    pts_centered = inlier_pts - centroid
    cov = pts_centered.T @ pts_centered
    
    # SVD of covariance matrix
    _, _, Vt = np.linalg.svd(cov)
    # The normal is the last row of Vt (smallest singular value)
    normal_refined = Vt[-1]
    norm_val = np.linalg.norm(normal_refined)
    if norm_val < 1e-6:
        return best_plane[0], best_plane[1], len(best_inliers)
    normal_refined = normal_refined / norm_val

    # Ensure consistent sign (Y component positive)
    if normal_refined[1] < 0:
        normal_refined = -normal_refined

    d_refined = -np.dot(normal_refined, centroid)

    # Check normal orientation again after refinement
    if abs(normal_refined[1]) < min_ny:
        return best_plane[0], best_plane[1], len(best_inliers)

    return normal_refined, d_refined, len(best_inliers)

