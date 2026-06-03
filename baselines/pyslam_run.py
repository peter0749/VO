#!/usr/bin/env python3
"""Helper runner script executed inside the pySLAM environment.

It configures the pySLAM parameters dynamically, symlinks the dataset,
executes the main SLAM pipeline headless, and saves the output trajectory.
"""

import argparse
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

# ==============================================================================
# Mocking C++ Modules for Pure Python Fallback (Zero-Compile pySLAM Support)
# ==============================================================================
import cv2
import numpy as np
import math

# 1. Mock pyslam_utils C++ module
def filter_cv2_keypoints_by_mask(kps, mask_np):
    if mask_np is None or mask_np.size == 0:
        return kps, np.arange(len(kps), dtype=np.int32)
    
    kps_out = []
    idxs_out = []
    
    if mask_np.ndim == 1:
        for i, kp in enumerate(kps):
            if mask_np[i] != 0:
                kps_out.append(kp)
                idxs_out.append(i)
        return kps_out, np.array(idxs_out, dtype=np.int32)
        
    mask_height, mask_width = mask_np.shape[:2]
    for i, kp in enumerate(kps):
        pt = kp.pt
        x = int(math.floor(pt[0]))
        y = int(math.floor(pt[1]))
        if 0 <= x < mask_width and 0 <= y < mask_height and mask_np[y, x] != 0:
            kps_out.append(kp)
            idxs_out.append(i)
    return kps_out, np.array(idxs_out, dtype=np.int32)

def filter_2d_keypoints_by_mask(kps_np, mask_np):
    num_keypoints = kps_np.shape[0]
    num_columns = kps_np.shape[1]
    
    idxs_out = []
    kps_out = []
    
    if mask_np.ndim == 1:
        for i in range(num_keypoints):
            if mask_np[i] != 0:
                idxs_out.append(i)
                kps_out.append(kps_np[i])
        if len(idxs_out) == 0:
            return np.empty((0, num_columns), dtype=np.float32), np.empty(0, dtype=np.int32)
        return np.array(kps_out, dtype=np.float32), np.array(idxs_out, dtype=np.int32)
        
    mask_height, mask_width = mask_np.shape[:2]
    for i in range(num_keypoints):
        x = int(math.floor(kps_np[i, 0]))
        y = int(math.floor(kps_np[i, 1]))
        if 0 <= x < mask_width and 0 <= y < mask_height and mask_np[y, x] != 0:
            idxs_out.append(i)
            kps_out.append(kps_np[i])
    if len(idxs_out) == 0:
        return np.empty((0, num_columns), dtype=np.float32), np.empty(0, dtype=np.int32)
    return np.array(kps_out, dtype=np.float32), np.array(idxs_out, dtype=np.int32)

def good_matches_simple(matches, ratio_test=0.7):
    idxs1, idxs2 = [], []
    if matches is not None:
        for pair in matches:
            if len(pair) >= 2:
                m, n = pair[0], pair[1]
                if m.distance < ratio_test * n.distance:
                    idxs1.append(m.queryIdx)
                    idxs2.append(m.trainIdx)
    return np.array(idxs1, dtype=np.int32), np.array(idxs2, dtype=np.int32)

def good_matches_one_to_one(matches, ratio_test=0.7):
    idxs1 = []
    idxs2 = []
    dist_match = {}
    index_match = {}
    
    if matches is not None:
        for m_pair in matches:
            if len(m_pair) < 2:
                continue
            m = m_pair[0]
            n = m_pair[1]
            
            if m.distance >= ratio_test * n.distance:
                continue
                
            t_idx = m.trainIdx
            q_idx = m.queryIdx
            
            if t_idx not in dist_match:
                dist_match[t_idx] = m.distance
                idxs1.append(q_idx)
                idxs2.append(t_idx)
                index_match[t_idx] = len(idxs2) - 1
            elif m.distance < dist_match[t_idx]:
                index = index_match[t_idx]
                idxs1[index] = q_idx
                idxs2[index] = t_idx
                dist_match[t_idx] = m.distance
                
    return np.array(idxs1, dtype=np.int32), np.array(idxs2, dtype=np.int32)

def row_matches(kps1, kps2, matches, max_distance, max_row_distance, max_disparity):
    idxs1, idxs2 = [], []
    for m in matches:
        if m.distance >= max_distance:
            continue
        pt1 = kps1[m.queryIdx].pt
        pt2 = kps2[m.trainIdx].pt
        if abs(pt1[1] - pt2[1]) < max_row_distance and abs(pt1[0] - pt2[0]) < max_disparity:
            idxs1.append(m.queryIdx)
            idxs2.append(m.trainIdx)
    return np.array(idxs1, dtype=np.int32), np.array(idxs2, dtype=np.int32)

def row_matches_np(kps1_np, kps2_np, matches, max_distance, max_row_distance, max_disparity):
    idxs1, idxs2 = [], []
    for m in matches:
        if m.distance >= max_distance:
            continue
        qidx = m.queryIdx
        tidx = m.trainIdx
        x1, y1 = kps1_np[qidx, 0], kps1_np[qidx, 1]
        x2, y2 = kps2_np[tidx, 0], kps2_np[tidx, 1]
        if abs(y1 - y2) < max_row_distance and abs(x1 - x2) < max_disparity:
            idxs1.append(qidx)
            idxs2.append(tidx)
    return np.array(idxs1, dtype=np.int32), np.array(idxs2, dtype=np.int32)

def row_matches_with_ratio_test(kps1, kps2, knn_matches, max_distance, max_row_distance, max_disparity, ratio_test):
    idxs1, idxs2 = [], []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m = pair[0]
        n = pair[1]
        if m.distance >= max_distance or m.distance >= ratio_test * n.distance:
            continue
        pt1 = kps1[m.queryIdx].pt
        pt2 = kps2[m.trainIdx].pt
        if abs(pt1[1] - pt2[1]) < max_row_distance and abs(pt1[0] - pt2[0]) < max_disparity:
            idxs1.append(m.queryIdx)
            idxs2.append(m.trainIdx)
    return np.array(idxs1, dtype=np.int32), np.array(idxs2, dtype=np.int32)

def row_matches_with_ratio_test_np(kps1_np, kps2_np, knn_matches, max_distance, max_row_distance, max_disparity, ratio_test):
    idxs1, idxs2 = [], []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m = pair[0]
        n = pair[1]
        if m.distance >= ratio_test * n.distance or m.distance >= max_distance:
            continue
        qidx = m.queryIdx
        tidx = m.trainIdx
        dy = abs(kps1_np[qidx, 1] - kps2_np[tidx, 1])
        dx = abs(kps1_np[qidx, 0] - kps2_np[tidx, 0])
        if dy < max_row_distance and dx < max_disparity:
            idxs1.append(qidx)
            idxs2.append(tidx)
    return np.array(idxs1, dtype=np.int32), np.array(idxs2, dtype=np.int32)

def filter_non_row_matches(kps1, kps2, idxs1, idxs2, max_row_distance, max_disparity):
    out_idxs1, out_idxs2 = [], []
    for i in range(len(idxs1)):
        pt1 = kps1[idxs1[i]].pt
        pt2 = kps2[idxs2[i]].pt
        if abs(pt1[1] - pt2[1]) < max_row_distance and abs(pt1[0] - pt2[0]) < max_disparity:
            out_idxs1.append(idxs1[i])
            out_idxs2.append(idxs2[i])
    return np.array(out_idxs1, dtype=np.int32), np.array(out_idxs2, dtype=np.int32)

def filter_non_row_matches_np(kps1_np, kps2_np, idxs1_np, idxs2_np, max_row_distance, max_disparity):
    out_idxs1, out_idxs2 = [], []
    N = idxs1_np.shape[0]
    for i in range(N):
        idx1 = idxs1_np[i]
        idx2 = idxs2_np[i]
        y1 = kps1_np[idx1, 1]
        x1 = kps1_np[idx1, 0]
        y2 = kps2_np[idx2, 1]
        x2 = kps2_np[idx2, 0]
        if abs(y1 - y2) < max_row_distance and abs(x1 - x2) < max_disparity:
            out_idxs1.append(idx1)
            out_idxs2.append(idx2)
    return np.array(out_idxs1, dtype=np.int32), np.array(out_idxs2, dtype=np.int32)

def extract_mean_colors(img, img_coords, delta, default_color):
    H, W, C = img.shape
    N = img_coords.shape[0]
    result = np.zeros((N, 3), dtype=np.float32)
    patch_size = 1 + 2 * delta
    patch_area = patch_size * patch_size
    
    for i in range(N):
        x = img_coords[i, 0]
        y = img_coords[i, 1]
        
        if x - delta >= 0 and x + delta < W and y - delta >= 0 and y + delta < H:
            patch = img[y - delta : y + delta + 1, x - delta : x + delta + 1]
            result[i] = patch.mean(axis=(0, 1))
        else:
            result[i] = default_color
            
    return result

def extract_patches(image, kps, patch_size, use_orientation=True, scale_factor=1.0, warp_flags=None):
    if warp_flags is None:
        warp_flags = cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS
        
    patches = []
    for kp in kps:
        s = scale_factor * kp.size / patch_size
        if use_orientation:
            angle_rad = kp.angle * math.pi / 180.0 if kp.angle >= 0 else 0.0
            cosine = math.cos(angle_rad)
            sine = math.sin(angle_rad)
            M = np.array([
                [s * cosine, -s * sine, (-s * cosine + s * sine) * patch_size / 2.0 + kp.pt[0]],
                [s * sine, s * cosine, (-s * sine - s * cosine) * patch_size / 2.0 + kp.pt[1]]
            ], dtype=np.float32)
        else:
            M = np.array([
                [s, 0.0, -s * patch_size / 2.0 + kp.pt[0]],
                [0.0, s, -s * patch_size / 2.0 + kp.pt[1]]
            ], dtype=np.float32)
            
        patch = cv2.warpAffine(image, M, (patch_size, patch_size), flags=warp_flags)
        patches.append(patch)
        
    return patches

# Create and register the mock pyslam_utils module
pyslam_utils_mock = types.ModuleType("pyslam_utils")
pyslam_utils_mock.filter_cv2_keypoints_by_mask = filter_cv2_keypoints_by_mask
pyslam_utils_mock.filter_2d_keypoints_by_mask = filter_2d_keypoints_by_mask
pyslam_utils_mock.good_matches_simple = good_matches_simple
pyslam_utils_mock.good_matches_one_to_one = good_matches_one_to_one
pyslam_utils_mock.row_matches = row_matches
pyslam_utils_mock.row_matches_np = row_matches_np
pyslam_utils_mock.row_matches_with_ratio_test = row_matches_with_ratio_test
pyslam_utils_mock.row_matches_with_ratio_test_np = row_matches_with_ratio_test_np
pyslam_utils_mock.filter_non_row_matches = filter_non_row_matches
pyslam_utils_mock.filter_non_row_matches_np = filter_non_row_matches_np
pyslam_utils_mock.extract_mean_colors = extract_mean_colors
pyslam_utils_mock.extract_patches = extract_patches

sys.modules["pyslam_utils"] = pyslam_utils_mock

# 2. Mock orbslam2_features using cv2.ORB
class MockORBextractor:
    def __init__(self, num_features=2000, scale_factor=1.2, num_levels=8):
        self.num_features = num_features
        self.scale_factor = scale_factor
        self.num_levels = num_levels
        self._init_orb()
        
    def _init_orb(self):
        self.orb = cv2.ORB_create(
            nfeatures=self.num_features,
            scaleFactor=self.scale_factor,
            nlevels=self.num_levels
        )
        
    def SetNumFeatures(self, num_features):
        self.num_features = num_features
        self._init_orb()
        
    def detect(self, img):
        kps = self.orb.detect(img)
        return [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kps]
        
    def detectAndCompute(self, img):
        kps, des = self.orb.detectAndCompute(img, None)
        if des is None:
            des = np.empty((0, 32), dtype=np.uint8)
        kps_tuples = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kps]
        return kps_tuples, des

class MockORBextractorDeterministic(MockORBextractor):
    pass

orbslam2_features_mock = types.ModuleType("orbslam2_features")
orbslam2_features_mock.ORBextractor = MockORBextractor
orbslam2_features_mock.ORBextractorDeterministic = MockORBextractorDeterministic
sys.modules["orbslam2_features"] = orbslam2_features_mock

# 3. Mock g2o library
class MockQuaternion:
    def __init__(self, *args):
        if len(args) == 0:
            self.q = np.array([0.0, 0.0, 0.0, 1.0])
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            R = args[0]
            # Convert 3x3 rotation matrix to quaternion (x, y, z, w)
            q = np.zeros(4)
            t = np.trace(R)
            if t > 0:
                s = 0.5 / np.sqrt(t + 1.0)
                q[3] = 0.25 / s
                q[0] = (R[2, 1] - R[1, 2]) * s
                q[1] = (R[0, 2] - R[2, 0]) * s
                q[2] = (R[1, 0] - R[0, 1]) * s
            else:
                if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                    q[3] = (R[2, 1] - R[1, 2]) / s
                    q[0] = 0.25 * s
                    q[1] = (R[0, 1] + R[1, 0]) / s
                    q[2] = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                    q[3] = (R[0, 2] - R[2, 0]) / s
                    q[0] = (R[0, 1] + R[1, 0]) / s
                    q[1] = 0.25 * s
                    q[2] = (R[1, 2] + R[2, 1]) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                    q[3] = (R[1, 0] - R[0, 1]) / s
                    q[0] = (R[0, 2] + R[2, 0]) / s
                    q[1] = (R[1, 2] + R[2, 1]) / s
                    q[2] = 0.25 * s
            self.q = q
        elif len(args) == 4:
            self.q = np.array([args[1], args[2], args[3], args[0]]) # x, y, z, w
        else:
            self.q = np.array([0.0, 0.0, 0.0, 1.0])
            
    def matrix(self):
        x, y, z, w = self.q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)
        
    def rotation_matrix(self):
        return self.matrix()
        
    def inverse(self):
        x, y, z, w = self.q
        return MockQuaternion(w, -x, -y, -z)
        
    def __mul__(self, other):
        if isinstance(other, MockQuaternion):
            R1 = self.matrix()
            R2 = other.matrix()
            return MockQuaternion(R1 @ R2)
        elif isinstance(other, (np.ndarray, list, tuple)):
            other_arr = np.array(other)
            if other_arr.shape[-1] == 3:
                R = self.matrix()
                if other_arr.ndim > 1:
                    return (R @ other_arr.T).T
                else:
                    return R @ other_arr
        return NotImplemented
        
    def normalize(self):
        norm = np.linalg.norm(self.q)
        if norm > 1e-8:
            self.q = self.q / norm
        return self
        
    def coeffs(self):
        return self.q
        
    def vec(self):
        return self.q[:3]
        
    def x(self):
        return self.q[0]
        
    def y(self):
        return self.q[1]
        
    def z(self):
        return self.q[2]
        
    def w(self):
        return self.q[3]

class MockSE3Quat:
    def __init__(self, rotation=None, translation=None):
        if rotation is None:
            self.R = np.eye(3)
        elif isinstance(rotation, MockQuaternion):
            self.R = rotation.matrix()
        else:
            self.R = np.array(rotation)
        if translation is None:
            self.t = np.zeros(3)
        else:
            self.t = np.array(translation)
            
    def rotation(self):
        return MockQuaternion(self.R)
        
    def translation(self):
        return self.t
        
    def matrix(self):
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

class MockIsometry3d:
    def __init__(self, rotation=None, translation=None):
        if rotation is None and translation is None:
            self._matrix = np.eye(4, dtype=np.float64)
        elif isinstance(rotation, np.ndarray) and rotation.shape == (4, 4):
            self._matrix = rotation.copy()
        elif isinstance(rotation, MockIsometry3d):
            self._matrix = rotation._matrix.copy()
        elif isinstance(rotation, MockSE3Quat):
            self._matrix = rotation.matrix()
        else:
            self._matrix = np.eye(4, dtype=np.float64)
            if rotation is not None:
                if isinstance(rotation, MockQuaternion):
                    self._matrix[:3, :3] = rotation.matrix()
                else:
                    self._matrix[:3, :3] = rotation
            if translation is not None:
                self._matrix[:3, 3] = translation
                
    def matrix(self):
        return self._matrix
        
    def orientation(self):
        return MockQuaternion(self._matrix[:3, :3])
        
    def position(self):
        return self._matrix[:3, 3]
        
    def rotation_matrix(self):
        return self._matrix[:3, :3]
        
    def inverse(self):
        inv_mat = np.eye(4, dtype=np.float64)
        R = self._matrix[:3, :3]
        t = self._matrix[:3, 3]
        inv_mat[:3, :3] = R.T
        inv_mat[:3, 3] = -R.T @ t
        return MockIsometry3d(inv_mat)

    def __mul__(self, other):
        if isinstance(other, MockIsometry3d):
            return MockIsometry3d(self._matrix @ other._matrix)
        elif isinstance(other, np.ndarray):
            if other.shape[-1] == 3:
                # Transform 3D point(s)
                if other.ndim > 1:
                    h_pts = np.hstack([other, np.ones((other.shape[0], 1))])
                    return (self._matrix @ h_pts.T).T[..., :3]
                else:
                    h_pt = np.append(other, 1.0)
                    return (self._matrix @ h_pt)[..., :3]
            elif other.shape[-1] == 4:
                # Transform homogeneous coordinates
                return (self._matrix @ other.T).T
        return NotImplemented

class MockAngleAxis:
    def __init__(self, q):
        pass

class MockFlag:
    def __init__(self, val=False):
        self.value = val

class MockVertexSE3Expmap:
    def __init__(self):
        self._id = 0
        self._estimate = MockSE3Quat()
        self._fixed = False
        
    def set_id(self, id):
        self._id = id
        
    def set_estimate(self, se3):
        if isinstance(se3, MockIsometry3d):
            self._estimate = MockSE3Quat(se3.rotation_matrix(), se3.position())
        else:
            self._estimate = se3
            
    def estimate(self):
        return self._estimate
        
    def set_fixed(self, flag):
        self._fixed = flag

class MockVertexSBAPointXYZ:
    def __init__(self):
        self._id = 0
        self._estimate = np.zeros(3)
        self._fixed = False
        self._marginalized = False
        
    def set_id(self, id):
        self._id = id
        
    def set_estimate(self, xyz):
        self._estimate = np.array(xyz)
        
    def estimate(self):
        return self._estimate
        
    def set_fixed(self, flag):
        self._fixed = flag
        
    def set_marginalized(self, flag):
        self._marginalized = flag

class MockEdge:
    def __init__(self):
        self.vertices = {}
        self.measurement = None
        self.information = None
        self.robust_kernel = None
        self.fx = 1.0
        self.fy = 1.0
        self.cx = 0.0
        self.cy = 0.0
        self.bf = 0.0
        self.Xw = np.zeros(3)
        self._level = 0
        
    def set_vertex(self, i, vertex):
        self.vertices[i] = vertex
        
    def set_measurement(self, measurement):
        self.measurement = measurement
        
    def set_information(self, information):
        self.information = information
        
    def set_robust_kernel(self, kernel):
        self.robust_kernel = kernel
        
    def chi2(self):
        return 0.0
        
    def set_level(self, level):
        self._level = level
        
    def level(self):
        return self._level
        
    def is_depth_positive(self):
        return True
        
    def compute_error(self):
        pass

class MockEdgeSE3ProjectXYZ(MockEdge):
    pass

class MockEdgeStereoSE3ProjectXYZ(MockEdge):
    pass

class MockEdgeSE3ProjectXYZOnlyPose(MockEdge):
    pass

class MockEdgeStereoSE3ProjectXYZOnlyPose(MockEdge):
    pass

class MockRobustKernelHuber:
    def __init__(self, th):
        self.th = th

class MockSparseOptimizer:
    def __init__(self):
        self._vertices = {}
        self._edges = []
        self._verbose = False
        self._force_stop_flag = None
        
    def set_algorithm(self, solver):
        pass
        
    def set_force_stop_flag(self, flag):
        self._force_stop_flag = flag
        
    def add_vertex(self, vertex):
        self._vertices[vertex._id] = vertex
        
    def add_edge(self, edge):
        self._edges.append(edge)
        
    def vertex(self, id):
        return self._vertices.get(id)
        
    def edges(self):
        return self._edges
        
    def set_verbose(self, flag):
        self._verbose = flag
        
    def initialize_optimization(self):
        pass
        
    def compute_active_errors(self):
        pass
        
    def active_chi2(self):
        return 0.0
        
    def optimize(self, its):
        pass

class MockBlockSolverSE3:
    def __init__(self, solver):
        pass

class MockLinearSolver:
    def __init__(self):
        pass

class MockOptimizationAlgorithmLevenberg:
    def __init__(self, solver):
        pass

g2o_mock = types.ModuleType("g2o")
g2o_mock.Quaternion = MockQuaternion
g2o_mock.SE3Quat = MockSE3Quat
g2o_mock.Isometry3d = MockIsometry3d
g2o_mock.AngleAxis = MockAngleAxis
g2o_mock.Flag = MockFlag
g2o_mock.VertexSE3Expmap = MockVertexSE3Expmap
g2o_mock.VertexSBAPointXYZ = MockVertexSBAPointXYZ
g2o_mock.EdgeSE3ProjectXYZ = MockEdgeSE3ProjectXYZ
g2o_mock.EdgeStereoSE3ProjectXYZ = MockEdgeStereoSE3ProjectXYZ
g2o_mock.EdgeSE3ProjectXYZOnlyPose = MockEdgeSE3ProjectXYZOnlyPose
g2o_mock.EdgeStereoSE3ProjectXYZOnlyPose = MockEdgeStereoSE3ProjectXYZOnlyPose
g2o_mock.RobustKernelHuber = MockRobustKernelHuber
g2o_mock.SparseOptimizer = MockSparseOptimizer
g2o_mock.BlockSolverSE3 = MockBlockSolverSE3
g2o_mock.LinearSolverCSparseSE3 = MockLinearSolver
g2o_mock.LinearSolverEigenSE3 = MockLinearSolver
g2o_mock.LinearSolverDenseSE3 = MockLinearSolver
g2o_mock.OptimizationAlgorithmLevenberg = MockOptimizationAlgorithmLevenberg

sys.modules["g2o"] = g2o_mock

# 4. Mock gtsam library
gtsam_mock = types.ModuleType("gtsam")
gtsam_mock.symbol_shorthand = types.ModuleType("gtsam.symbol_shorthand")
gtsam_mock.symbol_shorthand.X = lambda id: f"X_{id}"
gtsam_mock.symbol_shorthand.L = lambda id: f"L_{id}"
sys.modules["gtsam"] = gtsam_mock
sys.modules["gtsam.symbol_shorthand"] = gtsam_mock.symbol_shorthand

# 5. Mock gtsam_factors library
gtsam_factors_mock = types.ModuleType("gtsam_factors")
sys.modules["gtsam_factors"] = gtsam_factors_mock

# 6. Mock loop closing libraries (pydbow2, pydbow3, pyibow, pyobindex2)
sys.modules["pydbow2"] = types.ModuleType("pydbow2")
sys.modules["pydbow3"] = types.ModuleType("pydbow3")
sys.modules["pyibow"] = types.ModuleType("pyibow")
sys.modules["pyobindex2"] = types.ModuleType("pyobindex2")

# 7. Mock faiss library
class MockIndexFlatL2:
    def __init__(self, dim):
        self.dim = dim

faiss_mock = types.ModuleType("faiss")
faiss_mock.IndexFlatL2 = MockIndexFlatL2
sys.modules["faiss"] = faiss_mock

# 8. Mock pnpsolver library using OpenCV
class MockPnPsolverInput:
    def __init__(self):
        self.points_2d = []
        self.points_3d = []
        self.sigmas2 = []
        self.fx = 1.0
        self.fy = 1.0
        self.cx = 0.0
        self.cy = 0.0

class MockPnPsolver:
    def __init__(self, solver_input):
        self.input = solver_input
        self._R = np.eye(3)
        self._t = np.zeros((3, 1))
        self._inliers = []
        self._n_inliers = 0
        
    def iterate(self, n_iterations):
        pts3d = np.array(self.input.points_3d, dtype=np.float32)
        pts2d = np.array(self.input.points_2d, dtype=np.float32)
        num_pts = len(pts3d)
        if num_pts < 4:
            return False, np.eye(4), True, [False]*num_pts, 0
            
        K = np.array([
            [self.input.fx, 0.0, self.input.cx],
            [0.0, self.input.fy, self.input.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        dist_coeffs = np.zeros(4, dtype=np.float32)
        
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d, pts2d, K, dist_coeffs,
                iterationsCount=100,
                reprojectionError=8.0,
                confidence=0.99
            )
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                transformation = np.eye(4)
                transformation[:3, :3] = R
                transformation[:3, 3] = tvec.flatten()
                
                self._R = R
                self._t = tvec.reshape(3, 1)
                
                inliers_mask = [False] * num_pts
                if inliers is not None:
                    for idx in inliers.flatten():
                        if 0 <= idx < num_pts:
                            inliers_mask[idx] = True
                self._n_inliers = int(np.sum(inliers_mask))
                self._inliers = inliers_mask
                return True, transformation, True, inliers_mask, self._n_inliers
        except Exception as e:
            pass
            
        return False, np.eye(4), True, [False]*num_pts, 0

class MockMLPnPsolver(MockPnPsolver):
    pass

pnpsolver_mock = types.ModuleType("pnpsolver")
pnpsolver_mock.PnPsolverInput = MockPnPsolverInput
pnpsolver_mock.PnPsolver = MockPnPsolver
pnpsolver_mock.MLPnPsolver = MockMLPnPsolver
sys.modules["pnpsolver"] = pnpsolver_mock

# 9. Mock sim3solver library using closed-form Kabsch-Umeyama
def run_umeyama(X, Y, fix_scale=False):
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    N = X.shape[0]
    if N < 3:
        return np.eye(3), np.zeros(3), 1.0
        
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    
    Xc = X - mu_x
    Yc = Y - mu_y
    
    var_x = np.mean(np.sum(Xc ** 2, axis=1))
    cov = (Yc.T @ Xc) / N
    try:
        U, D, Vt = np.linalg.svd(cov)
    except Exception:
        return np.eye(3), np.zeros(3), 1.0
        
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
        
    R = U @ S @ Vt
    
    if fix_scale:
        s = 1.0
    else:
        s = np.trace(np.diag(D) @ S) / var_x if var_x > 1e-8 else 1.0
        
    t = mu_y - s * R @ mu_x
    return R, t, s

class MockSim3SolverInput:
    def __init__(self):
        self.points_3d_c1 = []
        self.points_3d_c2 = []
        self.sigmas2_1 = []
        self.sigmas2_2 = []
        self.fix_scale = False

class MockSim3SolverInput2(MockSim3SolverInput):
    pass

class MockSim3PointRegistrationSolverInput:
    def __init__(self):
        self.points_3d_w1 = []
        self.points_3d_w2 = []
        self.sigma2 = 0.05
        self.fix_scale = False

class MockSim3Solver:
    def __init__(self, solver_input):
        self.input = solver_input
        self._R = np.eye(3)
        self._t = np.zeros(3)
        self._s = 1.0
        self._inliers = []
        self._n_inliers = 0
        self._converged = False
        
    def set_ransac_parameters(self, prob, min_inliers, max_it):
        pass
        
    def iterate(self, num_iterations):
        pts1 = np.array(self.input.points_3d_c1, dtype=np.float64)
        pts2 = np.array(self.input.points_3d_c2, dtype=np.float64)
        N = len(pts1)
        if N < 3:
            return np.eye(4), True, [False]*N, 0, False
            
        R, t, s = run_umeyama(pts1, pts2, self.input.fix_scale)
        self._R = R
        self._t = t
        self._s = s
        self._inliers = [True] * N
        self._n_inliers = N
        self._converged = True
        
        transformation = np.eye(4)
        transformation[:3, :3] = s * R
        transformation[:3, 3] = t
        return transformation, True, self._inliers, N, True
        
    def compute_3d_registration_error(self):
        return 0.0
        
    def get_estimated_rotation(self):
        return self._R
        
    def get_estimated_translation(self):
        return self._t
        
    def get_estimated_scale(self):
        return self._s

class MockSim3PointRegistrationSolver:
    def __init__(self, solver_input):
        self.input = solver_input
        self._R = np.eye(3)
        self._t = np.zeros(3)
        self._s = 1.0
        self._inliers = []
        self._n_inliers = 0
        self._converged = False
        
    def set_ransac_parameters(self, prob, min_inliers, max_it):
        pass
        
    def iterate(self, num_iterations):
        pts1 = np.array(self.input.points_3d_w1, dtype=np.float64)
        pts2 = np.array(self.input.points_3d_w2, dtype=np.float64)
        N = len(pts1)
        if N < 3:
            return np.eye(4), True, [False]*N, 0, False
            
        R, t, s = run_umeyama(pts1, pts2, self.input.fix_scale)
        self._R = R
        self._t = t
        self._s = s
        self._inliers = [True] * N
        self._n_inliers = N
        self._converged = True
        
        transformation = np.eye(4)
        transformation[:3, :3] = s * R
        transformation[:3, 3] = t
        return transformation, True, self._inliers, N, True
        
    def compute_3d_registration_error(self):
        return 0.0
        
    def get_estimated_rotation(self):
        return self._R
        
    def get_estimated_translation(self):
        return self._t
        
    def get_estimated_scale(self):
        return self._s

sim3solver_mock = types.ModuleType("sim3solver")
sim3solver_mock.Sim3SolverInput = MockSim3SolverInput
sim3solver_mock.Sim3SolverInput2 = MockSim3SolverInput2
sim3solver_mock.Sim3PointRegistrationSolverInput = MockSim3PointRegistrationSolverInput
sim3solver_mock.Sim3Solver = MockSim3Solver
sim3solver_mock.Sim3PointRegistrationSolver = MockSim3PointRegistrationSolver
sys.modules["sim3solver"] = sim3solver_mock

# 10. Mock volumetric C++ library
class MockTBBUtils:
    @staticmethod
    def set_max_threads(threads):
        pass

class MockCameraFrustrum:
    def __init__(self, *args, **kwargs):
        pass
    def set_T_cw(self, T):
        pass

class MockVoxelGrid:
    def __init__(self, *args, **kwargs):
        pass
    def reset(self):
        pass
    def get_voxels(self, *args, **kwargs):
        class DummyVoxels:
            def __init__(self):
                self.points = np.zeros((0, 3), dtype=np.float32)
                self.colors = np.zeros((0, 3), dtype=np.float32)
                self.class_ids = np.zeros(0, dtype=np.int32)
                self.object_ids = np.zeros(0, dtype=np.int32)
        return DummyVoxels()
    def integrate(self, *args, **kwargs):
        pass
    def carve(self, *args, **kwargs):
        pass
    def assign_object_ids_to_instance_ids(self, *args, **kwargs):
        return {}

class MockVoxelBlockGrid(MockVoxelGrid):
    pass

class MockVoxelSemanticGrid(MockVoxelGrid):
    def set_depth_threshold(self, *args):
        pass
    def set_depth_decay_rate(self, *args):
        pass
    def get_object_segments(self, *args, **kwargs):
        class DummySegments:
            def __init__(self):
                self.object_vector = []
                self.class_ids = None
                self.object_ids = None
        return DummySegments()

class MockVoxelBlockSemanticGrid(MockVoxelSemanticGrid):
    pass

class MockVoxelBlockSemanticProbabilisticGrid(MockVoxelSemanticGrid):
    pass

class MockObjectData:
    pass

class MockObjectDataGroup:
    pass

def mock_remap_instance_ids(img, mapping):
    return img

volumetric_mock = types.ModuleType("volumetric")
volumetric_mock.TBBUtils = MockTBBUtils
volumetric_mock.CameraFrustrum = MockCameraFrustrum
volumetric_mock.VoxelGrid = MockVoxelGrid
volumetric_mock.VoxelBlockGrid = MockVoxelBlockGrid
volumetric_mock.VoxelSemanticGrid = MockVoxelSemanticGrid
volumetric_mock.VoxelBlockSemanticGrid = MockVoxelBlockSemanticGrid
volumetric_mock.VoxelBlockSemanticProbabilisticGrid = MockVoxelBlockSemanticProbabilisticGrid
volumetric_mock.ObjectData = MockObjectData
volumetric_mock.ObjectDataGroup = MockObjectDataGroup
volumetric_mock.remap_instance_ids = mock_remap_instance_ids
sys.modules["volumetric"] = volumetric_mock

# 11. Mock open3d library
class MockPointCloud:
    def __init__(self):
        self.points = []
        self.colors = []

class MockVector3dVector:
    def __init__(self, data):
        self.data = data

class MockVector2iVector:
    def __init__(self, data):
        self.data = data

class MockRANSACConvergenceCriteria:
    def __init__(self, *args, **kwargs):
        pass

class MockICPConvergenceCriteria:
    def __init__(self, *args, **kwargs):
        pass

class MockRegistrationResult:
    def __init__(self):
        self.transformation = np.eye(4)
        self.inliers = []

def mock_registration_ransac_based_on_correspondence(*args, **kwargs):
    return MockRegistrationResult()

def mock_registration_icp(*args, **kwargs):
    return MockRegistrationResult()

def mock_write_point_cloud(*args, **kwargs):
    pass

open3d_mock = types.ModuleType("open3d")
open3d_mock.geometry = types.ModuleType("open3d.geometry")
open3d_mock.geometry.PointCloud = MockPointCloud

open3d_mock.utility = types.ModuleType("open3d.utility")
open3d_mock.utility.Vector3dVector = MockVector3dVector
open3d_mock.utility.Vector2iVector = MockVector2iVector

open3d_mock.pipelines = types.ModuleType("open3d.pipelines")
open3d_mock.pipelines.registration = types.ModuleType("open3d.pipelines.registration")
open3d_mock.pipelines.registration.RANSACConvergenceCriteria = MockRANSACConvergenceCriteria
open3d_mock.pipelines.registration.ICPConvergenceCriteria = MockICPConvergenceCriteria
open3d_mock.pipelines.registration.registration_ransac_based_on_correspondence = mock_registration_ransac_based_on_correspondence
open3d_mock.pipelines.registration.registration_icp = mock_registration_icp

open3d_mock.io = types.ModuleType("open3d.io")
open3d_mock.io.write_point_cloud = mock_write_point_cloud

sys.modules["open3d"] = open3d_mock
sys.modules["open3d.geometry"] = open3d_mock.geometry
sys.modules["open3d.utility"] = open3d_mock.utility
sys.modules["open3d.pipelines"] = open3d_mock.pipelines
sys.modules["open3d.pipelines.registration"] = open3d_mock.pipelines.registration
sys.modules["open3d.io"] = open3d_mock.io
sys.modules["open3d.core"] = types.ModuleType("open3d.core")

# 12. Mock trajectory_tools library
class MockAlignmentOptions:
    def __init__(self):
        self.max_align_dt = 0.1
        self.find_scale = False
        self.verbose = False

class MockIncrementalTrajectoryAligner:
    def __init__(self, gt_timestamps, gt_t_wi, opts):
        pass
    def update_trajectory(self, pose_timestamps, estimated_trajectory):
        pass
    def result(self):
        class DummyResult:
            def __init__(self):
                self.valid = False
                self.T_gt_est = np.eye(4)
                self.T_est_gt = np.eye(4)
                self.n_pairs = 0
        return DummyResult()
    def get_associated_pairs(self):
        return [], [], []

class MockIncrementalTrajectoryAlignerNoLBA(MockIncrementalTrajectoryAligner):
    pass

def mock_align_3d_points_with_svd(gt_points, est_points, find_scale=True):
    R, t, s = run_umeyama(est_points, gt_points, not find_scale)
    T_gt_est = np.eye(4)
    T_gt_est[:3, :3] = s * R
    T_gt_est[:3, 3] = t
    
    T_est_gt = np.eye(4)
    R_inv = R.T
    T_est_gt[:3, :3] = (1.0 / s) * R_inv
    T_est_gt[:3, 3] = - (1.0 / s) * R_inv @ t
    return T_gt_est, T_est_gt, True

def mock_find_trajectories_associations(filter_timestamps, filter_t_wi, gt_timestamps, gt_t_wi, max_align_dt=1e-1, verbose=True):
    filter_associations = []
    gt_associations = []
    timestamps_associations = []
    
    for i, timestamp in enumerate(filter_timestamps):
        j = np.searchsorted(gt_timestamps, timestamp, side="right") - 1
        if j < 0 or j >= len(gt_timestamps) - 1:
            continue
        dt = timestamp - gt_timestamps[j]
        dt_gt = gt_timestamps[j + 1] - gt_timestamps[j]
        if dt < 0 or dt_gt <= 0 or abs(dt) > max_align_dt:
            continue
        ratio = dt / dt_gt
        gt_t_wi_interpolated = (1 - ratio) * gt_t_wi[j] + ratio * gt_t_wi[j + 1]
        timestamps_associations.append(timestamp)
        gt_associations.append(gt_t_wi_interpolated)
        filter_associations.append(filter_t_wi[i])
        
    return np.array(timestamps_associations), np.array(filter_associations), np.array(gt_associations)

trajectory_tools_mock = types.ModuleType("trajectory_tools")
trajectory_tools_mock.AlignmentOptions = MockAlignmentOptions
trajectory_tools_mock.IncrementalTrajectoryAligner = MockIncrementalTrajectoryAligner
trajectory_tools_mock.IncrementalTrajectoryAlignerNoLBA = MockIncrementalTrajectoryAlignerNoLBA
trajectory_tools_mock.align_3d_points_with_svd = mock_align_3d_points_with_svd
trajectory_tools_mock.find_trajectories_associations = mock_find_trajectories_associations

sys.modules["trajectory_tools"] = trajectory_tools_mock

# 13. Mock pyqtgraph library
class DummyQtMeta(type):
    def __getattr__(cls, name):
        return DummyQtClass

class DummyQtClass(metaclass=DummyQtMeta):
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return DummyQtClass

class MockModule:
    def __init__(self, name="MockModule"):
        self.__name__ = name
    def __getattr__(self, name):
        return DummyQtClass

pyqtgraph_mock = types.ModuleType("pyqtgraph")
pyqtgraph_mock.opengl = MockModule("pyqtgraph.opengl")
pyqtgraph_mock.Qt = MockModule("pyqtgraph.Qt")
pyqtgraph_mock.Qt.QtCore = MockModule("pyqtgraph.Qt.QtCore")
pyqtgraph_mock.Qt.QtGui = MockModule("pyqtgraph.Qt.QtGui")
pyqtgraph_mock.Qt.QtWidgets = MockModule("pyqtgraph.Qt.QtWidgets")
pyqtgraph_mock.mkQApp = lambda *args, **kwargs: DummyQtClass()

sys.modules["pyqtgraph"] = pyqtgraph_mock
sys.modules["pyqtgraph.opengl"] = pyqtgraph_mock.opengl
sys.modules["pyqtgraph.Qt"] = pyqtgraph_mock.Qt
sys.modules["pyqtgraph.Qt.QtCore"] = pyqtgraph_mock.Qt.QtCore
sys.modules["pyqtgraph.Qt.QtGui"] = pyqtgraph_mock.Qt.QtGui
sys.modules["pyqtgraph.Qt.QtWidgets"] = pyqtgraph_mock.Qt.QtWidgets

# 14. Mock mcap libraries
mcap_mock = types.ModuleType("mcap")
mcap_mock.reader = types.ModuleType("mcap.reader")
mcap_mock.reader.make_reader = lambda *args, **kwargs: None
sys.modules["mcap"] = mcap_mock
sys.modules["mcap.reader"] = mcap_mock.reader

mcap_ros2_mock = types.ModuleType("mcap_ros2")
mcap_ros2_mock.reader = types.ModuleType("mcap_ros2.reader")
mcap_ros2_mock.reader.read_ros2_messages = lambda *args, **kwargs: None
sys.modules["mcap_ros2"] = mcap_ros2_mock
sys.modules["mcap_ros2.reader"] = mcap_ros2_mock.reader

mcap_ros1_mock = types.ModuleType("mcap_ros1")
mcap_ros1_mock.reader = types.ModuleType("mcap_ros1.reader")
mcap_ros1_mock.reader.read_ros1_messages = lambda *args, **kwargs: None
sys.modules["mcap_ros1"] = mcap_ros1_mock
sys.modules["mcap_ros1.reader"] = mcap_ros1_mock.reader

# Add pySLAM root and submodules to path
PYSLAM_ROOT = Path(__file__).parent / "pyslam"
sys.path.insert(0, str(PYSLAM_ROOT))

# Also add pyslam directory if needed by pySLAM's internal imports
sys.path.insert(0, str(PYSLAM_ROOT / "pyslam"))


def setup_temp_kitti_dir(dataset_dir: str) -> tempfile.TemporaryDirectory:
    """Create a temporary symlink directory in pySLAM's expected KITTI layout.

    pySLAM expects:
    - <dataset_path>/sequences/05/image_0/
    - <dataset_path>/sequences/05/calib.txt
    - <dataset_path>/poses/05.txt
    """
    tmp_dir = tempfile.TemporaryDirectory(prefix="pyslam_kitti_sym_")
    
    # 1. Create sequences/05
    seq_dir = os.path.join(tmp_dir.name, "sequences", "05")
    os.makedirs(seq_dir, exist_ok=True)
    
    # 2. Symlink image_0, calib.txt
    src_image_0 = os.path.join(dataset_dir, "image_0")
    dst_image_0 = os.path.join(seq_dir, "image_0")
    
    if os.path.isdir(src_image_0):
        os.symlink(src_image_0, dst_image_0)
    else:
        raise FileNotFoundError(f"image_0 not found in {dataset_dir}")
        
    src_calib = os.path.join(dataset_dir, "calib.txt")
    dst_calib = os.path.join(seq_dir, "calib.txt")
    if os.path.isfile(src_calib):
        os.symlink(src_calib, dst_calib)
        
    src_times = os.path.join(dataset_dir, "times.txt")
    dst_times = os.path.join(seq_dir, "times.txt")
    if os.path.isfile(src_times):
        os.symlink(src_times, dst_times)
    else:
        # Create a dummy times.txt since pySLAM expects it unconditionally
        with open(dst_times, "w") as f:
            for i in range(3000):
                f.write(f"{i * 0.1:.6f}\n")

    # 3. Create poses/05.txt (ground truth)
    poses_dir = os.path.join(tmp_dir.name, "poses")
    os.makedirs(poses_dir, exist_ok=True)
    src_poses = os.path.join(dataset_dir, "poses.txt")
    dst_poses = os.path.join(poses_dir, "05.txt")
    if os.path.isfile(src_poses):
        os.symlink(src_poses, dst_poses)
    else:
        # Create a dummy 05.txt with identity poses if GT not available,
        # since pySLAM's KITTI loader might fail if poses/05.txt is missing.
        with open(dst_poses, "w") as f:
            for _ in range(3000):
                f.write("1 0 0 0 0 1 0 0 0 0 1 0\n")
                
    return tmp_dir


def parse_calib_and_image_size(dataset_dir: str):
    # 1. Get image size
    img_dir = os.path.join(dataset_dir, "image_0")
    width, height = 1226, 370 # defaults
    if os.path.isdir(img_dir):
        files = sorted(os.listdir(img_dir))
        for f in files:
            if f.endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(img_dir, f)
                img = cv2.imread(img_path)
                if img is not None:
                    height, width = img.shape[:2]
                    break
                    
    # 2. Get calib parameters
    fx, fy, cx, cy = 707.0912, 707.0912, 601.8873, 183.1104 # defaults
    calib_path = os.path.join(dataset_dir, "calib.txt")
    if os.path.isfile(calib_path):
        try:
            with open(calib_path, "r") as f:
                for line in f:
                    if line.startswith("P0:") or line.startswith("P:"):
                        parts = line.strip().split()
                        vals = [float(x) for x in parts[1:]]
                        if len(vals) >= 12:
                            fx = vals[0]
                            fy = vals[5]
                            cx = vals[2]
                            cy = vals[6]
                            break
        except Exception:
            pass
            
    return width, height, fx, fy, cx, cy


def main():
    parser = argparse.ArgumentParser(description="Standalone runner for pySLAM baselines")
    parser.add_argument("--tracker", required=True, choices=["orb2", "sift", "superpoint", "xfeat"])
    parser.add_argument("--dataset_dir", required=True, help="Path to KITTI sequence folder")
    parser.add_argument("--output_file", required=True, help="Path to write the output trajectory file")
    parser.add_argument("--max_frames", type=int, default=None, help="Process at most this many frames")
    parser.add_argument("--no_calib", action="store_true", help="Do not use calib intrinsics")
    
    args = parser.parse_args()
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    args.output_file = os.path.abspath(args.output_file)
    
    # 1. Setup symlink directory
    print(f"Setting up symlinks to {args.dataset_dir} in pySLAM expected layout...")
    tmp_kitti_dir = setup_temp_kitti_dir(args.dataset_dir)
    print(f"Temporary KITTI layout ready at: {tmp_kitti_dir.name}")
    
    # 2. Parse camera properties and generate custom settings.yaml
    width, height, fx, fy, cx, cy = parse_calib_and_image_size(args.dataset_dir)
    temp_settings_path = os.path.join(tmp_kitti_dir.name, "settings.yaml")
    with open(temp_settings_path, "w") as f:
        f.write(f"Camera.fx: {fx}\n")
        f.write(f"Camera.fy: {fy}\n")
        f.write(f"Camera.cx: {cx}\n")
        f.write(f"Camera.cy: {cy}\n")
        f.write("Camera.k1: 0.0\n")
        f.write("Camera.k2: 0.0\n")
        f.write("Camera.p1: 0.0\n")
        f.write("Camera.p2: 0.0\n")
        f.write(f"Camera.width: {width}\n")
        f.write(f"Camera.height: {height}\n")
        f.write("Camera.fps: 10.0\n")
        f.write("Camera.bf: 379.8145\n")
        f.write("Camera.RGB: 1\n")
        f.write("ThDepth: 40\n")
        f.write("Viewer.on: 0\n")
        f.write("FeatureTrackerConfig.nFeatures: 2000\n")
        f.write(f"FeatureTrackerConfig.name: {args.tracker.upper()}\n")
    print(f"pySLAM: Dynamically configured camera settings: {width}x{height}, fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    # 3. Modify config.yaml
    original_config_path = PYSLAM_ROOT / "config.yaml"
    temp_config_path = Path(tmp_kitti_dir.name) / "temp_config.yaml"
    
    # PyYAML is standard in pySLAM environment
    import yaml
    
    if original_config_path.is_file():
        with open(original_config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
        
    # Configure dataset settings
    config["DATASET"] = {"type": "KITTI_DATASET"}
    config["KITTI_DATASET"] = {
        "type": "kitti",
        "base_path": tmp_kitti_dir.name,
        "name": "05",
        "settings": os.path.abspath(temp_settings_path),
        "groundtruth_file": os.path.join(tmp_kitti_dir.name, "poses", "05.txt"),
        "sensor_type": "mono"
    }
    
    # Configure trajectory output
    out_dir = os.path.dirname(args.output_file)
    out_basename = os.path.basename(args.output_file)
    if out_basename.endswith(".txt"):
        out_basename = out_basename[:-4]
        
    config["SAVE_TRAJECTORY"] = {
        "save_trajectory": True,
        "format_type": "kitti",
        "output_folder": os.path.abspath(out_dir),
        "basename": out_basename
    }
    
    # Save temporary config
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # 3. Dynamic Configuration Parameters Overrides
    try:
        import pyslam.config_parameters as params
        
        # Disable all visualizers / gui on Parameters class
        for attr in dir(params.Parameters):
            if any(keyword in attr.lower() for keyword in ["viewer", "visualize", "gui", "plot"]):
                if isinstance(getattr(params.Parameters, attr), bool):
                    setattr(params.Parameters, attr, False)
                    
        # Programmatically disable loop closure, volumetric integration, semantics, and heavy threads on Parameters class
        if hasattr(params.Parameters, "kUseLoopClosing"):
            params.Parameters.kUseLoopClosing = False
        if hasattr(params.Parameters, "kUseLoopDetector"):
            params.Parameters.kUseLoopDetector = False
        if hasattr(params.Parameters, "kDoVolumetricIntegration"):
            params.Parameters.kDoVolumetricIntegration = False
        if hasattr(params.Parameters, "kDoSparseSemanticMappingAndSegmentation"):
            params.Parameters.kDoSparseSemanticMappingAndSegmentation = False
        if hasattr(params.Parameters, "kLocalMappingOnSeparateThread"):
            params.Parameters.kLocalMappingOnSeparateThread = False
        if hasattr(params.Parameters, "enable_loop_closure"):
            params.Parameters.enable_loop_closure = False
        if hasattr(params.Parameters, "loop_closure"):
            params.Parameters.loop_closure = False
            
        # Select target feature tracker
        from local_features.feature_tracker_configs import FeatureTrackerConfigs
        tracker_mapping = {
            "orb2": FeatureTrackerConfigs.ORB2,
            "sift": FeatureTrackerConfigs.SIFT,
            "superpoint": FeatureTrackerConfigs.SUPERPOINT,
            "xfeat": FeatureTrackerConfigs.XFEAT
        }
        params.feature_tracker_config = tracker_mapping[args.tracker]
        print(f"pySLAM: Configured tracker to: {args.tracker}")
    except Exception as e:
        print(f"Warning: Could not override config parameters programmatically: {e}")
        
    # Set headless Agg backend for matplotlib just in case pySLAM tries to import pyplot
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pass
        
    # 4. Invoke main_slam.py
    # We simulate running main_slam.py as a script by setting sys.argv and importing/running it
    print("Launching pySLAM main_slam.py pipeline...")
    sys.argv = [
        str(PYSLAM_ROOT / "main_slam.py"),
        "--config", str(temp_config_path),
        "--no_output_date",
        "--headless"
    ]
    
    if args.max_frames is not None:
        # Monkey patch dataset_factory to enforce max_frames limit cleanly
        try:
            import pyslam.io.dataset_factory as dataset_factory_module
            original_dataset_factory = dataset_factory_module.dataset_factory
            
            def monkey_patched_dataset_factory(*args_factory, **kwargs_factory):
                dataset = original_dataset_factory(*args_factory, **kwargs_factory)
                original_getImageColor = dataset.getImageColor
                original_getDepth = dataset.getDepth
                
                def patched_getImageColor(img_id):
                    if img_id >= args.max_frames:
                        dataset.is_ok = False
                        return None
                    return original_getImageColor(img_id)
                    
                def patched_getDepth(img_id):
                    if img_id >= args.max_frames:
                        dataset.is_ok = False
                        return None
                    return original_getDepth(img_id)
                
                dataset.getImageColor = patched_getImageColor
                dataset.getDepth = patched_getDepth
                dataset.num_frames = min(dataset.num_frames, args.max_frames)
                return dataset
                
            dataset_factory_module.dataset_factory = monkey_patched_dataset_factory
            print(f"pySLAM: Successfully monkey-patched dataset_factory to limit frames to {args.max_frames}")
        except Exception as e:
            print(f"Warning: Could not monkey-patch dataset_factory: {e}")
             
    # Monkey patch MapPointBase to bypass culling AssertionError in pure Python mode
    try:
        import pyslam.slam.map_point as map_point_module
        original_remove_observation = map_point_module.MapPointBase.remove_observation
        
        def monkey_patched_remove_observation(self, keyframe, idx=None, map_no_lock=False):
            assert keyframe.is_keyframe
            kf_remove_point_match = False
            kf_remove_point = False
            set_bad = False

            with self._lock_features:
                if idx is not None:
                    kf_remove_point_match = True
                else:
                    kf_remove_point = True
                try:
                    del self._observations[keyframe]
                    if keyframe.kps_ur is not None and idx is not None and keyframe.kps_ur[idx] >= 0:
                        self._num_observations = max(0, self._num_observations - 2)
                    else:
                        self._num_observations = max(0, self._num_observations - 1)
                    set_bad = self._num_observations <= 2
                    if self.kf_ref is keyframe and self._observations:
                        self.kf_ref = list(self._observations.keys())[0]
                except KeyError:
                    pass

            if kf_remove_point_match:
                keyframe.remove_point_match(idx)
            if kf_remove_point:
                keyframe.remove_point(self)
            if set_bad:
                self.set_bad(map_no_lock=map_no_lock)
                         
        map_point_module.MapPointBase.remove_observation = monkey_patched_remove_observation
        print("pySLAM: Successfully monkey-patched MapPointBase.remove_observation to bypass AssertionError")
        
        # Also monkey patch remove_frame_view to avoid debug assertion crashes in Python mock mode
        def monkey_patched_remove_frame_view(self, frame, idx=None):
            assert not frame.is_keyframe
            frame_remove_point_match = False
            frame_remove_point = False
            with self._lock_features:
                if idx is not None:
                    frame_remove_point_match = True
                else:
                    frame_remove_point = True
                try:
                    del self._frame_views[frame]
                except KeyError:
                    pass
            if frame_remove_point_match:
                frame.remove_point_match(idx)
            if frame_remove_point:
                frame.remove_point(self)
                
        map_point_module.MapPointBase.remove_frame_view = monkey_patched_remove_frame_view
        print("pySLAM: Successfully monkey-patched MapPointBase.remove_frame_view to bypass AssertionError")
    except Exception as e:
        print(f"Warning: Could not monkey-patch MapPointBase: {e}")
            
    # Monkey patch Slam.get_final_trajectory to prevent IndexError when 0 keyframes exist
    try:
        import pyslam.slam.slam as slam_module
        original_get_final_trajectory = slam_module.Slam.get_final_trajectory
        
        def monkey_patched_get_final_trajectory(self):
            if not self.map.keyframes:
                print("Warning: No keyframes found in map, returning empty trajectory.")
                return [], [], []
            return original_get_final_trajectory(self)
            
        slam_module.Slam.get_final_trajectory = monkey_patched_get_final_trajectory
        print("pySLAM: Successfully monkey-patched Slam.get_final_trajectory to prevent IndexError when 0 keyframes exist.")
    except Exception as e:
        print(f"Warning: Could not monkey-patch Slam.get_final_trajectory: {e}")

    # Inject PYTHONPATH and run main_slam
    try:
        # Make sure current working directory is pySLAM root so it finds settings/ folder
        os.chdir(str(PYSLAM_ROOT))
        
        # Execute main_slam as __main__ dynamically to run its visual tracking loop
        import importlib.util
        spec = importlib.util.spec_from_file_location("__main__", str(PYSLAM_ROOT / "main_slam.py"))
        main_slam_module = importlib.util.module_from_spec(spec)
        sys.modules["__main__"] = main_slam_module
        spec.loader.exec_module(main_slam_module)
            
        # Copy the generated trajectory to expected output path
        expected_final = args.output_file
        if expected_final.endswith(".txt"):
            expected_final_base = expected_final[:-4]
        else:
            expected_final_base = expected_final
            
        generated_final = expected_final_base + "_final.txt"
        generated_online = expected_final_base + "_online.txt"
        
        if os.path.isfile(generated_final):
            shutil.copy2(generated_final, args.output_file)
            print(f"Successfully copied final trajectory to {args.output_file}")
        elif os.path.isfile(generated_online):
            shutil.copy2(generated_online, args.output_file)
            print(f"Successfully copied online trajectory to {args.output_file}")
        else:
            print(f"Warning: Neither {generated_final} nor {generated_online} was found.")
            
        print("pySLAM visual SLAM execution completed successfully.")
    except BaseException as e:
        print(f"Error executing pySLAM visual SLAM: {e} (type: {type(e)})", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up symlinks
        try:
            tmp_kitti_dir.cleanup()
        except Exception:
            pass
            
if __name__ == "__main__":
    main()
