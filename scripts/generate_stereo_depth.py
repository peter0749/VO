#!/usr/bin/env python3
"""Generates metric depth maps from stereo camera images for the KITTI dataset."""
import os
import cv2
import numpy as np
from pathlib import Path


def main():
    data_dir = Path("data/kitti/05")
    left_dir = data_dir / "image_0"
    right_dir = data_dir / "image_1"
    depth_dir = data_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    if not left_dir.exists() or not right_dir.exists():
        print(f"Error: image directories not found under {data_dir}")
        return

    # 1. Parse calibration for fx and baseline
    calib_path = data_dir / "calib.txt"
    if not calib_path.exists():
        print(f"Error: calibration file not found at {calib_path}")
        return

    calib = {}
    with open(calib_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split(":")
            name = parts[0].strip()
            vals = np.fromstring(parts[1], sep=" ")
            calib[name] = vals.reshape(3, 4)

    P0 = calib["P0"]
    P1 = calib["P1"]
    
    fx = P0[0, 0]
    # -fx * baseline = P1[0, 3] -> baseline = -P1[0, 3] / fx
    baseline = -P1[0, 3] / fx
    print(f"Loaded calibration:")
    print(f"  Focal length (fx): {fx:.4f}")
    print(f"  Stereo baseline: {baseline:.4f} meters")

    # 2. Get list of files
    left_files = sorted(os.listdir(left_dir))
    
    # We will compute depth maps for the first 100 frames
    max_frames = 100
    frames_to_process = left_files[:max_frames]
    
    print(f"Computing depth maps for {len(frames_to_process)} frames...")
    
    # 3. Initialize StereoSGBM
    window_size = 3
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    for i, filename in enumerate(frames_to_process):
        left_img_path = left_dir / filename
        right_img_path = right_dir / filename
        
        img_left = cv2.imread(str(left_img_path), cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(str(right_img_path), cv2.IMREAD_GRAYSCALE)
        
        if img_left is None or img_right is None:
            print(f"Failed to read {filename}")
            continue

        # Compute disparity
        disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
        
        # Filter invalid disparities (disparity should be > 0.1 for divide-by-zero prevention)
        valid_mask = disparity > 0.1
        
        # Calculate metric depth in meters
        depth = np.zeros_like(disparity)
        depth[valid_mask] = (fx * baseline) / disparity[valid_mask]
        
        # Clamp depth to reasonable range for outdoor scenes
        depth[depth < 0.1] = 0.1
        depth[depth > 80.0] = 80.0
        
        # Convert to 16-bit PNG format (pixel_value = depth_meters * 256.0)
        depth_scaled = (depth * 256.0).astype(np.uint16)
        
        # Save output
        out_path = depth_dir / filename
        cv2.imwrite(str(out_path), depth_scaled)
        
        if (i + 1) % 20 == 0 or (i + 1) == len(frames_to_process):
            print(f"  Processed {i + 1}/{len(frames_to_process)} frames")

    print(f"Completed! Depth maps saved to {depth_dir}")


if __name__ == "__main__":
    main()
