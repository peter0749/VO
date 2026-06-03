"""Wrapper for the pySLAM baseline VO/SLAM system.

This module provides an adapter to run pySLAM's visual odometry pipeline on
KITTI-format datasets with multiple feature tracker configurations (ORB2, SIFT,
SuperPoint, XFeat) for baseline comparison with slam_dnn.

IMPORTANT:
- This file is part of slam_dnn, NOT pySLAM.
- No slam_dnn core module (slam_dnn.*) should import from this file.
- Imports happen at runtime to allow graceful failure when dependencies are missing.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PYSLAM_DIR = PROJECT_ROOT / "baselines" / "pyslam"
RUNNER_SCRIPT = PROJECT_ROOT / "baselines" / "pyslam_run.py"


def check_pyslam_available() -> bool:
    """Returns True if pySLAM is installed and can be run.

    Checks that the pySLAM folder exists and the helper runner script is present.
    """
    if not PYSLAM_DIR.is_dir():
        logger.debug("pySLAM directory not found at: %s", PYSLAM_DIR)
        return False
    
    # Check if a python interpreter is available in one of the expected pySLAM environments
    python_exe = _find_pyslam_python()
    if python_exe is None:
        logger.debug("No Python environment found inside pySLAM directory")
        return False
        
    return RUNNER_SCRIPT.is_file()


def _find_pyslam_python() -> str | None:
    """Search for the python executable of pySLAM's virtual environment.

    Checks standard locations:
    - baselines/pyslam/venv/bin/python
    - baselines/pyslam/.venv/bin/python
    - baselines/pyslam/pyslam/bin/python
    - System python as a fallback if the above are not found but pySLAM is present
    """
    paths_to_check = [
        PYSLAM_DIR / "venv" / "bin" / "python",
        PYSLAM_DIR / ".venv" / "bin" / "python",
        PYSLAM_DIR / "pyslam" / "bin" / "python",
    ]
    
    for path in paths_to_check:
        if path.is_file():
            return str(path)
            
    # Check if a system python is available
    if PYSLAM_DIR.is_dir():
        return sys.executable
        
    return None


def run_pyslam_on_kitti(
    data_dir: str,
    output_dir: str,
    tracker: str = "orb2",
    use_calib_intrinsics: bool = True,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Run pySLAM on a KITTI-format dataset with the selected feature tracker.

    Invokes `baselines/pyslam_run.py` in the pySLAM virtual environment via a
    subprocess, then reads and parses the generated KITTI format trajectory.

    Args:
        data_dir: Path to KITTI sequence directory. Expected layout containing image_0, calib.txt, poses.txt.
        output_dir: Directory where output files are written.
        tracker: The feature tracker to use ('orb2', 'sift', 'superpoint', 'xfeat').
        use_calib_intrinsics: If True, parse intrinsics from calib.txt.
        max_frames: Process at most this many frames. None = all frames.

    Returns:
        List of 4x4 estimated pose matrices (np.ndarray, float64).
        Returns an empty list on failure.
    """
    if not check_pyslam_available():
        logger.error("pySLAM baseline is not available or not configured.")
        return []

    python_exe = _find_pyslam_python()
    if python_exe is None:
        logger.error("Could not find python environment inside baselines/pyslam")
        return []

    os.makedirs(output_dir, exist_ok=True)
    temp_trajectory_file = os.path.join(output_dir, f"pyslam_{tracker}_raw_trajectory.txt")
    
    # Construct subprocess command
    cmd = [
        python_exe,
        str(RUNNER_SCRIPT),
        "--tracker", tracker,
        "--dataset_dir", os.path.abspath(data_dir),
        "--output_file", os.path.abspath(temp_trajectory_file),
    ]
    
    if max_frames is not None:
        cmd.extend(["--max_frames", str(max_frames)])
        
    if not use_calib_intrinsics:
        cmd.append("--no_calib")
        
    logger.info("Executing pySLAM (%s) on KITTI sequence via subprocess...", tracker)
    logger.info("Command: %s", " ".join(cmd))
    
    try:
        # Run subprocess with environment PYTHONPATH set to include pySLAM root
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PYSLAM_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYSLAM_USE_CPP"] = "false"
        
        result = subprocess.run(
            cmd,
            cwd=str(PYSLAM_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,  # 30-minute timeout
        )
        
        # Log stdout/stderr on failure or debug level on success
        if result.returncode != 0:
            logger.error("pySLAM execution failed with exit code %d", result.returncode)
            logger.error("Subprocess output:\n%s", result.stdout)
            return []
        else:
            logger.debug("pySLAM execution completed successfully.")
            logger.debug("Subprocess output:\n%s", result.stdout)
            
    except subprocess.TimeoutExpired:
        logger.error("pySLAM execution timed out.")
        return []
    except Exception as e:
        logger.error("Failed to run pySLAM subprocess: %s", e)
        return []
        
    # Read the output trajectory
    # If the process exited abruptly via os._exit(), the copying phase in pyslam_run.py might have been bypassed.
    # We fallback to check if _final.txt or _online.txt exist in the output folder.
    if not os.path.isfile(temp_trajectory_file):
        base_path = temp_trajectory_file[:-4] if temp_trajectory_file.endswith(".txt") else temp_trajectory_file
        final_path = base_path + "_final.txt"
        online_path = base_path + "_online.txt"
        import shutil
        if os.path.isfile(final_path):
            shutil.copy2(final_path, temp_trajectory_file)
            logger.info("pySLAM wrapper: Recovered trajectory from %s", final_path)
        elif os.path.isfile(online_path):
            shutil.copy2(online_path, temp_trajectory_file)
            logger.info("pySLAM wrapper: Recovered trajectory from %s", online_path)

    if not os.path.isfile(temp_trajectory_file):
        logger.error("pySLAM output trajectory file was not generated at: %s", temp_trajectory_file)
        return []
        
    poses = []
    try:
        with open(temp_trajectory_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [float(v) for v in line.split()]
                if len(parts) == 12:
                    # KITTI format: 12 floats representing row-major 3x4 matrix
                    pose = np.eye(4, dtype=np.float64)
                    pose[:3, :] = np.array(parts).reshape(3, 4)
                    poses.append(pose)
                else:
                    logger.warning("Malformed trajectory line: %s", line)
    except Exception as e:
        logger.error("Error reading pySLAM trajectory file: %s", e)
        return []
        
    # Move/Rename final trajectory for the baseline framework
    final_output_path = os.path.join(output_dir, f"pyslam_{tracker}_trajectory.txt")
    try:
        if os.path.exists(final_output_path):
            os.remove(final_output_path)
        os.rename(temp_trajectory_file, final_output_path)
        logger.info("pySLAM (%s) trajectory saved: %s (%d poses)", tracker, final_output_path, len(poses))
    except Exception as e:
        logger.warning("Could not rename trajectory file: %s", e)
        
    return poses
