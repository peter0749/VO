"""Tests for eval/compare.py - visual odometry evaluation comparison pipeline."""
import glob
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_PATH = PROJECT_ROOT / "eval/reports/comparison_report.md"
REPORTS_DIR = PROJECT_ROOT / "eval/reports"


@pytest.fixture(autouse=True)
def cleanup_reports():
    """Clean up reports directory before each test."""
    if REPORTS_DIR.exists():
        for f in REPORTS_DIR.glob("*"):
            if f.is_file():
                f.unlink()
    yield


def test_mock_mode_in_process():
    """Mock mode runs in-process, completes in milliseconds, and generates files."""
    from eval.compare import main
    
    # Run mock mode with baseline skipped
    test_args = ["compare.py", "--mode", "mock", "--skip-baseline"]
    with patch("sys.argv", test_args):
        main()
        
    assert REPORT_PATH.exists()
    
    with open(REPORT_PATH) as f:
        content = f.read()
    assert "Trajectory Comparison Report" in content
    assert "baseline" in content.lower()
    
    pngs = list(REPORTS_DIR.glob("*.png"))
    assert len(pngs) >= 1
    # Confirm side-view plot also created
    assert any("side" in f.name for f in pngs)


@pytest.mark.slow
def test_ci_mode_runs_and_evaluates():
    """CI mode (synthetic orbit trajectory) runs end-to-end and computes evo metrics."""
    from eval.compare import main
    
    # Run CI mode on synthetic orbit sequence using robust lightglue matcher
    test_args = ["compare.py", "--mode", "ci", "--max-frames", "10", "--skip-baseline", "--matcher", "lightglue"]
    
    start_time = time.time()
    with patch("sys.argv", test_args):
        main()
    elapsed = time.time() - start_time
    print(f"\nCI mode in-process elapsed: {elapsed:.2f}s")
    
    assert REPORT_PATH.exists()
    
    with open(REPORT_PATH) as f:
        content = f.read()
    assert "Trajectory Comparison Report" in content
    assert "Ours (slam_dnn)" in content
    
    # Check that plots exist
    pngs = list(REPORTS_DIR.glob("*.png"))
    assert len(pngs) >= 1
