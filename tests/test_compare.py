"""Tests for eval/compare.py - minimal evaluation comparison pipeline.

All tests must complete in < 10 seconds total.
Requires: mock mode (--mode mock --skip-baseline) to be run first.
"""

import glob
import os
import subprocess
import sys
import time

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(PROJECT_ROOT, "eval/reports/comparison_report.md")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "eval/reports")


_mock_result = {}


@pytest.fixture(scope="module", autouse=True)
def run_mock_mode():
    """Run mock mode once before all tests in this module."""
    # Clean up previous reports
    for f in glob.glob(os.path.join(REPORTS_DIR, "*")):
        os.remove(f)

    start = time.time()
    result = subprocess.run(
        [sys.executable, "eval/compare.py", "--mode", "mock", "--skip-baseline"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    elapsed = time.time() - start
    print(f"\nMock mode elapsed: {elapsed:.2f}s")
    _mock_result["elapsed"] = elapsed
    _mock_result["returncode"] = result.returncode
    _mock_result["stderr"] = result.stderr
    _mock_result["stdout"] = result.stdout
    yield


def test_mock_mode_runs_fast():
    """Mock mode completes in < 5 seconds."""
    assert _mock_result.get("returncode") == 0, f"Mock mode failed: {_mock_result.get('stderr')}"
    elapsed = _mock_result["elapsed"]
    print(f"Elapsed: {elapsed:.2f}s")
    assert elapsed < 5.0, f"Mock mode too slow: {elapsed:.2f}s"


def test_report_file_created():
    """Report file exists after mock run."""
    assert os.path.exists(REPORT_PATH), f"Report not found at {REPORT_PATH}"


def test_plot_png_created():
    """At least 1 PNG plot exists after mock run."""
    pngs = glob.glob(os.path.join(REPORTS_DIR, "*.png"))
    assert len(pngs) >= 1, f"No PNGs found in {REPORTS_DIR}: {pngs}"


def test_skip_baseline_note():
    """Report contains 'Baseline' when --skip-baseline is used."""
    with open(REPORT_PATH) as f:
        content = f.read()
    assert "Baseline" in content, f"Expected 'Baseline' in report, got:\n{content[:500]}"
