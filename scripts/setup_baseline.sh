#!/bin/bash
# setup_baseline.sh - Initialize and configure baseline submodules.
#
# This script is idempotent: safe to re-run at any time.
# It initializes the minislam submodule, installs its dependencies,
# and verifies the wrapper is importable.
#
# Usage:
#   bash scripts/setup_baseline.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== Baseline Setup ==="
echo "Project root: $PROJECT_ROOT"

# Step 1: Initialize submodule if needed
if [ ! -f baselines/minislam/.git ] && [ ! -d baselines/minislam/.git ]; then
    echo "Initializing minislam submodule..."
    git submodule update --init --recursive baselines/minislam
else
    echo "Submodule already initialized."
fi

# Step 2: Verify pinned commit
PINNED="962096d5bb8919317cceef9c0f2f98f023d9fcf3"
ACTUAL="$(git -C baselines/minislam rev-parse HEAD 2>/dev/null || echo 'unknown')"
if [ "$ACTUAL" = "$PINNED" ]; then
    echo "Submodule at pinned commit: ${PINNED:0:8}"
else
    echo "WARNING: submodule at $ACTUAL, expected $PINNED"
    echo "Run: cd baselines/minislam && git checkout $PINNED"
fi

# Step 3: Install minislam dependencies (uses hatchling/pyproject.toml)
echo "Installing minislam package..."
_install_ok=false
for _pip in "pip" "pip3" "python3 -m pip"; do
    if $_pip install -e baselines/minislam --no-build-isolation --quiet 2>/dev/null; then
        _install_ok=true
        break
    fi
done
if [ "$_install_ok" = true ]; then
    echo "minislam installed successfully."
else
    echo "NOTE: pip install failed. Try: python3 -m pip install -e baselines/minislam --break-system-packages"
    echo "The wrapper handles missing dependencies gracefully."
fi
unset _install_ok

# Step 4: Verify wrapper import works
echo "Verifying wrapper import..."
if python3 -c "from baselines.minislam_wrapper import run_minislam_on_kitti, check_minislam_available" 2>/dev/null; then
    echo "Wrapper import: OK"
else
    echo "WARNING: Wrapper import failed. Check that baselines/minislam_wrapper.py exists."
    exit 1
fi

# Step 5: Check availability
python3 -c "
from baselines.minislam_wrapper import check_minislam_available
if check_minislam_available():
    print('minislam: available and ready')
else:
    print('minislam: not importable (dependencies may be missing)')
    print('This is OK - the wrapper will report baseline unavailable at runtime.')
"

echo "=== Baseline setup complete ==="
