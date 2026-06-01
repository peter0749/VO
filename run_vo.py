#!/usr/bin/env python3
"""Backward-compatible CLI entry point — delegates to slam_dnn.cli.main()."""
from slam_dnn.cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
