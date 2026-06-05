#!/usr/bin/env python3
"""Download ETH RPG lightweight dataset subsets for VO evaluation.

Supported datasets:
  - kitti05: KITTI sequence 05 (2761 frames, ~1.4 GB)
  - parking: Parking dataset (~208 MB)

Usage:
    python scripts/download_data.py --dataset kitti05 --output-dir data/
    python scripts/download_data.py --dataset all --output-dir data/
    python scripts/download_data.py --dataset kitti05 --output-dir data/ --verify
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional

# Dataset definitions
DATASETS: Dict[str, dict] = {
    "kitti05": {
        "url": "https://rpg.ifi.uzh.ch/docs/teaching/2024/kitti05.zip",
        "filename": "kitti05.zip",
        "sha256": None,  # Update after first verified download
        "expected_structure": {
            "image_0": "dir",
            "calib.txt": "file",
            "poses.txt": "file",
            "times.txt": "file",
        },
    },
    "parking": {
        "url": "https://rpg.ifi.uzh.ch/docs/teaching/2024/parking.zip",
        "filename": "parking.zip",
        "sha256": None,  # Update after first verified download
        "expected_structure": None,  # Structure validated post-extract
    },
}


def compute_sha256(filepath: Path, chunk_size: int = 8192 * 1024) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_with_wget(url: str, dest: Path) -> bool:
    """Download using wget with resume support. Returns True on success."""
    if shutil.which("wget") is None:
        return False
    try:
        result = subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "-O", str(dest), url],
            timeout=3600,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def download_with_urllib(url: str, dest: Path) -> bool:
    """Download using urllib as fallback. Supports partial resume via Range header."""
    try:
        existing_size = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url)
        if existing_size > 0:
            req.add_header("Range", f"bytes={existing_size}-")

        response = urllib.request.urlopen(req, timeout=60)
        total_size = response.headers.get("Content-Length")

        if response.status == 206:
            # Partial content - server supports resume
            mode = "ab"
            print(f"  Resuming from {existing_size / (1024**2):.1f} MB")
            if total_size:
                total_size = existing_size + int(total_size)
        elif response.status == 200:
            # Full download - server doesn't support Range
            mode = "wb"
            existing_size = 0
            if total_size:
                total_size = int(total_size)
        else:
            return False

        downloaded = existing_size

        with open(dest, mode) as f:
            while True:
                chunk = response.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = downloaded / total_size * 100
                    print(
                        f"\r  Downloading: {downloaded / (1024**2):.1f} / "
                        f"{total_size / (1024**2):.1f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  Downloading: {downloaded / (1024**2):.1f} MB",
                        end="",
                        flush=True,
                    )
        print()
        return True
    except Exception as e:
        print(f"\n  urllib download failed: {e}")
        return False


def download_file(url: str, dest: Path) -> bool:
    """Download a file with resume support. Tries wget first, then urllib."""
    print(f"Downloading {url}")
    print(f"  Destination: {dest}")

    if download_with_wget(url, dest):
        return True

    print("  wget not available or failed, trying urllib...")
    return download_with_urllib(url, dest)


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract a zip file to the specified directory."""
    print(f"Extracting {zip_path.name} to {extract_dir}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Check if zip has a top-level directory matching the dataset name
            names = zf.namelist()
            top_dirs = {n.split("/")[0] for n in names if "/" in n}

            # If there's exactly one top-level directory, extract directly
            # Otherwise extract into a subdirectory named after the zip
            if len(top_dirs) == 1:
                target = extract_dir / list(top_dirs)[0]
                if target.exists():
                    print(f"  Dataset directory already exists: {target}")
                    print("  Skipping extraction (remove manually to re-extract)")
                    return True
                zf.extractall(extract_dir)
            else:
                # No single top-level dir - extract into dataset-named dir
                dataset_name = zip_path.stem
                target = extract_dir / dataset_name
                if target.exists():
                    print(f"  Dataset directory already exists: {target}")
                    print("  Skipping extraction (remove manually to re-extract)")
                    return True
                target.mkdir(parents=True, exist_ok=True)
                zf.extractall(target)

        print("  Extraction complete")
        return True
    except (zipfile.BadZipFile, OSError) as e:
        print(f"  Extraction failed: {e}")
        return False


def verify_dataset(name: str, dataset_dir: Path) -> bool:
    """Verify dataset structure after extraction."""
    info = DATASETS[name]
    structure = info.get("expected_structure")

    if structure is None:
        print(f"  No structure validation defined for {name}")
        return True

    all_ok = True
    for entry, entry_type in structure.items():
        path = dataset_dir / entry
        if entry_type == "dir":
            if not path.is_dir():
                print(f"  MISSING directory: {entry}")
                all_ok = False
            else:
                file_count = len(list(path.iterdir()))
                print(f"  OK: {entry}/ ({file_count} files)")
        elif entry_type == "file":
            if not path.is_file():
                print(f"  MISSING file: {entry}")
                all_ok = False
            else:
                size = path.stat().st_size
                print(f"  OK: {entry} ({size} bytes)")

    return all_ok


def process_dataset(
    name: str, output_dir: Path, verify: bool, keep_zip: bool
) -> bool:
    """Download, extract, and optionally verify a single dataset."""
    info = DATASETS[name]
    zip_path = output_dir / info["filename"]
    dataset_dir = output_dir / name

    # Download
    if not download_file(info["url"], zip_path):
        print(f"ERROR: Failed to download {name}")
        return False

    # Verify SHA256 if requested
    if verify:
        print(f"Computing SHA256 of {zip_path.name}...")
        actual_hash = compute_sha256(zip_path)
        expected_hash = info.get("sha256")

        if expected_hash is None:
            print(f"  No known hash for {name}. Computed: {actual_hash}")
            print("  Update DATASETS sha256 field to enable verification.")
        elif actual_hash != expected_hash:
            print(f"  SHA256 MISMATCH!")
            print(f"  Expected: {expected_hash}")
            print(f"  Got:      {actual_hash}")
            return False
        else:
            print(f"  SHA256 OK: {actual_hash[:16]}...")

    # Extract
    if not extract_zip(zip_path, output_dir):
        print(f"ERROR: Failed to extract {name}")
        return False

    # Handle nested directory: if zip extracts to kitti05/ inside output_dir,
    # and output_dir is data/, the dataset ends up at data/kitti05/
    # But if the zip has a different top-level name, rename it
    extracted_dirs = [
        d for d in output_dir.iterdir() if d.is_dir() and d.name != "synthetic"
    ]
    if not dataset_dir.exists():
        # Check if zip extracted with a different name
        for d in extracted_dirs:
            if d.name != name and (d / "poses.txt").exists():
                print(f"  Renaming {d.name} -> {name}")
                d.rename(dataset_dir)
                break

    # Verify structure
    if dataset_dir.exists():
        print(f"Verifying {name} structure...")
        if not verify_dataset(name, dataset_dir):
            print(f"WARNING: {name} structure verification failed")
    else:
        print(f"WARNING: Expected dataset directory not found: {dataset_dir}")

    # Clean up zip
    if not keep_zip:
        print(f"Removing {zip_path.name}")
        zip_path.unlink()

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ETH RPG lightweight dataset subsets"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="kitti05",
        help="Which dataset to download (default: kitti05)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to download and extract datasets (default: data/)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify SHA256 checksum after download",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep downloaded zip files after extraction",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to process
    if args.dataset == "all":
        datasets = list(DATASETS.keys())
    else:
        datasets = [args.dataset]

    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Datasets: {', '.join(datasets)}")
    print()

    all_ok = True
    for name in datasets:
        print(f"{'=' * 60}")
        print(f"Processing: {name}")
        print(f"{'=' * 60}")
        ok = process_dataset(name, args.output_dir, args.verify, args.keep_zip)
        if ok:
            print(f"OK: {name}")
        else:
            print(f"FAILED: {name}")
            all_ok = False
        print()

    if all_ok:
        print("All datasets processed successfully.")
        return 0
    else:
        print("Some datasets failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
