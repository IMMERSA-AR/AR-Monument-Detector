"""
create_zip.py
=============
Creates panel_augmented_dataset.zip with UNIX-style forward-slash paths.

PowerShell's Compress-Archive writes Windows backslashes inside the zip,
which causes Python's zipfile on Linux (Colab) to treat them as literal
filename characters instead of directory separators.

This script uses Python's zipfile module directly, which always writes
forward slashes — compatible with Linux, macOS, and Windows.

Usage (run from ObeliskScene root):
    python panel\\create_zip.py
"""

import zipfile
import os

SOURCE_DIR = "panel/panel_augmented_dataset"
OUTPUT_ZIP = "panel/panel_augmented_dataset.zip"

if not os.path.isdir(SOURCE_DIR):
    print(f"ERROR: '{SOURCE_DIR}' not found. Run from ObeliskScene root.")
    raise SystemExit(1)

file_count = 0

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(SOURCE_DIR):
        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            # arcname uses forward slashes (replace OS separator)
            arcname = abs_path.replace("\\", "/")
            zf.write(abs_path, arcname)
            file_count += 1

size_mb = os.path.getsize(OUTPUT_ZIP) / 1e6
print(f"Created: {OUTPUT_ZIP}")
print(f"Files  : {file_count}")
print(f"Size   : {size_mb:.1f} MB")
print("(This zip uses forward-slash paths -- safe to extract on Linux/Colab)")
