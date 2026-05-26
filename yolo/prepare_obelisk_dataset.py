"""
prepare_yolo_dataset.py
=======================
Runs obelisk detection on every image in 'Obelisk Photos/' and exports
a YOLO-ready dataset that you can feed straight into 'yolo train'.

Output structure
----------------
yolo_dataset/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt        (YOLO format: "0 cx cy w h")
    labels/val/*.txt
    previews/train/*.jpg      (annotated for visual review)
    previews/val/*.jpg
    data.yaml
    detection_report.txt

Usage
-----
    python prepare_yolo_dataset.py

Optional: place your clearest obelisk photo as 'reference_obelisk.jpg'
in this folder. If not present, the first photo in 'Obelisk Photos/' is
used as the reference automatically.

Requirements
------------
    pip install opencv-contrib-python numpy pyyaml
"""

import cv2
import numpy as np
import os
import random
import shutil
import sys
import yaml

from obelisk.obelisk_detection_fm import ObeliskFeatureMatcher


# ── Configuration ──────────────────────────────────────────────────────────

PHOTOS_FOLDER   = "Obelisk Photos"
OUTPUT_FOLDER   = "yolo_dataset"
REFERENCE_IMAGE = "reference_obelisk.jpg"   # optional — auto-selected if missing

VAL_SPLIT       = 0.20      # 20 % of detected images go to val
RANDOM_SEED     = 42

# Feature matcher parameters
MIN_MATCHES     = 10        # lower → more detections, possibly noisier bboxes
RATIO_THRESHOLD = 0.75      # Lowe ratio test (0.70–0.80 is typical)

# Class name used in data.yaml
CLASS_NAME = "obelisk"


# ── Helpers ────────────────────────────────────────────────────────────────

def build_output_dirs():
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUTPUT_FOLDER, "images",   split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, "labels",   split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, "previews", split), exist_ok=True)


def write_yolo_label(path, bbox):
    """Write a single-object YOLO label file."""
    cx = bbox["cx"]
    cy = bbox["cy"]
    w  = bbox["width"]
    h  = bbox["height"]
    # Clamp to valid range
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.001, min(1.0, w))
    h  = max(0.001, min(1.0, h))
    with open(path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def draw_bbox(frame, bbox, inliers):
    """Draw a green bounding box and inlier count on a copy of the frame."""
    vis = frame.copy()
    fh, fw = vis.shape[:2]
    x  = int(bbox["x"]  * fw)
    y  = int(bbox["y"]  * fh)
    x2 = int((bbox["x"] + bbox["width"])  * fw)
    y2 = int((bbox["y"] + bbox["height"]) * fh)
    cx = int(bbox["cx"] * fw)
    cy = int(bbox["cy"] * fh)

    cv2.rectangle(vis, (x, y), (x2, y2), (0, 255, 0), 3)
    cv2.circle(vis, (cx, cy), 7, (0, 255, 255), -1)
    label = f"obelisk  inliers={inliers}"
    cv2.putText(vis, label, (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return vis


def write_data_yaml(train_path, val_path):
    data = {
        "path"  : os.path.abspath(OUTPUT_FOLDER),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : 1,
        "names" : [CLASS_NAME],
    }
    yaml_path = os.path.join(OUTPUT_FOLDER, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def find_reference():
    """Return path to reference image: explicit file > first photo in folder."""
    if os.path.exists(REFERENCE_IMAGE):
        return REFERENCE_IMAGE

    all_photos = sorted(
        f for f in os.listdir(PHOTOS_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not all_photos:
        print(f"ERROR: No images found in '{PHOTOS_FOLDER}/'.")
        sys.exit(1)

    ref = os.path.join(PHOTOS_FOLDER, all_photos[0])
    print(f"No '{REFERENCE_IMAGE}' found.")
    print(f"Using first photo as reference: '{ref}'")
    print("Tip: copy your sharpest close-up as 'reference_obelisk.jpg' for better results.\n")
    return ref


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    build_output_dirs()

    # ── Reference ──────────────────────────────────────────────────
    reference_path = find_reference()
    matcher = ObeliskFeatureMatcher(
        reference_path  = reference_path,
        min_matches     = MIN_MATCHES,
        ratio_threshold = RATIO_THRESHOLD,
    )

    # ── Collect all photos (exclude the reference itself) ──────────
    all_photos = sorted(
        f for f in os.listdir(PHOTOS_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.abspath(os.path.join(PHOTOS_FOLDER, f)) != os.path.abspath(reference_path)
    )
    print(f"\nProcessing {len(all_photos)} images...\n")

    detected = []   # list of (fname, frame, result)
    failed   = []   # list of (fname, reason)

    for fname in all_photos:
        src_path = os.path.join(PHOTOS_FOLDER, fname)
        frame    = cv2.imread(src_path)
        if frame is None:
            failed.append((fname, "cannot read file"))
            continue

        result = matcher.detect(frame)

        if result["detected"]:
            detected.append((fname, frame, result))
            print(f"  OK   {fname:<30}  inliers={result['matches']:3d}  "
                  f"bbox_h={result['bbox']['height']:.2f}")
        else:
            failed.append((fname, result["reason"]))
            print(f"  MISS {fname:<30}  {result['reason']}")

    # ── Train / val split ──────────────────────────────────────────
    random.shuffle(detected)
    n_val   = max(1, int(len(detected) * VAL_SPLIT)) if detected else 0
    val_set = detected[:n_val]
    trn_set = detected[n_val:]

    splits = [("train", trn_set), ("val", val_set)]

    for split, items in splits:
        for fname, frame, result in items:
            stem  = os.path.splitext(fname)[0]
            bbox  = result["bbox"]
            inliers = result["matches"]

            # Copy original image
            dst_img = os.path.join(OUTPUT_FOLDER, "images", split, fname)
            shutil.copy2(os.path.join(PHOTOS_FOLDER, fname), dst_img)

            # YOLO label
            dst_lbl = os.path.join(OUTPUT_FOLDER, "labels", split, stem + ".txt")
            write_yolo_label(dst_lbl, bbox)

            # Annotated preview
            preview = draw_bbox(frame, bbox, inliers)
            dst_prv = os.path.join(OUTPUT_FOLDER, "previews", split, fname)
            cv2.imwrite(dst_prv, preview)

    # ── data.yaml ──────────────────────────────────────────────────
    yaml_path = write_data_yaml(
        os.path.join(OUTPUT_FOLDER, "images", "train"),
        os.path.join(OUTPUT_FOLDER, "images", "val"),
    )

    # ── Report ─────────────────────────────────────────────────────
    report_lines = [
        f"Obelisk YOLO Dataset Preparation Report",
        f"========================================",
        f"Reference image   : {reference_path}",
        f"Photos processed  : {len(all_photos)}",
        f"Detected          : {len(detected)}",
        f"Not detected      : {len(failed)}",
        f"Train images      : {len(trn_set)}",
        f"Val   images      : {len(val_set)}",
        f"",
        f"YOLO label format : 0 cx cy w h  (class 0 = obelisk)",
        f"",
        f"NOT DETECTED ({len(failed)} images)",
        f"-" * 40,
    ]
    for fname, reason in failed:
        report_lines.append(f"  {fname:<35}  {reason}")

    report_path = os.path.join(OUTPUT_FOLDER, "detection_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Dataset ready:  {len(detected)} / {len(all_photos)} images annotated")
    print(f"  Train  : {len(trn_set)} images  →  {OUTPUT_FOLDER}/images/train/")
    print(f"  Val    : {len(val_set)} images  →  {OUTPUT_FOLDER}/images/val/")
    print(f"  Labels : {OUTPUT_FOLDER}/labels/")
    print(f"  YAML   : {yaml_path}")
    print(f"  Report : {report_path}")
    print(f"  Previews: {OUTPUT_FOLDER}/previews/  ← visually review these!")
    print(f"\nTo train YOLO (example with YOLOv8):")
    print(f"  pip install ultralytics")
    print(f"  yolo train model=yolov8n.pt data={yaml_path} epochs=100 imgsz=640")
    print(f"{'='*55}")

    if len(detected) < 5:
        print("\nWARNING: fewer than 5 images detected.")
        print("Consider:")
        print("  1. Setting a better reference (rename your sharpest photo to 'reference_obelisk.jpg')")
        print("  2. Lowering MIN_MATCHES (currently {MIN_MATCHES}) in this script")
        print("  3. Checking previews/ to verify detection quality")


if __name__ == "__main__":
    main()
