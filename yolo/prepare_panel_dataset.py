"""
prepare_panel_dataset.py
========================
Generates YOLO-format annotation labels for all 9 exhibition panel photos.

How it works
------------
1. Loads all images from OLD_GP_PHOTOS_FOLDER (your 9 panel photos).
2. For each image, runs Stage 1 of PanelDetector (dark frame detection)
   to get the bounding box of the panel in the image.
3. The class ID is determined by the file's alphabetical index
   (IMG_7066 = class 0, IMG_7067 = class 1, ... IMG_7074 = class 8).
   This is the ground truth — we KNOW which image belongs to which panel.
4. Writes a YOLO label file:  class_id  cx  cy  width  height  (all 0-1)
5. Saves an annotated preview image so you can visually verify every bbox.
6. Writes data.yaml with all 9 class names.

Why class_id comes from file index, not from the detector
----------------------------------------------------------
The detector identifies panels by matching against references.
Since each image IS its own reference, the detected panel_id would
always match the file index anyway — but using the file index directly
as ground truth is safer and simpler.

Output structure
----------------
panel_yolo_dataset/
    images/       all 9 images (flat — no train/val split yet)
    labels/       9 YOLO .txt files  (class_id cx cy w h)
    previews/     9 annotated images for visual review
    data.yaml     ready for Roboflow upload or direct YOLO training

Next step after running this script
------------------------------------
Upload panel_yolo_dataset/ to Roboflow:
    - Roboflow applies augmentations -> ~1000+ images
    - Roboflow splits into train/val automatically
    - Export from Roboflow in YOLOv8 format
    - Then run:  yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640

Requirements
------------
    pip install opencv-contrib-python numpy pyyaml
"""

import cv2
import numpy as np
import os
import shutil
import sys
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detection.panel.pipeline import PanelDetector


# ── Configuration ──────────────────────────────────────────────────────────

OLD_GP_PHOTOS_FOLDER = "data/raw/panel"
OUTPUT_FOLDER        = "data/yolo/panel"

# Panel names — must match the ALPHABETICAL order of image files.
# IMG_7066.jpg -> class 0, IMG_7067.jpg -> class 1, ... IMG_7074.jpg -> class 8
# Edit these names to match the actual content of each panel.
PANEL_NAMES = [
    "College History Text",        # class 0 — IMG_7066.jpg
    "University Dome Building",    # class 1 — IMG_7067.jpg
    "Mechanical Engineering",      # class 2 — IMG_7068.jpg
    "Gate and Mining Cart",        # class 3 — IMG_7069.jpg
    "Hydraulics and Tile Work",    # class 4 — IMG_7070.jpg
    "Architecture and Ornament Studies",      # class 5 — IMG_7071.jpg
    "Survey Instrument and Classical Facades", # class 6 — IMG_7072.jpg
    "Corinthian Capital and Industrial Machinery", # class 7 — IMG_7073.jpg
    "Steam Engine and Polytechnique Relief",  # class 8 — IMG_7074.jpg
]


# ── Helpers ────────────────────────────────────────────────────────────────

def build_output_dirs():
    os.makedirs(os.path.join(OUTPUT_FOLDER, "images"),   exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "labels"),   exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "previews"), exist_ok=True)


def write_yolo_label(label_path, class_id, bbox):
    """
    Write one YOLO label file for a single detected object.

    YOLO format (one line per object):
        class_id  cx  cy  width  height
    All values are normalized 0.0 - 1.0.
    class_id is an integer (0 = first panel, 1 = second panel, etc.)
    """
    cx = float(bbox["cx"])
    cy = float(bbox["cy"])
    w  = float(bbox["width"])
    h  = float(bbox["height"])

    # Clamp to valid range — just in case the detector goes slightly outside
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.001, min(1.0, w))
    h  = max(0.001, min(1.0, h))

    with open(label_path, "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def draw_bbox(frame, class_id, class_name, bbox, score, stage):
    """
    Draw the bounding box on a copy of the frame for visual review.
    Green box  = bounding box
    Yellow dot = center point
    Label shows class name, class ID, RANSAC score, and detection stage.
    """
    vis = frame.copy()
    fh, fw = vis.shape[:2]

    x  = int(bbox["x"]      * fw)
    y  = int(bbox["y"]      * fh)
    x2 = int((bbox["x"] + bbox["width"])  * fw)
    y2 = int((bbox["y"] + bbox["height"]) * fh)
    cx_px = int(bbox["cx"] * fw)
    cy_px = int(bbox["cy"] * fh)

    # Bounding box
    cv2.rectangle(vis, (x, y), (x2, y2), (0, 255, 0), 4)

    # Center dot
    cv2.circle(vis, (cx_px, cy_px), 10, (0, 255, 255), -1)

    # Label
    label = f"[{class_id}] {class_name}  score={score}  [{stage}]"
    cv2.putText(vis, label, (x, max(0, y - 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    return vis


def write_data_yaml(class_names):
    """Write data.yaml that YOLO and Roboflow understand."""
    data = {
        "path"  : os.path.abspath(OUTPUT_FOLDER),
        "train" : "images",   # flat structure — Roboflow will split into train/val
        "val"   : "images",
        "nc"    : len(class_names),
        "names" : class_names,
    }
    yaml_path = os.path.join(OUTPUT_FOLDER, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


# ── Main ───────────────────────────────────────────────────────────────────

def main():

    # ── Sanity check ───────────────────────────────────────────────
    if not os.path.isdir(OLD_GP_PHOTOS_FOLDER):
        print(f"ERROR: Folder '{OLD_GP_PHOTOS_FOLDER}/' not found.")
        sys.exit(1)

    build_output_dirs()

    # ── Collect image files (sorted = ground-truth class order) ────
    all_files = sorted(
        f for f in os.listdir(OLD_GP_PHOTOS_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not all_files:
        print(f"ERROR: No images found in '{OLD_GP_PHOTOS_FOLDER}/'.")
        sys.exit(1)

    print(f"Found {len(all_files)} images in '{OLD_GP_PHOTOS_FOLDER}/':")
    for idx, fname in enumerate(all_files):
        name = PANEL_NAMES[idx] if idx < len(PANEL_NAMES) else f"Panel {idx}"
        print(f"  [{idx}] {fname}  ->  '{name}'")
    print()

    # ── Load PanelDetector (used for Stage 1 bounding boxes) ───────
    detector = PanelDetector(
        references_folder = OLD_GP_PHOTOS_FOLDER,
        panel_names       = PANEL_NAMES,
        min_inliers       = 10,
        ratio_threshold   = 0.75,
    )

    # ── Process each image ──────────────────────────────────────────
    success = []
    failed  = []

    for idx, fname in enumerate(all_files):

        # Ground-truth class ID = alphabetical file index
        class_id   = idx
        class_name = PANEL_NAMES[idx] if idx < len(PANEL_NAMES) else f"Panel {idx}"

        src_path = os.path.join(OLD_GP_PHOTOS_FOLDER, fname)
        frame    = cv2.imread(src_path)

        if frame is None:
            print(f"  SKIP   [{class_id}] {fname}  (cannot read)")
            failed.append((fname, class_id, "cannot read file"))
            continue

        # ── Run full detector to get bbox ───────────────────────────
        result = detector.detect(frame)

        if result["detected"]:
            bbox   = result["bbox"]
            score  = result["score"]
            stage  = result["stage"]

            # Warn if the detector's panel_id does not match the expected class_id.
            # This should never happen since each image matches itself, but
            # it is a useful sanity check.
            if result["panel_id"] != class_id:
                print(f"  WARN   [{class_id}] {fname}  "
                      f"detector returned panel_id={result['panel_id']} "
                      f"(expected {class_id}) — using file index as ground truth")

            # Write YOLO label with ground-truth class_id
            stem      = os.path.splitext(fname)[0]
            label_path  = os.path.join(OUTPUT_FOLDER, "labels",   stem + ".txt")
            image_path  = os.path.join(OUTPUT_FOLDER, "images",   fname)
            preview_path = os.path.join(OUTPUT_FOLDER, "previews", fname)

            write_yolo_label(label_path, class_id, bbox)
            shutil.copy2(src_path, image_path)

            preview = draw_bbox(frame, class_id, class_name, bbox, score, stage)
            cv2.imwrite(preview_path, preview)

            success.append((fname, class_id, class_name, bbox, score))
            print(f"  OK     [{class_id}] {fname:<20}  "
                  f"'{class_name}'  score={score}  [{stage}]")

        else:
            # Stage 1 failed AND Stage 2 fallback also failed.
            # Fall back to using the full image as the bounding box.
            # This gives YOLO the whole image labeled as that class — not ideal,
            # but better than skipping the image entirely.
            print(f"  NOBOX  [{class_id}] {fname}  "
                  f"{result['reason']}  -> using full-frame bbox as fallback")

            fallback_bbox = {
                "x": 0.0, "y": 0.0,
                "width": 1.0, "height": 1.0,
                "cx": 0.5, "cy": 0.5,
            }

            stem         = os.path.splitext(fname)[0]
            label_path   = os.path.join(OUTPUT_FOLDER, "labels",   stem + ".txt")
            image_path   = os.path.join(OUTPUT_FOLDER, "images",   fname)
            preview_path = os.path.join(OUTPUT_FOLDER, "previews", fname)

            write_yolo_label(label_path, class_id, fallback_bbox)
            shutil.copy2(src_path, image_path)

            # Draw full-frame box in orange to visually flag it as a fallback
            vis = frame.copy()
            fh, fw = vis.shape[:2]
            cv2.rectangle(vis, (5, 5), (fw - 5, fh - 5), (0, 140, 255), 6)
            cv2.putText(vis, f"[{class_id}] FALLBACK: full frame",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 140, 255), 4)
            cv2.imwrite(preview_path, vis)

            failed.append((fname, class_id, result["reason"]))

    # ── data.yaml ──────────────────────────────────────────────────
    names_used = [
        PANEL_NAMES[i] if i < len(PANEL_NAMES) else f"Panel {i}"
        for i in range(len(all_files))
    ]
    yaml_path = write_data_yaml(names_used)

    # ── Report ─────────────────────────────────────────────────────
    report = [
        "Panel YOLO Dataset Report",
        "=" * 50,
        f"Source folder  : {OLD_GP_PHOTOS_FOLDER}/",
        f"Output folder  : {OUTPUT_FOLDER}/",
        f"Total images   : {len(all_files)}",
        f"Labeled (OK)   : {len(success)}",
        f"Fallback bbox  : {len(failed)}",
        f"Classes        : {len(all_files)}",
        "",
        "Class mapping (class_id -> panel name -> filename)",
        "-" * 50,
    ]
    for idx, fname in enumerate(all_files):
        name = PANEL_NAMES[idx] if idx < len(PANEL_NAMES) else f"Panel {idx}"
        report.append(f"  {idx}  {name:<30}  {fname}")

    if failed:
        report += ["", "Fallback images (check previews — orange box = full frame used)", "-" * 50]
        for fname, class_id, reason in failed:
            report.append(f"  [{class_id}] {fname}  {reason}")

    report_path = os.path.join(OUTPUT_FOLDER, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report) + "\n")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"Done.  {len(success)}/{len(all_files)} images labeled correctly.")
    if failed:
        print(f"       {len(failed)} images used full-frame fallback (check previews).")
    print(f"\nOutput:")
    print(f"  Images   : {OUTPUT_FOLDER}/images/")
    print(f"  Labels   : {OUTPUT_FOLDER}/labels/")
    print(f"  Previews : {OUTPUT_FOLDER}/previews/  <- REVIEW THESE FIRST")
    print(f"  YAML     : {yaml_path}")
    print(f"  Report   : {report_path}")
    print(f"\nNext step — upload to Roboflow for augmentation:")
    print(f"  1. Go to roboflow.com -> New Project -> Object Detection")
    print(f"  2. Upload the contents of '{OUTPUT_FOLDER}/'")
    print(f"  3. Apply augmentations (brightness, perspective, noise, blur, crop)")
    print(f"  4. Generate dataset -> Export as YOLOv8 format")
    print(f"  5. Download the zip -> train with:")
    print(f"     yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
