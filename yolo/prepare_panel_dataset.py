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

OLD_GP_PHOTOS_FOLDER = "data/raw/panel"
OUTPUT_FOLDER        = "data/yolo/panel"

PANEL_NAMES = [
    "Architecture and Ornament Studies",       
    "College History Text",                       
    "Corinthian Capital and Industrial Machinery", 
    "Gate and Mining Cart",                       
    "Hydraulics and Tile Work",                  
    "Mechanical Engineering",                      
    "Steam Engine and Polytechnique Relief",       
    "Survey Instrument and Classical Facades",    
    "University Dome Building",                    
]



def build_output_dirs():
    os.makedirs(os.path.join(OUTPUT_FOLDER, "images"),   exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "labels"),   exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "previews"), exist_ok=True)


def write_yolo_label(label_path, class_id, bbox):
    cx = float(bbox["cx"])
    cy = float(bbox["cy"])
    w  = float(bbox["width"])
    h  = float(bbox["height"])

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.001, min(1.0, w))
    h  = max(0.001, min(1.0, h))

    with open(label_path, "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def draw_bbox(frame, class_id, class_name, bbox, score, stage):
    vis = frame.copy()
    fh, fw = vis.shape[:2]

    x  = int(bbox["x"]* fw)
    y  = int(bbox["y"] * fh)
    x2 = int((bbox["x"] + bbox["width"])  * fw)
    y2 = int((bbox["y"] + bbox["height"]) * fh)
    cx_px = int(bbox["cx"] * fw)
    cy_px = int(bbox["cy"] * fh)

    cv2.rectangle(vis, (x, y), (x2, y2), (0, 255, 0), 4)
    cv2.circle(vis, (cx_px, cy_px), 10, (0, 255, 255), -1)
    label = f"[{class_id}] {class_name}  score={score}  [{stage}]"
    cv2.putText(vis, label, (x, max(0, y - 14)),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    return vis


def write_data_yaml(class_names):
    data = {
        "path"  : os.path.abspath(OUTPUT_FOLDER),
        "train": "images",  
        "val" : "images",
        "nc": len(class_names),
        "names" : class_names,
    }
    yaml_path = os.path.join(OUTPUT_FOLDER, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def main():
    if not os.path.isdir(OLD_GP_PHOTOS_FOLDER):
        print(f"ERROR: Folder '{OLD_GP_PHOTOS_FOLDER}/' not found.")
        sys.exit(1)

    build_output_dirs()

    all_files = sorted(f for f in os.listdir(OLD_GP_PHOTOS_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    if not all_files:
        print(f"ERROR: No images found in '{OLD_GP_PHOTOS_FOLDER}/'.")
        sys.exit(1)

    print(f"Found {len(all_files)} images in '{OLD_GP_PHOTOS_FOLDER}/':")
    for idx, fname in enumerate(all_files):
        name = PANEL_NAMES[idx] if idx < len(PANEL_NAMES) else f"Panel {idx}"
        print(f"  [{idx}] {fname}  ->  '{name}'")
    print()

    detector = PanelDetector(references_folder = OLD_GP_PHOTOS_FOLDER,panel_names= PANEL_NAMES, min_inliers  = 10,ratio_threshold   = 0.75,)

    success = []
    failed  = []

    for idx, fname in enumerate(all_files):
        class_id   = idx
        class_name = PANEL_NAMES[idx] if idx < len(PANEL_NAMES) else f"Panel {idx}"

        src_path = os.path.join(OLD_GP_PHOTOS_FOLDER, fname)
        frame    = cv2.imread(src_path)

        if frame is None:
            print(f"skip [{class_id}] {fname} ")
            failed.append((fname, class_id, "cannot read file"))
            continue

        result = detector.detect(frame)

        if result["detected"]:
            bbox = result["bbox"]
            score = result["score"]
            stage = result["stage"]
            if result["panel_id"] != class_id:
                print(f" warn[{class_id}] {fname},detector returned panel_id={result['panel_id']} ")

            stem= os.path.splitext(fname)[0]
            label_path = os.path.join(OUTPUT_FOLDER, "labels",   stem + ".txt")
            image_path = os.path.join(OUTPUT_FOLDER, "images",   fname)
            preview_path = os.path.join(OUTPUT_FOLDER, "previews", fname)

            write_yolo_label(label_path, class_id, bbox)
            shutil.copy2(src_path, image_path)

            preview = draw_bbox(frame, class_id, class_name, bbox, score, stage)
            cv2.imwrite(preview_path, preview)

            success.append((fname, class_id, class_name, bbox, score))
            print(f" ok[{class_id}] {fname:<20} ")

        else:
            print(f"  NOBOX  [{class_id}] {fname}")

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

    names_used = [PANEL_NAMES[i] if i < len(PANEL_NAMES) else f"Panel {i}" for i in range(len(all_files))]
    yaml_path = write_data_yaml(names_used)

if __name__ == "__main__":
    main()
