"""
yolo/train.py
=============
Kaggle training script — trains a SINGLE YOLOv8 model to detect
BOTH obelisks AND information panels in one pass, then exports to ONNX
for Unity Sentis.

Class layout (10 classes total)
--------------------------------
  0  : obelisk
  1  : College History Text
  2  : University Dome Building
  3  : Mechanical Engineering
  4  : Gate and Mining Cart
  5  : Hydraulics and Tile Work
  6  : Architecture and Ornament Studies
  7  : Survey Instrument and Classical Facades
  8  : Corinthian Capital and Industrial Machinery
  9  : Steam Engine and Polytechnique Relief

How to use (Kaggle)
--------------------
1.  Upload your two zip files as a Kaggle dataset:
      obelisk_augmented_dataset.zip
      panel_augmented_dataset.zip
2.  In your Kaggle notebook, add that dataset as input.
3.  Update OBELISK_ZIP and PANEL_ZIP below to match the input paths.
4.  Set accelerator to GPU (Settings → Accelerator → GPU T4 x2).
5.  Run all cells.
6.  Download best.pt and best.onnx from the Output tab when done.
"""

import os
import zipfile
import shutil
import subprocess
import glob
import yaml
import torch

# ─────────────────────────────────────────────────────────────────────────────
#  0.  PATHS  — update these to match your Kaggle dataset input paths
# ─────────────────────────────────────────────────────────────────────────────

# Path to your uploaded zip files on Kaggle.
# After adding your dataset in Kaggle, the files appear under /kaggle/input/<dataset-slug>/
# Example: if your dataset slug is "gp-obelisk-panel", the paths would be:
#   /kaggle/input/gp-obelisk-panel/obelisk_augmented_dataset.zip
#   /kaggle/input/gp-obelisk-panel/panel_augmented_dataset.zip

OBELISK_ZIP = "/kaggle/input/gp-dataset/obelisk_augmented_dataset.zip"
PANEL_ZIP   = "/kaggle/input/gp-dataset/panel_augmented_dataset.zip"

# Working directories — everything in /kaggle/working/ is saved as output
WORK_DIR           = "/kaggle/working/combined_dataset"
OBELISK_DIR        = "/kaggle/working/obelisk_raw"
PANEL_DIR          = "/kaggle/working/panel_raw"
RUNS_DIR           = "/kaggle/working/runs"
OUTPUT_DIR         = "/kaggle/working"   # final model files go here

COMBINED_IMG_TRAIN = f"{WORK_DIR}/images/train"
COMBINED_IMG_VAL   = f"{WORK_DIR}/images/val"
COMBINED_LBL_TRAIN = f"{WORK_DIR}/labels/train"
COMBINED_LBL_VAL   = f"{WORK_DIR}/labels/val"

for d in [WORK_DIR, OBELISK_DIR, PANEL_DIR, RUNS_DIR,
          COMBINED_IMG_TRAIN, COMBINED_IMG_VAL,
          COMBINED_LBL_TRAIN, COMBINED_LBL_VAL]:
    os.makedirs(d, exist_ok=True)

print("Kaggle paths ready.")
print(f"  Obelisk zip : {OBELISK_ZIP}")
print(f"  Panel zip   : {PANEL_ZIP}")
print(f"  Working dir : {WORK_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
#  1.  INSTALL ULTRALYTICS
# ─────────────────────────────────────────────────────────────────────────────
subprocess.run(["pip", "install", "ultralytics", "onnxsim", "-q"], check=True)
print("Ultralytics + onnxsim installed.")


# ─────────────────────────────────────────────────────────────────────────────
#  2.  VERIFY GPU
# ─────────────────────────────────────────────────────────────────────────────
print("\nGPU check:")
print("  CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU :", torch.cuda.get_device_name(0))
    print("  VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    print("  WARNING: No GPU detected — go to Settings → Accelerator → GPU T4")


# ─────────────────────────────────────────────────────────────────────────────
#  3.  EXTRACT ZIPS
# ─────────────────────────────────────────────────────────────────────────────
def extract_zip(zip_path, dest):
    """Extract a zip, normalising backslash paths written by PowerShell."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"\nZip not found: {zip_path}\n"
            f"Make sure you added your dataset to this notebook in Kaggle.\n"
            f"Go to: Notebook → Add data → Your datasets → select your zip dataset."
        )
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.infolist():
            fixed  = member.filename.replace('\\', '/')
            target = os.path.join(dest, fixed)
            if fixed.endswith('/'):
                os.makedirs(target, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(member) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
    print(f"  Extracted: {zip_path}  →  {dest}")


print("\nExtracting datasets...")
extract_zip(OBELISK_ZIP, OBELISK_DIR)
extract_zip(PANEL_ZIP,   PANEL_DIR)


# ─────────────────────────────────────────────────────────────────────────────
#  4.  LOCATE data.yaml IN EACH EXTRACTED TREE
# ─────────────────────────────────────────────────────────────────────────────
def find_dataset_dir(root):
    for dirpath, _, filenames in os.walk(root):
        if "data.yaml" in filenames:
            return dirpath
    return None


obelisk_ds = find_dataset_dir(OBELISK_DIR)
panel_ds   = find_dataset_dir(PANEL_DIR)

if obelisk_ds is None:
    raise FileNotFoundError(f"data.yaml not found inside {OBELISK_DIR}")
if panel_ds is None:
    raise FileNotFoundError(f"data.yaml not found inside {PANEL_DIR}")

print(f"\nObelisk dataset : {obelisk_ds}")
print(f"Panel dataset   : {panel_ds}")

with open(os.path.join(obelisk_ds, "data.yaml")) as f:
    obelisk_yaml = yaml.safe_load(f)
with open(os.path.join(panel_ds, "data.yaml")) as f:
    panel_yaml = yaml.safe_load(f)

print(f"Obelisk classes : {obelisk_yaml.get('names')}")
print(f"Panel classes   : {panel_yaml.get('names')}")


# ─────────────────────────────────────────────────────────────────────────────
#  5.  MERGE DATASETS  (remap panel class IDs by +1)
# ─────────────────────────────────────────────────────────────────────────────
# Combined class mapping:
#   obelisk  class 0       →  stays 0
#   panel    class 0..N-1  →  becomes 1..N

PANEL_CLASS_OFFSET = 1


def copy_images(src_img_dir, dst_img_dir, prefix=""):
    """Copy every image from src to dst, optionally adding a filename prefix."""
    copied = 0
    for img in (glob.glob(os.path.join(src_img_dir, "*.jpg")) +
                glob.glob(os.path.join(src_img_dir, "*.png"))):
        fname = prefix + os.path.basename(img)
        shutil.copy2(img, os.path.join(dst_img_dir, fname))
        copied += 1
    return copied


def copy_labels(src_lbl_dir, dst_lbl_dir, class_offset=0, prefix=""):
    """
    Copy YOLO .txt labels from src to dst.
    If class_offset != 0, add it to every class_id.
    """
    copied = 0
    for txt in glob.glob(os.path.join(src_lbl_dir, "*.txt")):
        fname    = prefix + os.path.basename(txt)
        out_path = os.path.join(dst_lbl_dir, fname)
        with open(txt) as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                parts    = line.split()
                class_id = int(parts[0]) + class_offset
                f_out.write(f"{class_id} {' '.join(parts[1:])}\n")
        copied += 1
    return copied


print("\nMerging datasets...")

# Obelisk — class 0, no remapping
n = copy_images(f"{obelisk_ds}/images/train", COMBINED_IMG_TRAIN, prefix="obelisk_")
print(f"  Obelisk train images : {n}")
n = copy_labels(f"{obelisk_ds}/labels/train", COMBINED_LBL_TRAIN, class_offset=0, prefix="obelisk_")
print(f"  Obelisk train labels : {n}")

n = copy_images(f"{obelisk_ds}/images/val", COMBINED_IMG_VAL, prefix="obelisk_")
print(f"  Obelisk val   images : {n}")
n = copy_labels(f"{obelisk_ds}/labels/val", COMBINED_LBL_VAL, class_offset=0, prefix="obelisk_")
print(f"  Obelisk val   labels : {n}")

# Panels — classes 0-8 remapped to 1-9
n = copy_images(f"{panel_ds}/images/train", COMBINED_IMG_TRAIN, prefix="panel_")
print(f"  Panel   train images : {n}")
n = copy_labels(f"{panel_ds}/labels/train", COMBINED_LBL_TRAIN, class_offset=PANEL_CLASS_OFFSET, prefix="panel_")
print(f"  Panel   train labels : {n}")

n = copy_images(f"{panel_ds}/images/val", COMBINED_IMG_VAL, prefix="panel_")
print(f"  Panel   val   images : {n}")
n = copy_labels(f"{panel_ds}/labels/val", COMBINED_LBL_VAL, class_offset=PANEL_CLASS_OFFSET, prefix="panel_")
print(f"  Panel   val   labels : {n}")

# Verify counts match
print()
for split in ("train", "val"):
    imgs = len(glob.glob(f"{WORK_DIR}/images/{split}/*"))
    lbls = len(glob.glob(f"{WORK_DIR}/labels/{split}/*.txt"))
    status = "✓" if imgs == lbls else "← MISMATCH, check zips"
    print(f"  {split:5s}  images={imgs}  labels={lbls}  {status}")


# ─────────────────────────────────────────────────────────────────────────────
#  6.  WRITE COMBINED data.yaml
# ─────────────────────────────────────────────────────────────────────────────
panel_names    = panel_yaml.get("names", [])
combined_names = ["obelisk"] + panel_names   # 10 classes total

combined_yaml = {
    "path"  : WORK_DIR,
    "train" : "images/train",
    "val"   : "images/val",
    "nc"    : len(combined_names),
    "names" : combined_names,
}

COMBINED_YAML_PATH = f"{WORK_DIR}/data.yaml"
with open(COMBINED_YAML_PATH, "w") as f:
    yaml.dump(combined_yaml, f, default_flow_style=False, sort_keys=False)

print("\nCombined data.yaml:")
print(f"  nc = {combined_yaml['nc']}")
for i, name in enumerate(combined_yaml["names"]):
    print(f"  [{i}] {name}")


# ─────────────────────────────────────────────────────────────────────────────
#  7.  TRAIN
# ─────────────────────────────────────────────────────────────────────────────
from ultralytics import YOLO

MODEL    = "yolov8s.pt"   # change to yolo26s.pt or any other version if needed
EPOCHS   = 100
IMG_SIZE = 640
BATCH    = 16             # lower to 8 if you get out-of-memory errors
RUN_NAME = "obelisk_panel_detector"

print(f"\nStarting training...")
print(f"  Model   : {MODEL}")
print(f"  Epochs  : {EPOCHS}")
print(f"  Classes : {len(combined_names)}")

model = YOLO(MODEL)

results = model.train(
    data     = COMBINED_YAML_PATH,
    epochs   = EPOCHS,
    imgsz    = IMG_SIZE,
    batch    = BATCH,
    patience = 20,          # early-stop after 20 epochs without improvement
    device   = 0,           # GPU
    project  = RUNS_DIR,
    name     = RUN_NAME,
    exist_ok = True,
    verbose  = True,
)

WEIGHTS_DIR = f"{RUNS_DIR}/{RUN_NAME}/weights"
print(f"\nTraining complete!")
print(f"  Best weights: {WEIGHTS_DIR}/best.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  8.  VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning validation...")
best_model  = YOLO(f"{WEIGHTS_DIR}/best.pt")
val_results = best_model.val(data=COMBINED_YAML_PATH, imgsz=IMG_SIZE, device=0)

print("\n=== Validation Results ===")
print(f"  mAP50    : {val_results.box.map50:.4f}")
print(f"  mAP50-95 : {val_results.box.map:.4f}")
print(f"  Precision: {val_results.box.mp:.4f}")
print(f"  Recall   : {val_results.box.mr:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  9.  EXPORT TO ONNX  (Unity Sentis compatible)
# ─────────────────────────────────────────────────────────────────────────────
print("\nExporting to ONNX...")
best_model.export(
    format   = "onnx",
    imgsz    = IMG_SIZE,
    opset    = 12,       # opset 12 is safest for Unity Sentis
    simplify = True,
)

ONNX_PATH = f"{WEIGHTS_DIR}/best.onnx"
print(f"  ONNX exported : {ONNX_PATH}")
print(f"  File size     : {os.path.getsize(ONNX_PATH)/1e6:.1f} MB")
print(f"  Input  shape  : [1, 3, {IMG_SIZE}, {IMG_SIZE}]")
print(f"  Output classes: {combined_names}")


# ─────────────────────────────────────────────────────────────────────────────
#  10. COPY OUTPUTS TO /kaggle/working/  (shown in Kaggle Output tab)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCopying outputs...")

shutil.copy(f"{WEIGHTS_DIR}/best.pt",   f"{OUTPUT_DIR}/best.pt")
shutil.copy(ONNX_PATH,                   f"{OUTPUT_DIR}/best.onnx")

# Save class names so Unity can load them alongside the model
with open(f"{OUTPUT_DIR}/classes.txt", "w") as f:
    for name in combined_names:
        f.write(name + "\n")

# Copy full training run (plots, metrics, confusion matrix)
shutil.copytree(
    f"{RUNS_DIR}/{RUN_NAME}",
    f"{OUTPUT_DIR}/training_run",
    dirs_exist_ok=True,
)

print("\n" + "=" * 55)
print("All done! Files saved to Kaggle Output tab:")
print("  best.pt        ← PyTorch weights (for fine-tuning)")
print("  best.onnx      ← ONNX model      (drag into Unity Assets)")
print("  classes.txt    ← class names in order (load in Unity)")
print("  training_run/  ← metrics, plots, confusion matrix")
print("\nDownload them from the Output tab on the right side of Kaggle.")
print("=" * 55)
