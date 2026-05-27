import os
import zipfile
import yaml
import subprocess
import torch
import glob
import shutil

from google.colab import drive
from google.colab import files
from IPython.display import Image, display

drive.mount('/content/drive')


# Unzip Dataset 


ZIP_PATH    = "/content/drive/MyDrive/panel_augmented_dataset.zip"
EXTRACT_DIR = "/content/panel_dataset"

os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    for member in z.infolist():
        # PowerShell Compress-Archive writes backslashes — fix them to forward slashes
        fixed_name  = member.filename.replace('\\', '/')
        target_path = os.path.join(EXTRACT_DIR, fixed_name)

        if fixed_name.endswith('/'):          # it's a directory entry
            os.makedirs(target_path, exist_ok=True)
        else:                                 # it's a file
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with z.open(member) as src, open(target_path, 'wb') as dst:
                dst.write(src.read())

print("Extraction complete.")

# ── Auto-discover the folder that contains data.yaml ────────────────────────
def find_dataset_dir(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if "data.yaml" in filenames:
            return dirpath
    return None

DATASET_DIR = find_dataset_dir(EXTRACT_DIR)

if DATASET_DIR is None:
    raise FileNotFoundError(
        f"data.yaml not found inside {EXTRACT_DIR}.\n"
        f"Top-level contents: {os.listdir(EXTRACT_DIR)}"
    )

print("Dataset found at :", DATASET_DIR)
print("Contents         :", sorted(os.listdir(DATASET_DIR)))



#Fix data.yaml paths for Colab

yaml_path = os.path.join(DATASET_DIR, "data.yaml")

with open(yaml_path) as f:
    data = yaml.safe_load(f)

# Update path to the Colab location
data["path"]  = DATASET_DIR
data["train"] = "images/train"
data["val"]   = "images/val"

with open(yaml_path, "w") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print("data.yaml updated:")
print(f"  path  = {data['path']}")
print(f"  train = {data['train']}")
print(f"  val   = {data['val']}")
print(f"  nc    = {data['nc']}")
print(f"  names = {data['names']}")


# Install Ultralytics 
subprocess.run(["pip", "install", "ultralytics", "-q"], check=True)
print("Ultralytics installed.")


# Verify GPU 

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    print("WARNING: No GPU detected. Go to Runtime -> Change runtime type -> T4 GPU")


# TRAIN 
#
#  Model choice:
#    yolov8n.pt  = nano   (~3.2M params)  -- fastest, least accurate
#    yolov8s.pt  = small  (~11M params)   -- good balance  <-- recommended
#    yolov8m.pt  = medium (~25M params)   -- more accurate, slower
#
#  For a GP demo with 9 classes and ~1000 images, yolov8s is ideal.
#
from ultralytics import YOLO

MODEL    = "yolov8s.pt"   # pretrained on COCO (80 classes), fine-tuned here
DATA     = yaml_path
EPOCHS   = 100
IMG_SIZE = 640
BATCH    = 16             # reduce to 8 if Colab runs out of memory

model = YOLO(MODEL)

results = model.train(
    data      = DATA,
    epochs    = EPOCHS,
    imgsz     = IMG_SIZE,
    batch     = BATCH,
    patience  = 20,        # stop early if no improvement for 20 epochs
    device    = 0,         # GPU 0
    project   = "/content/runs",
    name      = "panel_detector",
    exist_ok  = True,
    verbose   = True,
)

print("\nTraining complete!")
print("Best weights saved at: /content/runs/panel_detector/weights/best.pt")

#Evaluate on validation set 
#
#  This prints per-class AP (Average Precision) and overall mAP.
#  mAP50 > 0.90  = excellent -- ready for Unity
#  mAP50 0.75-0.90 = good -- will work for most panels
#  mAP50 < 0.75  = needs more data or label fixes
#
best_model = YOLO("/content/runs/panel_detector/weights/best.pt")
val_results = best_model.val(data=DATA, imgsz=IMG_SIZE, device=0)

print("\n=== Validation Results ===")
print(f"mAP50      : {val_results.box.map50:.4f}")
print(f"mAP50-95   : {val_results.box.map:.4f}")
print(f"Precision  : {val_results.box.mp:.4f}")
print(f"Recall     : {val_results.box.mr:.4f}")


# Show confusion matrix 

cm_path = glob.glob("/content/runs/panel_detector/confusion_matrix_normalized.png")
if cm_path:
    display(Image(cm_path[0]))
else:
    print("Confusion matrix not found - check /content/runs/panel_detector/")


#  Export to ONNX (for Unity Sentis) 
#
#  opset=12  is required for Unity Sentis compatibility.
#  simplify=True  reduces the model size and speeds up inference.
#
best_model.export(
    format   = "onnx",
    imgsz    = IMG_SIZE,
    opset    = 12,
    simplify = True,
)

onnx_path = "/content/runs/panel_detector/weights/best.onnx"
print(f"\nONNX model exported to: {onnx_path}")

size_mb = os.path.getsize(onnx_path) / 1e6
print(f"File size: {size_mb:.1f} MB")


# Download both .pt and .onnx to your machine 

files.download("/content/runs/panel_detector/weights/best.pt")
files.download("/content/runs/panel_detector/weights/best.onnx")
print("Downloads started. Check your browser's download folder.")


#  Save to Google Drive


SAVE_DIR = "/content/drive/MyDrive/panel_model_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

shutil.copy("/content/runs/panel_detector/weights/best.pt",
            f"{SAVE_DIR}/best.pt")
shutil.copy("/content/runs/panel_detector/weights/best.onnx",
            f"{SAVE_DIR}/best.onnx")

# Save the full results folder
shutil.copytree(
    "/content/runs/panel_detector",
    f"{SAVE_DIR}/training_run",
    dirs_exist_ok=True
)

print(f"All outputs saved to Google Drive: {SAVE_DIR}/")
print("  best.pt      <- PyTorch model (for further training)")
print("  best.onnx    <- ONNX model   (for Unity Sentis)")
print("  training_run/ <- all metrics, confusion matrix, plots")
