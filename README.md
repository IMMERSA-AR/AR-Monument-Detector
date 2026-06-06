# AR Monument Detector

This repository contains the full data pipeline for training a YOLOv8 object detection model that recognises **obelisks** and **information panels** in real-world photos.
The trained model is exported to ONNX and used inside a **Meta Quest Augmented Reality app** (Unity Sentis) to trigger AR experiences when the user points their headset at the obelisk or one of its surrounding panels.

---

## What this repo does

```
Raw photos
    ↓
Classical image processing  →  Auto-label images
    ↓
Manual correction (for obelisk)
    ↓
Data augmentation
    ↓
Export zips  →  Upload to Google Drive
    ↓
YOLOv8 training
    ↓
ONNX export  →  Drop into Unity Sentis
```

The pipeline handles two types of objects:

| Class                                       | ID  | Source                                        |
| ------------------------------------------- | --- | --------------------------------------------- |
| Obelisk                                     | 0   | 69 real photos, auto-labelled by classical CV |
| College History Text                        | 1   | Panel photos, manually labelled               |
| University Dome Building                    | 2   | Panel photos, manually labelled               |
| Mechanical Engineering                      | 3   | Panel photos, manually labelled               |
| Gate and Mining Cart                        | 4   | Panel photos, manually labelled               |
| Hydraulics and Tile Work                    | 5   | Panel photos, manually labelled               |
| Architecture and Ornament Studies           | 6   | Panel photos, manually labelled               |
| Survey Instrument and Classical Facades     | 7   | Panel photos, manually labelled               |
| Corinthian Capital and Industrial Machinery | 8   | Panel photos, manually labelled               |
| Steam Engine and Polytechnique Relief       | 9   | Panel photos, manually labelled               |

---

## Repository structure

```
ObeliskScene/
│
├── data/
│   ├── raw/
│   │   ├── obelisk/
│   │   └── panel/
│   │
│   ├── yolo/
│   │   ├── obelisk/
│   │   │   ├── images/         <- copies of detected obelisk photos
│   │   │   ├── labels/         <- YOLO .txt labels
│   │   │   ├── previews/       <- annotated images for visual review
│   │   │   └── data.yaml
│   │   │
│   │   └── panel/
│   │       ├── images/         <- copies of detected panel photos
│   │       ├── labels/         <- YOLO .txt labels
│   │       ├── previews/       <- annotated images for visual review
│   │       └── data.yaml
│   │
│   └── processed/
│       ├── obelisk/
│       │   ├── augmented/
│       │   ├── debug/
│       │   └── obelisk_augmented_dataset.zip
│       │
│       └── panel/
│           ├── augmented/
│           └── panel_augmented_dataset.zip
│
├── detection/
│   ├── obelisk/
│   │   └── pipeline.py
│   ├── panel/
│   │   └── pipeline.py
│   └── augment.py
│
└── yolo/
    ├── prepare_obelisk_dataset.py
    ├── prepare_panel_dataset.py
    ├── manual_label.py
    ├── export_zip.py
    ├── train.py
    └── find_errors.py
```

---

## Requirements

```bash
pip install opencv-contrib-python numpy pyyaml ultralytics
```

For training (Kaggle):

```bash
pip install ultralytics onnxsim
```

---

## Step-by-step pipeline

### Step 1 - Prepare raw data

Place all raw photos in the correct folders:

- Obelisk photos: `data/raw/obelisk/`
- Panel photos: `data/raw/panel/`

### Step 2a - Auto-label obelisk images

**File:** `yolo/prepare_obelisk_dataset.py`

Runs the classical detection pipeline on all photos in `data/raw/obelisk/`.
Uses computer vision (not ML) to find the obelisk tip and shaft, then writes a YOLO bounding box for each image.

**Output:**

```
data/yolo/obelisk/images/      <- copies of the photos
data/yolo/obelisk/labels/      <- one .txt label per photo  (class 0)
data/yolo/obelisk/previews/    <- green bounding box drawn on each photo
data/processed/obelisk/debug/  <- intermediate debug images
```

---

### Step 2b - Auto-label panel images

**File:** `yolo/prepare_panel_dataset.py`

Runs the panel detector on all 9 photos in `data/raw/panel/`.
Class ID is assigned by alphabetical file order.

**Output:**

```
data/yolo/panel/images/      <- copies of the panel photos
data/yolo/panel/labels/      <- one .txt label per photo  (classes 0-8)
data/yolo/panel/previews/    <- annotated images for visual review
data/yolo/panel/data.yaml    <- class names ready for training
```

---

### Step 3 - Fix wrong or missed labels

**File:** `yolo/manual_label.py`

Run this for any image where the green box in Step 2 was wrong or missing.
Lets you draw the correct bounding box by hand and saves it to `data/yolo/obelisk/labels/`.

---

### Step 4 - Augment both datasets

**File:** `detection/augment.py`

Generates augmented training data for **both** obelisk and panel datasets.
Augmentations include: brightness, contrast, noise, blur, grayscale, HSV jitter, shadow, rotation, perspective warp, and crop.

| Dataset | Source images | Augs per image | Total  |
| ------- | ------------- | -------------- | ------ |
| Panel   | 9             | x 120          | ~1,089 |
| Obelisk | 69            | x 15           | ~1,104 |

Both datasets are balanced to ~1,100 images so neither class dominates training.

**Output:**

```
data/processed/obelisk/augmented/   images/train,val   labels/train,val   data.yaml
data/processed/panel/augmented/     images/train,val   labels/train,val   data.yaml
```

---

### Step 5 - Export zips

**File:** `yolo/export_zip.py`

Creates zip files with Unix forward-slash paths so they extract correctly on Linux (Google Colab).
PowerShell's built-in Compress-Archive writes Windows backslashes which break on Linux - this script avoids that.

**Output:**

```
data/processed/obelisk/obelisk_augmented_dataset.zip
data/processed/panel/panel_augmented_dataset.zip
```

---

### Step 6 - Upload to Google Drive

Upload both zips to Google Drive at exactly these paths:

```
MyDrive/GP/obelisk_augmented_dataset.zip
MyDrive/GP/panel_augmented_dataset.zip
```

---

### Step 7 - Train on Google Colab

**File:** `yolo/train.py`

1. Open Kaggle
2. Set **Runtime -> Change runtime type -> T4 GPU**
3. Upload and run `yolo/train.py` cell by cell

<!-- The script will:

- Mount Google Drive and extract both zips
- Merge the two datasets, remapping panel classes by +1 so they do not collide with obelisk (class 0)
- Train a **YOLOv8s** model for 100 epochs across all 10 classes
- Validate and print mAP, Precision, Recall
- Export `best.onnx` (opset 12, simplified - compatible with Unity Sentis)
- Save everything to `MyDrive/GP/combined_model_outputs/` -->

**Outputs saved to Google Drive:**

```
combined_model_outputs/
    best.pt        <- PyTorch weights (for fine-tuning)
    best.onnx      <- ONNX model     (drag into Unity Assets)
    classes.txt    <- class names in order
    training_run/  <- metrics, plots, confusion matrix
```

---

### Step 8 - Verify the model (optional but recommended)

**File:** `yolo/find_errors.py`

1. Download `best.pt` from `MyDrive/GP/combined_model_outputs/`
2. Place it at `yolo/best.pt`
3. Run:

Runs inference on the validation set and saves annotated images showing:

- **Green box** = ground truth label
- **Red box** = model prediction

```
data/processed/obelisk/error_analysis/
    false_positives/   <- model detected obelisk where there is none
    false_negatives/   <- real obelisk that the model missed
    true_positives/    <- correct detections (reference)
```

---
