# Overview

# Project Structure

## Panel Processing Pipeline

The panel detection system follows this architecture:

```
detection/panel/pipeline.py
    ├── PanelDetector class
    ├── Stage 1: Localization
        └── Find black metal frame and extract panel region(homography warp)
    └── Stage 2: Identification
        └── Match panel against references using SIFT + RANSAC
        └── Output: Which panel + bounding box coordinates
        ↓ (Algorithm used by)
yolo/prepare_panel_dataset.py
    ├── Applies PanelDetector to 9 raw photos
    ├── Generates YOLO-format training labels
    └── Output: data/yolo/panel/ (9 labeled images)
        ↓ (Data augmented by)
detection/panel/augment.py
    ├── Creates 120 variants per image
    ├── Applies geometric & pixel augmentations
    |── Generates train/val split (80/20)
    └── Output: data/processed/panel/augmented/ (~1,089 training images)
```

## Directory Layout

```
heritage-scene-recognition/
├── detection/panel/          # Classical IP algorithms
│   ├── pipeline.py          # PanelDetector class (localization + identification)
│   └── augment.py           # Dataset augmentation script
├── yolo/
│   └── prepare_panel_dataset.py   # Script to generate YOLO labels using PanelDetector
├── data/
│   ├── raw/panel/           # Input: 9 raw panel photos
│   ├── yolo/panel/          # Generated: 9 labeled images from Stage 1
│   └── processed/panel/augmented/  # Generated: ~1,089 augmented training images
└── panel/                   # Legacy panel detection output folder
```

# Pipeline — Classical IP (Steps 1–9)

# Pipeline — YOLO Fine-tuning

# How to Run

## Prerequisites

Install required dependencies:

```powershell
pip install -r requirements.txt
```

This installs:

- `opencv-contrib-python` - Computer vision library with SIFT feature matching
- `numpy` - Numerical computing
- `pyyaml` - YAML configuration file handling

## Panel Detection Pipeline

### Processing Stages

**Stage 1: Generate YOLO Dataset from Raw Photos**

Uses `PanelDetector` (classical IP algorithm) to label raw images:

```powershell
python yolo/prepare_panel_dataset.py
```

**What it does:**

1. Loads 9 panel photos from `data/raw/panel/`
2. Runs PanelDetector on each image:
   - Localization: Finds and extracts the panel region
   - Identification: Determines which of the 9 panels it is
3. Generates YOLO-format labels from the detected bounding boxes
4. Creates annotated preview images for visual verification
5. Outputs to `data/yolo/panel/` (9 labeled images + metadata)

**Output:**

```
data/yolo/panel/
├── images/       (9 panel images)
├── labels/       (9 YOLO .txt files)
├── previews/     (9 annotated images)
├── data.yaml     (YOLO dataset config)
└── report.txt    (processing report)
```

---

**Stage 2: Augment Dataset for Training**

Creates a large training dataset by applying augmentations:

```powershell
python detection/panel/augment.py
```

**What it does:**

1. Reads the 9 labeled panels from `data/yolo/panel/` (output from Stage 1)
2. Applies 120 augmentations per source image:
   - Pixel-level: brightness, contrast, noise, blur, HSV jitter
   - Geometric: rotation, perspective warp, crop/zoom
3. Generates ~1,089 total images with automatic 80/20 train/val split
4. Outputs to `data/processed/panel/augmented/` (ready for YOLO training)

**Output:**

```
data/processed/panel/augmented/
├── images/
│   ├── train/    (~870 augmented images)
│   └── val/      (~219 augmented images)
├── labels/
│   ├── train/    (YOLO labels)
│   └── val/
└── data.yaml     (ready for YOLO training)
```

### Full Pipeline Workflow

Run the panel pipeline in sequence from the project root:

```powershell
# Step 1: Generate labeled dataset using PanelDetector
python yolo/prepare_panel_dataset.py

# Step 2: Augment the labeled dataset for training
python detection/panel/augment.py
```

**What happens:**

1. `prepare_panel_dataset.py` uses `PanelDetector` to label 9 raw images → produces `data/yolo/panel/`
2. `augment.py` reads from `data/yolo/panel/` and creates augmented variants → produces `data/processed/panel/augmented/`

The final augmented dataset in `data/processed/panel/augmented/` is ready for YOLO model training.

### Design Notes

- All scripts work from the project root directory
- They automatically locate data files and resolve import paths
- Scripts can be run in sequence or independently (as long as prior steps completed)
- PanelDetector uses classical image processing (SIFT + RANSAC) for robust panel detection

# Results

# Team
