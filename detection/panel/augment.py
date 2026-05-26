"""
augment_panel_dataset.py
========================
Generates a large augmented YOLO training dataset from the 9 labeled
panel images.  No Roboflow or external augmentation library is used --
everything is implemented from scratch with NumPy + OpenCV.

Memory-efficient design
-----------------------
Each source image is read from disk ONCE.  All of its augmented variants
are generated and saved immediately, so only one source image is in RAM
at any time (never accumulating all 1000+ images at once).

The train / val split is pre-computed from a shuffled plan so the split
is still random even though we process source images sequentially.

Augmentations
-------------
Pixel-only  (bbox unchanged):
  brightness   -- multiply pixels by a random factor  0.4 - 1.6
  contrast     -- linear stretch  alpha * img + beta
  noise        -- additive Gaussian noise
  blur         -- Gaussian blur with a random kernel
  hsv_jitter   -- random hue / saturation / value shift
  grayscale    -- convert to gray and back to BGR
  shadow       -- apply a random dark vertical band

Geometric  (bbox corners transformed, new axis-aligned bbox computed):
  rotate       -- rotate +-15 degrees around image centre
  perspective  -- random perspective warp (up to 8 % corner shift)
  crop         -- zoom into the image (60-90 %), resize back

Output
------
  panel_augmented_dataset/
      images/train/      .jpg  (80 %)
      images/val/               (20 %)
      labels/train/      .txt  (YOLO format)
      labels/val/
      data.yaml

Usage (run from ObeliskScene root)
-----------------------------------
  python panel\\augment_panel_dataset.py
"""

import cv2
import numpy as np
import os
import random
import yaml
from collections import defaultdict

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

INPUT_IMAGES  = "data/yolo/panel/images"
INPUT_LABELS  = "data/yolo/panel/labels"
DATA_YAML_SRC = "data/yolo/panel/data.yaml"
OUTPUT_DIR    = "data/processed/panel/augmented"

AUGMENTATIONS_PER_IMAGE = 120   # per source  =>  9 x 120 + 9 originals = 1089 total
SAVE_SIZE               = 640   # all outputs are saved at 640 x 640
VAL_FRACTION            = 0.20  # 20 % goes to the validation split
SEED                    = 42

# ---------------------------------------------------------------------------
#  Label helpers
# ---------------------------------------------------------------------------

def read_label(path: str) -> list:
    """Return list of [class_id, cx, cy, w, h]  (all float / int)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 5:
                rows.append([int(p[0]),
                              float(p[1]), float(p[2]),
                              float(p[3]), float(p[4])])
    return rows


def write_label(path: str, bboxes: list) -> None:
    """Write list of [class_id, cx, cy, w, h] in YOLO format."""
    with open(path, "w") as f:
        for b in bboxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} "
                    f"{b[3]:.6f} {b[4]:.6f}\n")

# ---------------------------------------------------------------------------
#  Bounding-box geometry helpers
# ---------------------------------------------------------------------------

def yolo_to_corners(bbox: list, W: int, H: int) -> np.ndarray:
    """
    [cx, cy, w, h] normalised  ->  4 pixel corners  (TL, TR, BR, BL).
    """
    cx, cy, bw, bh = bbox
    cx *= W;  cy *= H;  bw *= W;  bh *= H
    return np.array([
        [cx - bw/2, cy - bh/2],
        [cx + bw/2, cy - bh/2],
        [cx + bw/2, cy + bh/2],
        [cx - bw/2, cy + bh/2],
    ], dtype=np.float32)


def corners_to_yolo(corners: np.ndarray, W: int, H: int):
    """
    4 pixel corners  ->  axis-aligned [cx, cy, w, h] normalised.
    Returns None if the resulting bbox has zero area.
    """
    xs = np.clip(corners[:, 0], 0, W)
    ys = np.clip(corners[:, 1], 0, H)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [(x1 + x2) / 2 / W,
            (y1 + y2) / 2 / H,
            (x2 - x1) / W,
            (y2 - y1) / H]

# ---------------------------------------------------------------------------
#  Pixel-only augmentations  (bbox is returned unchanged)
# ---------------------------------------------------------------------------

def aug_brightness(img, bboxes):
    factor = random.uniform(0.4, 1.6)
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out, bboxes


def aug_contrast(img, bboxes):
    alpha = random.uniform(0.5, 1.8)
    beta  = random.randint(-40, 40)
    out = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    return out, bboxes


def aug_noise(img, bboxes):
    std   = random.uniform(5, 35)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    out   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, bboxes


def aug_blur(img, bboxes):
    k   = random.choice([3, 5, 7, 9, 11])
    out = cv2.GaussianBlur(img, (k, k), 0)
    return out, bboxes


def aug_grayscale(img, bboxes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out, bboxes


def aug_hsv(img, bboxes):
    h_shift = random.randint(-18, 18)
    s_scale = random.uniform(0.5, 1.5)
    v_scale = random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, bboxes


def aug_shadow(img, bboxes):
    """Dark vertical band simulating a cast shadow."""
    out = img.copy().astype(np.float32)
    W   = img.shape[1]
    x1  = random.randint(0, W // 2)
    x2  = random.randint(W // 2, W)
    out[:, x1:x2] *= random.uniform(0.20, 0.60)
    return np.clip(out, 0, 255).astype(np.uint8), bboxes

# ---------------------------------------------------------------------------
#  Geometric augmentations  (bbox corners must be transformed)
# ---------------------------------------------------------------------------

def aug_rotate(img, bboxes):
    """
    Rotate image +-15 degrees around centre.

    How the bbox is updated:
      1. Convert [cx, cy, w, h] -> 4 pixel corners.
      2. Apply the exact same rotation matrix M to those corners.
      3. Take the axis-aligned envelope (min/max of rotated x, y).
      4. Normalise back to [0, 1].
    """
    H, W  = img.shape[:2]
    angle = random.uniform(-15.0, 15.0)
    M     = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)

    rotated = cv2.warpAffine(img, M, (W, H),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    new_bboxes = []
    for cls_id, *bbox in bboxes:
        corners = yolo_to_corners(bbox, W, H)          # (4, 2)
        ones    = np.ones((4, 1), dtype=np.float32)
        rot_c   = (M @ np.hstack([corners, ones]).T).T  # (4, 2)
        yolo    = corners_to_yolo(rot_c, W, H)
        if yolo:
            new_bboxes.append([cls_id] + yolo)

    return rotated, (new_bboxes if new_bboxes else bboxes)


def aug_perspective(img, bboxes):
    """
    Random perspective warp: perturb each image corner by up to 8 %
    of the smaller image dimension.

    How the bbox is updated:
      1. Convert [cx, cy, w, h] -> 4 pixel corners.
      2. Apply cv2.perspectiveTransform with the same homography H.
      3. Axis-aligned envelope + normalise.
    """
    H, W = img.shape[:2]
    d    = int(min(W, H) * 0.08)

    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([
        [random.randint(0, d),     random.randint(0, d)],
        [W - random.randint(0, d), random.randint(0, d)],
        [W - random.randint(0, d), H - random.randint(0, d)],
        [random.randint(0, d),     H - random.randint(0, d)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, (W, H),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
    new_bboxes = []
    for cls_id, *bbox in bboxes:
        corners = yolo_to_corners(bbox, W, H).reshape(-1, 1, 2)
        wc      = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
        yolo    = corners_to_yolo(wc, W, H)
        if yolo:
            new_bboxes.append([cls_id] + yolo)

    return warped, (new_bboxes if new_bboxes else bboxes)


def aug_crop(img, bboxes):
    """
    Zoom in by cropping 60-90 % of the image and resizing back.
    The crop origin is constrained so all bboxes remain fully visible.

    How the bbox is updated:
      pixel_in_crop  = original_pixel - crop_origin
      pixel_in_resized = pixel_in_crop * (output_size / crop_size)
      new_yolo = pixel_in_resized / output_size
    """
    H, W  = img.shape[:2]
    scale = random.uniform(0.60, 0.90)
    cW, cH = int(W * scale), int(H * scale)

    # Pixel extents of all bboxes (format [cls, cx, cy, w, h])
    all_x1 = min((b[1] - b[3] / 2) * W for b in bboxes)
    all_y1 = min((b[2] - b[4] / 2) * H for b in bboxes)
    all_x2 = max((b[1] + b[3] / 2) * W for b in bboxes)
    all_y2 = max((b[2] + b[4] / 2) * H for b in bboxes)

    # Crop origin constraints so every bbox stays inside the crop
    x0_min = max(0, int(all_x2) - cW)
    x0_max = max(0, min(W - cW, int(all_x1)))
    y0_min = max(0, int(all_y2) - cH)
    y0_max = max(0, min(H - cH, int(all_y1)))

    if x0_min > x0_max or y0_min > y0_max:
        return img, bboxes      # bbox too large for this crop scale

    x0 = random.randint(x0_min, x0_max)
    y0 = random.randint(y0_min, y0_max)

    crop    = img[y0 : y0 + cH, x0 : x0 + cW]
    resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

    sx, sy = W / cW, H / cH    # scale-back factors

    new_bboxes = []
    for cls_id, cx, cy, bw, bh in bboxes:
        x1 = np.clip((cx - bw / 2) * W - x0, 0, cW) * sx
        y1 = np.clip((cy - bh / 2) * H - y0, 0, cH) * sy
        x2 = np.clip((cx + bw / 2) * W - x0, 0, cW) * sx
        y2 = np.clip((cy + bh / 2) * H - y0, 0, cH) * sy
        nw, nh = x2 - x1, y2 - y1
        if nw > 1 and nh > 1:
            new_bboxes.append([cls_id,
                                (x1 + x2) / 2 / W,
                                (y1 + y2) / 2 / H,
                                nw / W, nh / H])

    return resized, (new_bboxes if new_bboxes else bboxes)

# ---------------------------------------------------------------------------
#  Augmentation pipeline
# ---------------------------------------------------------------------------

PIXEL_AUGS = [aug_brightness, aug_contrast, aug_noise, aug_blur,
              aug_grayscale,  aug_hsv,      aug_shadow]

GEO_AUGS   = [aug_rotate, aug_perspective, aug_crop]


def augment_once(img: np.ndarray, bboxes: list) -> tuple:
    """
    Random augmentation chain:
      Step 1 -- 1 to 3 pixel augmentations (always applied).
      Step 2 -- one geometric augmentation with 70 % probability.
    """
    out_img    = img.copy()
    out_bboxes = [list(b) for b in bboxes]

    n_pix = random.randint(1, 3)
    for fn in random.sample(PIXEL_AUGS, min(n_pix, len(PIXEL_AUGS))):
        out_img, out_bboxes = fn(out_img, out_bboxes)

    if random.random() < 0.70:
        fn = random.choice(GEO_AUGS)
        out_img, out_bboxes = fn(out_img, out_bboxes)

    return out_img, out_bboxes

# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------

def build_dirs() -> None:
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)


def save_sample(img: np.ndarray, bboxes: list,
                img_out: str, lbl_out: str) -> None:
    """Resize to SAVE_SIZE x SAVE_SIZE, write JPEG + YOLO label."""
    sq = cv2.resize(img, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(img_out, sq, [cv2.IMWRITE_JPEG_QUALITY, 92])
    write_label(lbl_out, bboxes)

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    build_dirs()

    img_files = sorted(
        f for f in os.listdir(INPUT_IMAGES)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not img_files:
        print(f"ERROR: No images found in '{INPUT_IMAGES}/'")
        return

    total = len(img_files) * (1 + AUGMENTATIONS_PER_IMAGE)
    print(f"Source images : {len(img_files)}")
    print(f"Aug per image : {AUGMENTATIONS_PER_IMAGE}")
    print(f"Total planned : {total}  (inc. originals)")
    print(f"Save size     : {SAVE_SIZE} x {SAVE_SIZE} px")
    print()

    # ------------------------------------------------------------------
    #  Pre-compute train / val assignment
    #  Each entry: (file_index, aug_index)   aug_index = -1 => original
    # ------------------------------------------------------------------
    plan = []
    for fi, fname in enumerate(img_files):
        plan.append((fi, -1))                              # original
        for i in range(AUGMENTATIONS_PER_IMAGE):
            plan.append((fi, i))                           # augmented

    random.shuffle(plan)
    n_val = int(len(plan) * VAL_FRACTION)

    # Group by file index so we read each source image only once
    assignment = defaultdict(list)   # fi -> [(aug_idx, split), ...]
    for rank, (fi, aug_idx) in enumerate(plan):
        split = "val" if rank < n_val else "train"
        assignment[fi].append((aug_idx, split))

    # ------------------------------------------------------------------
    #  Process each source image once, generate + save immediately
    # ------------------------------------------------------------------
    saved = {"train": 0, "val": 0}

    for fi, fname in enumerate(img_files):
        stem     = os.path.splitext(fname)[0]
        img_path = os.path.join(INPUT_IMAGES, fname)
        lbl_path = os.path.join(INPUT_LABELS,  stem + ".txt")

        img    = cv2.imread(img_path)
        bboxes = read_label(lbl_path)

        if img is None:
            print(f"  SKIP  {fname}  (cannot read)")
            continue
        if not bboxes:
            print(f"  SKIP  {fname}  (no label)")
            continue

        entries = assignment[fi]   # list of (aug_idx, split) for this image

        for aug_idx, split in entries:
            if aug_idx == -1:                   # original, no augmentation
                out_img, out_bboxes = img, bboxes
                out_stem = f"{stem}_orig"
            else:
                out_img, out_bboxes = augment_once(img, bboxes)
                out_stem = f"{stem}_aug{aug_idx:04d}"

            img_out = os.path.join(OUTPUT_DIR, "images", split,
                                   out_stem + ".jpg")
            lbl_out = os.path.join(OUTPUT_DIR, "labels", split,
                                   out_stem + ".txt")
            save_sample(out_img, out_bboxes, img_out, lbl_out)
            saved[split] += 1

        n_train = sum(1 for _, s in entries if s == "train")
        n_val_f = sum(1 for _, s in entries if s == "val")
        print(f"  {fname:<22}  class {bboxes[0][0]}  "
              f"train={n_train:3d}  val={n_val_f:3d}")

    # ------------------------------------------------------------------
    #  Write data.yaml
    # ------------------------------------------------------------------
    if os.path.exists(DATA_YAML_SRC):
        with open(DATA_YAML_SRC) as f:
            data = yaml.safe_load(f)
        data["path"]  = os.path.abspath(OUTPUT_DIR)
        data["train"] = "images/train"
        data["val"]   = "images/val"
        yaml_out = os.path.join(OUTPUT_DIR, "data.yaml")
        with open(yaml_out, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"\ndata.yaml  ->  {yaml_out}")
    else:
        yaml_out = "(not written -- source data.yaml not found)"

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Done.  Train: {saved['train']}  |  Val: {saved['val']}")
    print(f"Dataset at   : {OUTPUT_DIR}/")
    print(f"{'='*60}")
    print(f"Next step -- train YOLOv8:")
    print(f"  pip install ultralytics")
    print(f"  yolo train model=yolov8n.pt \\")
    print(f"        data={os.path.abspath(OUTPUT_DIR)}/data.yaml \\")
    print(f"        epochs=100 imgsz=640 batch=16")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
