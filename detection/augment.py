"""
detection/panel/augment.py
==========================
Generates augmented YOLO training datasets for BOTH the panel and obelisk
sources.  Each source gets its own output folder:

  data/processed/panel/augmented/
      images/train/   images/val/
      labels/train/   labels/val/
      data.yaml

  data/processed/obelisk/augmented/
      images/train/   images/val/
      labels/train/   labels/val/
      data.yaml

Memory-efficient design
-----------------------
Each source image is read from disk ONCE.  All of its augmented variants
are generated and saved immediately, so only one source image is in RAM
at any time.

The train / val split is pre-computed from a shuffled plan so the split
is random even though we process source images sequentially.

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

Usage (run from ObeliskScene root)
-----------------------------------
  python detection\\panel\\augment.py
"""

import cv2
import numpy as np
import os
import random
import yaml
from collections import defaultdict


DATASETS = [
    {
        "name"       : "panel",
        "input_images"  : "data/yolo/panel/images",
        "input_labels"  : "data/yolo/panel/labels",
        "data_yaml_src" : "data/yolo/panel/data.yaml",
        "output_dir"    : "data/processed/panel/augmented",
        "augs_per_image": 120,
    },
    {
        "name"       : "obelisk",
        "input_images"  : "data/yolo/obelisk/images",
        "input_labels"  : "data/yolo/obelisk/labels",
        "data_yaml_src" : "data/yolo/obelisk/data.yaml",
        "output_dir"    : "data/processed/obelisk/augmented",
        "augs_per_image": 15,
    },
]

SAVE_SIZE    = 640
VALIDATION_FRACTION = 0.20
SEED         = 42
MIN_BOX_AREA = 0.02   # skip augmented sample if any GT box covers < 2 % of the image
               # (prevents saving crops where the obelisk is barely visible)

# Helper Functions

def read_label(path: str) -> list:
    """
    Read a YOLO-format label file and return all bounding boxes inside it.

    Args: 
        path : Full path to the .txt label file.

    Returns:
        list of [class_id, cx, cy, w, h]
            - class_id : int, object class 
            - cx, cy   : float, bounding box centre as fraction of image width/height
            - w, h     : float, bounding box size  as fraction of image width/height
    """

    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 5:
                rows.append([int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return rows


def write_label(path: str, bboxes: list) -> None:
    """
    Write list of [class_id, cx, cy, w, h] in YOLO format.
    
    Args:
        path   : Full path to the output .txt label file.
        bboxes : list of [class_id, cx, cy, w, h]
            - class_id : int, object class
            - cx, cy   : float, bounding box centre as fraction of image width/height
            - w, h     : float, bounding box size  as fraction of image width/height
    """
    with open(path, "w") as f:
        for b in bboxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} "
                    f"{b[3]:.6f} {b[4]:.6f}\n")


#  Bounding-box geometry helpers

def yolo_to_corners(bbox: list, W: int, H: int) -> np.ndarray:
    """
    Convert a YOLO bounding box into its 4 pixel corner coordinates.

    Args:
        bbox : [cx, cy, w, h]
        W : Image width  in pixels.
        H : Image height in pixels.

    Returns:
        np.ndarray of shape (4, 2), dtype float32. Pixel coordinates [[TL], [TR], [BR], [BL]].
    """
    cx, cy, bw, bh = bbox
    cx *= W;  cy *= H;  bw *= W;  bh *= H
    return np.array([ [cx - bw/2, cy - bh/2], [cx + bw/2, cy - bh/2], [cx + bw/2, cy + bh/2],[cx - bw/2, cy + bh/2],], dtype=np.float32)


def corners_to_yolo(corners: np.ndarray, W: int, H: int):
    """
    Convert 4 pixel corner coordinates back into a YOLO bounding box.

    Args:
        corners : np.ndarray of shape (4, 2). Pixel (x, y) coordinates of the 4 corners after transformation.
        W : Image width  in pixels.
        H : Image height in pixels.

    Returns:
        [cx, cy, w, h]: normalised YOLO box, or None if the resulting box has zero area.
    """
    xs = np.clip(corners[:, 0], 0, W)   
    ys = np.clip(corners[:, 1], 0, H)   
    x1, x2 = xs.min(), xs.max()         
    y1, y2 = ys.min(), ys.max()        
    if x2 - x1 < 1 or y2 - y1 < 1:    
        return None
    return [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W,(y2 - y1) / H]              


#  Pixel-only augmentations 

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


#  Geometric augmentations 

def aug_rotate(img, bboxes):
    """Rotate ±15 degrees around image centre."""
    H, W  = img.shape[:2]
    angle = random.uniform(-15.0, 15.0)
    M     = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    new_bboxes = []
    for cls_id, *bbox in bboxes:
        corners = yolo_to_corners(bbox, W, H)
        ones    = np.ones((4, 1), dtype=np.float32)
        rot_c   = (M @ np.hstack([corners, ones]).T).T
        yolo    = corners_to_yolo(rot_c, W, H)
        if yolo:
            new_bboxes.append([cls_id] + yolo)
    return rotated, (new_bboxes if new_bboxes else bboxes)


def aug_perspective(img, bboxes):
    """Random perspective warp — perturb each corner by up to 8 %."""
    H, W = img.shape[:2]
    d    = int(min(W, H) * 0.08)
    src  = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst  = np.float32([ [random.randint(0, d),     random.randint(0, d)], [W - random.randint(0, d), random.randint(0, d)], [W - random.randint(0, d), H - random.randint(0, d)], [random.randint(0, d), H - random.randint(0, d)],])
    M      = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    new_bboxes = []
    for cls_id, *bbox in bboxes:
        corners = yolo_to_corners(bbox, W, H).reshape(-1, 1, 2)
        wc      = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
        yolo    = corners_to_yolo(wc, W, H)
        if yolo:
            new_bboxes.append([cls_id] + yolo)
    return warped, (new_bboxes if new_bboxes else bboxes)


def aug_crop(img, bboxes):
    """Zoom in by cropping 60-90 % of the image and resizing back."""
    H, W  = img.shape[:2]
    scale = random.uniform(0.60, 0.90)
    cW, cH = int(W * scale), int(H * scale)

    all_x1 = min((b[1] - b[3] / 2) * W for b in bboxes)
    all_y1 = min((b[2] - b[4] / 2) * H for b in bboxes)
    all_x2 = max((b[1] + b[3] / 2) * W for b in bboxes)
    all_y2 = max((b[2] + b[4] / 2) * H for b in bboxes)

    x0_min = max(0, int(all_x2) - cW)
    x0_max = max(0, min(W - cW, int(all_x1)))
    y0_min = max(0, int(all_y2) - cH)
    y0_max = max(0, min(H - cH, int(all_y1)))

    if x0_min > x0_max or y0_min > y0_max:
        return img, bboxes

    x0 = random.randint(x0_min, x0_max)
    y0 = random.randint(y0_min, y0_max)

    crop    = img[y0 : y0 + cH, x0 : x0 + cW]
    resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    sx, sy  = W / cW, H / cH

    new_bboxes = []
    for cls_id, cx, cy, bw, bh in bboxes:
        x1 = np.clip((cx - bw / 2) * W - x0, 0, cW) * sx
        y1 = np.clip((cy - bh / 2) * H - y0, 0, cH) * sy
        x2 = np.clip((cx + bw / 2) * W - x0, 0, cW) * sx
        y2 = np.clip((cy + bh / 2) * H - y0, 0, cH) * sy
        nw, nh = x2 - x1, y2 - y1
        if nw > 1 and nh > 1:
            new_bboxes.append([cls_id, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, nw / W, nh / H])
    return resized, (new_bboxes if new_bboxes else bboxes)


#  Augmentation pipeline

PIXEL_AUGS = [aug_brightness, aug_contrast, aug_noise, aug_blur,aug_grayscale, aug_hsv, aug_shadow]
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


def save_sample(img: np.ndarray, bboxes: list, img_out: str, lbl_out: str) -> None:
    """Resize to SAVE_SIZE × SAVE_SIZE, write JPEG + YOLO label."""
    sq = cv2.resize(img, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(img_out, sq, [cv2.IMWRITE_JPEG_QUALITY, 92])
    write_label(lbl_out, bboxes)


def run_augmentation(cfg: dict) -> bool:
    """
    Augment one dataset according to the given config dict.
    Returns True if augmentation ran, False if skipped.
    """
    name            = cfg["name"]
    input_images    = cfg["input_images"]
    input_labels    = cfg["input_labels"]
    data_yaml_src   = cfg["data_yaml_src"]
    output_dir      = cfg["output_dir"]
    augs_per_image  = cfg["augs_per_image"]

    # ── Skip gracefully if input folders are missing ─────────────────────────
    missing = []
    if not os.path.isdir(input_images):
        missing.append(f"images folder '{input_images}'")
    if not os.path.isdir(input_labels):
        missing.append(f"labels folder '{input_labels}'")
    if not os.path.exists(data_yaml_src):
        missing.append(f"data.yaml '{data_yaml_src}'")

    if missing:
        print(f"\n Skip  {name.upper()} missing input(s):")
        for m in missing:
            print(f"✗  {m}")
        print(f" Run the labelling step first, then re-run augment.py.")
        return False

    print(f"  Dataset : {name.upper()}")
    print(f"  Input   : {input_images}")
    print(f"  Output  : {output_dir}")

    # Create output folders
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    img_files = sorted( f for f in os.listdir(input_images) if f.lower().endswith((".jpg", ".jpeg", ".png")) )
    if not img_files:
        print(f"  ERROR: No images found in '{input_images}/' — skipping.")
        return

    total = len(img_files) * (1 + augs_per_image)
    print(f"  Source images   : {len(img_files)}")
    print(f"  Augs per image  : {augs_per_image}")
    print(f"  Total planned   : {total}  (inc. originals)")
    print(f"  Save size       : {SAVE_SIZE} x {SAVE_SIZE} px")
    print()

    # ── Pre-compute train / val assignment ──────────────────────────────────
    plan = []
    for fi, fname in enumerate(img_files):
        plan.append((fi, -1))                          # original
        for i in range(augs_per_image):
            plan.append((fi, i))                       # augmented

    random.shuffle(plan)
    n_val = int(len(plan) * VALIDATION_FRACTION)

    assignment = defaultdict(list)   # fi -> [(aug_idx, split), ...]
    for rank, (fi, aug_idx) in enumerate(plan):
        split = "val" if rank < n_val else "train"
        assignment[fi].append((aug_idx, split))

    # ── Process each source image once ──────────────────────────────────────
    saved = {"train": 0, "val": 0}

    for fi, fname in enumerate(img_files):
        stem     = os.path.splitext(fname)[0]
        img_path = os.path.join(input_images, fname)
        lbl_path = os.path.join(input_labels, stem + ".txt")

        img    = cv2.imread(img_path)
        bboxes = read_label(lbl_path)

        if img is None:
            print(f"Skip {fname} (cannot read)")
            continue
        if not bboxes:
            print(f"Skip {fname} (no label)")
            continue

        entries = assignment[fi]

        for aug_idx, split in entries:
            if aug_idx == -1:
                out_img, out_bboxes = img, bboxes
                out_stem = f"{stem}_orig"
            else:
                out_img, out_bboxes = augment_once(img, bboxes)
                out_stem = f"{stem}_aug{aug_idx:04d}"

            # Skip if any GT box shrank below the minimum area threshold.
            # This happens when a crop augmentation pushes the object to the
            # image edge, leaving only a sliver of the bounding box visible.
            # The model still detects the full shape → low IoU → false error.
            if any(b[3] * b[4] < MIN_BOX_AREA for b in out_bboxes):
                continue

            img_out = os.path.join(output_dir, "images", split, out_stem + ".jpg")
            lbl_out = os.path.join(output_dir, "labels", split, out_stem + ".txt")
            save_sample(out_img, out_bboxes, img_out, lbl_out)
            saved[split] += 1

        n_tr = sum(1 for _, s in entries if s == "train")
        n_vl = sum(1 for _, s in entries if s == "val")
        print(f"  {fname:<28}  class {bboxes[0][0]}  "
              f"train={n_tr:3d}  val={n_vl:3d}")

    # ── Write data.yaml ─────────────────────────────────────────────────────
    if os.path.exists(data_yaml_src):
        with open(data_yaml_src) as f:
            data = yaml.safe_load(f)
        data["path"]  = os.path.abspath(output_dir)
        data["train"] = "images/train"
        data["val"]   = "images/val"
        yaml_out = os.path.join(output_dir, "data.yaml")
        with open(yaml_out, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"\n  data.yaml  ->  {yaml_out}")
    else:
        print(f"\n  WARNING: {data_yaml_src} not found — data.yaml not written.")

    print(f"\n  Done.  Train: {saved['train']}  |  Val: {saved['val']}")
    print(f"  Output folder: {output_dir}/")
    return True


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Datasets to process: {[d['name'] for d in DATASETS]}")

    ran     = 0
    skipped = 0
    for cfg in DATASETS:
        ok = run_augmentation(cfg)
        if ok:
            ran += 1
        else:
            skipped += 1

    print(f"Finished.  {ran} dataset(s) augmented, {skipped} skipped.")


if __name__ == "__main__":
    main()
