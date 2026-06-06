import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH   = "yolo/inshallahlast.pt"   

VAL_IMG_DIR  = "data/processed/panel/augmented/images/val"
VAL_LBL_DIR  = "data/processed/panel/augmented/labels/val"

OUT_DIR      = "data/processed/panel/error_analysis"
FP_DIR       = os.path.join(OUT_DIR, "false_positives")
FN_DIR       = os.path.join(OUT_DIR, "false_negatives")
TP_DIR       = os.path.join(OUT_DIR, "true_positives")

CONF_THRESH  = 0.25    
IOU_THRESH   = 0.30   

PANEL_CLASSES = list(range(9))

for d in [FP_DIR, FN_DIR, TP_DIR]:
    os.makedirs(d, exist_ok=True)

model = YOLO(MODEL_PATH)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def load_gt_boxes(label_path, img_w, img_h):
    """Load YOLO-format GT boxes → list of (cls, [x1, y1, x2, y2]) in pixel coords."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls not in PANEL_CLASSES:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((cls, [x1, y1, x2, y2]))
    return boxes


def iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (aA + aB - inter)


def draw_boxes(img, gt_boxes, pred_boxes, pred_confs):
    """Draw GT (green) and predicted (red) boxes on the image."""
    vis = img.copy()
    for box in gt_boxes:
        cv2.rectangle(vis,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 255, 0), 2)
        cv2.putText(vis, "GT", (int(box[0]), int(box[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for box, conf in zip(pred_boxes, pred_confs):
        cv2.rectangle(vis,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 0, 255), 2)
        cv2.putText(vis, f"pred {conf:.2f}", (int(box[0]), int(box[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
img_paths = (glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg")) +
             glob.glob(os.path.join(VAL_IMG_DIR, "*.png")))

fp_count = fn_count = tp_count = 0

for img_path in img_paths:
    fname     = os.path.basename(img_path)
    stem      = os.path.splitext(fname)[0]
    lbl_path  = os.path.join(VAL_LBL_DIR, stem + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] Cannot read {img_path}")
        continue
    H, W = img.shape[:2]

    # ── Ground truth ──────────────────────────────────────────────────────
    gt_boxes = load_gt_boxes(lbl_path, W, H)

    # ── Prediction ────────────────────────────────────────────────────────
    results = model(img_path, conf=CONF_THRESH, verbose=False, device='cpu')[0]

    pred_boxes = []
    pred_confs = []
    pred_clses = []
    for box in results.boxes:
        cls = int(box.cls[0].item())
        if cls not in PANEL_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())
        pred_boxes.append([x1, y1, x2, y2])
        pred_confs.append(conf)
        pred_clses.append(cls)

    # ── Match predictions → GT using IoU ──────────────────────────────────
    matched_gt   = set()
    matched_pred = set()
    best_ious    = {}   # pi → best IoU score (for diagnostics)

    gt_coords = [box for (_, box) in gt_boxes]

    for pi, pb in enumerate(pred_boxes):
        best_iou = 0
        best_gi  = -1
        for gi, gb in enumerate(gt_coords):
            if gi in matched_gt:
                continue
            score = iou(pb, gb)
            if score > best_iou:
                best_iou = score
                best_gi  = gi
        best_ious[pi] = best_iou
        if best_iou >= IOU_THRESH:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    # ── Classify this image ───────────────────────────────────────────────
    fp_preds = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    fn_gts   = [i for i in range(len(gt_boxes))   if i not in matched_gt]
    tp_preds = list(matched_pred)

    vis = draw_boxes(img, gt_coords, pred_boxes, pred_confs)

    if fp_preds:
        out = os.path.join(FP_DIR, fname)
        cv2.imwrite(out, vis)
        fp_count += 1
        iou_info = [f"IoU={best_ious[i]:.2f}" for i in fp_preds]
        print(f"  [FP] {fname}  — {len(fp_preds)} unmatched detection(s)  "
              f"cls={[pred_clses[i] for i in fp_preds]}  "
              f"conf={[f'{pred_confs[i]:.2f}' for i in fp_preds]}  {iou_info}")

    if fn_gts:
        out = os.path.join(FN_DIR, fname)
        cv2.imwrite(out, vis)
        fn_count += 1
        missed_cls = [gt_boxes[i][0] for i in fn_gts]
        print(f"  [FN] {fname}  — {len(fn_gts)} missed GT box(es)  cls={missed_cls}"
              f"  (best IoU found: {[f'{best_ious.get(i, 0):.2f}' for i in range(len(pred_boxes))]})")

    if tp_preds and not fp_preds and not fn_gts:
        out = os.path.join(TP_DIR, fname)
        cv2.imwrite(out, vis)
        tp_count += 1

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
total = len(img_paths)
print("\n" + "=" * 50)
print(f"Validation set : {total} images")
print(f"True Positives : {tp_count}  ({100*tp_count/max(1,total):.1f}%)")
print(f"False Positives: {fp_count}  ({100*fp_count/max(1,total):.1f}%)")
print(f"False Negatives: {fn_count}  ({100*fn_count/max(1,total):.1f}%)")
print(f"\nImages saved to: {OUT_DIR}/")
print(f"  false_positives/ — {fp_count} images  (green=GT, red=wrong prediction)")
print(f"  false_negatives/ — {fn_count} images  (green=GT missed by model)")
print(f"  true_positives/  — {tp_count} images  (correct detections, reference)")
