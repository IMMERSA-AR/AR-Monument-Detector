import cv2
import os
import shutil
import numpy as np
from pathlib import Path


RAW_DIR    = "data/raw/obelisk"
OUT_IMAGES = "data/yolo/obelisk/images"
OUT_LABELS = "data/yolo/obelisk/labels"
DEBUG_DIR  = "data/processed/obelisk/debug"

# Images that were detected wrongly (pipeline found something but box is wrong)
WRONG_LABELS = [
    "IMG_1357.jpg",
    "IMG_1358.jpg",
    "IMG_1363.jpg",
    "IMG_1376.jpg",  
    "IMG_6850.jpg",   
    "IMG_6851.jpg",   
    "IMG_6873.jpg",
    "IMG_6874.jpg",
]

# Images that were missed entirely by the pipeline
MISSED = [
    "IMG_1355.jpg",
    "IMG_1360.jpg",
    "IMG_1361.jpg",
    "IMG_1372.jpg",
    "IMG_1373.jpg",
    "IMG_6860.jpg",
    "IMG_6861.jpg",
    "IMG_6862.jpg",
    "IMG_6863.jpg",
    "IMG_6864.jpg",
    "IMG_6868.jpg"
]

DISPLAY_WIDTH  = 900   
DISPLAY_HEIGHT = 1200   


state = {
    "pt1": None,   
    "pt2": None,  
    "drawing": False,
    "done": False,
}


def mouse_cb(event, x, y, flags, param):
    s = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if s["pt1"] is None:
            s["pt1"] = (x, y)
            s["drawing"] = True
        else:
            s["pt2"] = (x, y)
            s["drawing"] = False
            s["done"] = True
    elif event == cv2.EVENT_MOUSEMOVE and s["drawing"]:
        s["pt2"] = (x, y)


def draw_canvas(base_img, s, scale, existing_label=None):
    """Draw the base image with existing label (yellow) and current box (green)."""
    vis = base_img.copy()
    dh, dw = vis.shape[:2]

    if existing_label is not None:
        cx, cy, w, h = existing_label
        x1 = int((cx - w / 2) * dw)
        y1 = int((cy - h / 2) * dh)
        x2 = int((cx + w / 2) * dw)
        y2 = int((cy + h / 2) * dh)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(vis, "OLD", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Current box being drawn
    if s["pt1"] is not None and s["pt2"] is not None:
        p1 = (min(s["pt1"][0], s["pt2"][0]), min(s["pt1"][1], s["pt2"][1]))
        p2 = (max(s["pt1"][0], s["pt2"][0]), max(s["pt1"][1], s["pt2"][1]))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        cv2.putText(vis, "NEW", (p1[0], max(0, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif s["pt1"] is not None:
        cv2.circle(vis, s["pt1"], 6, (0, 255, 0), -1)

    return vis


def load_display_image(img_path):
    """Load and scale image to fit display window, return (display_img, scale)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, 1.0
    h, w = img.shape[:2]
    scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img, scale


def load_existing_label(stem):
    """Return (cx, cy, w, h) if label file exists, else None."""
    p = Path(OUT_LABELS) / (stem + ".txt")
    if p.exists():
        parts = p.read_text().strip().split()
        if len(parts) == 5:
            return tuple(float(v) for v in parts[1:])
    return None


def box_to_yolo(pt1, pt2, dh, dw):
    """Convert two display-coords points to YOLO cx,cy,w,h (0–1)."""
    x1 = min(pt1[0], pt2[0]) / dw
    y1 = min(pt1[1], pt2[1]) / dh
    x2 = max(pt1[0], pt2[0]) / dw
    y2 = max(pt1[1], pt2[1]) / dh
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    return cx, cy, w, h


def save_label(stem, cx, cy, w, h):
    os.makedirs(OUT_LABELS, exist_ok=True)
    p = Path(OUT_LABELS) / (stem + ".txt")
    p.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    print(f"  Saved: {p.name}   0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")


def ensure_image_in_dataset(fname):
    """Copy image from raw/ to yolo/images/ if not already there."""
    dst = Path(OUT_IMAGES) / fname
    if not dst.exists():
        src = Path(RAW_DIR) / fname
        if src.exists():
            os.makedirs(OUT_IMAGES, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Copied {fname}, added to {OUT_IMAGES}/")
        else:
            print(f" WARNING: {src} not found in raw dir!")


def label_image(fname, mode="missed"):
    """
    Open one image for interactive labelling.
    Returns True = saved, False = skipped/deleted, None = quit.
    """
    stem = Path(fname).stem
    img_path = Path(OUT_IMAGES) / fname

    if mode == "missed":
        ensure_image_in_dataset(fname)

    if not img_path.exists():
        print(f"Skip: {img_path} not found")
        return False

    display_img, scale = load_display_image(img_path)
    if display_img is None:
        print(f"Skip: cannot read {img_path}")
        return False

    existing = load_existing_label(stem)
    dh, dw = display_img.shape[:2]

    # Reset state
    for k in ("pt1", "pt2", "drawing", "done"):
        state[k] = None if k in ("pt1", "pt2") else False

    win = "Obelisk Labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    cv2.setMouseCallback(win, mouse_cb, state)

    tag = "WRONG" if mode == "wrong" else "MISSED"
    print(f"\n[{tag}] {fname}")
    print("  Click top-left then bottom-right of the OBELISK.")
    print("  ENTER/S=save   R=reset   D=delete/skip   ESC/Q=quit")

    while True:
        canvas = draw_canvas(display_img, state, scale, existing)
        # Instructions overlay
        cv2.putText(canvas,
                    f"{fname}  |  ENTER=save  R=reset  D=delete  ESC=quit",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(canvas,
                    "Click corner1, then corner2 to draw GREEN box",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 200), 1)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, ord('s'), ord('S')):   # ENTER or S
            if state["pt1"] is not None and state["pt2"] is not None:
                cx, cy, w, h = box_to_yolo(state["pt1"], state["pt2"], dh, dw)
                w = max(0.005, w)
                h = max(0.005, h)
                save_label(stem, cx, cy, w, h)
                cv2.destroyWindow(win)
                return True
            else:
                print("  Draw a box first (click two corners).")

        elif key in (ord('r'), ord('R')):      # R = reset
            state["pt1"] = None
            state["pt2"] = None
            state["drawing"] = False
            state["done"] = False
            print("  Box reset.")

        elif key in (ord('d'), ord('D')):      # D = delete/skip
            lp = Path(OUT_LABELS) / (stem + ".txt")
            ip = Path(OUT_IMAGES) / fname
            if lp.exists():
                lp.unlink()
                print(f"  Deleted label: {lp.name}")
            if ip.exists():
                ip.unlink()
                print(f"  Deleted image: {ip.name}")
            cv2.destroyWindow(win)
            return False

        elif key in (27, ord('q'), ord('Q')):  # ESC / Q = quit
            print("  Quitting.")
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)
    return False


def main():
    print(" Obelisk Manual Labeler")
    print(f" Wrong labels to fix : {len(WRONG_LABELS)},  Missed images: {len(MISSED)},  Total: {len(WRONG_LABELS) + len(MISSED)}")

    saved  = 0
    skipped = 0

    print("Wrong detections")
    for fname in WRONG_LABELS:
        result = label_image(fname, mode="wrong")
        if result is True:
            saved += 1
        elif result is False:
            skipped += 1
        else: 
            break

    print("\n Missed images")
    for fname in MISSED:
        result = label_image(fname, mode="missed")
        if result is True:
            saved += 1
        elif result is False:
            skipped += 1
        else:  
            break

    cv2.destroyAllWindows()
    total_labels = len(list(Path(OUT_LABELS).glob("*.txt")))
    total_images = len(list(Path(OUT_IMAGES).glob("*.jpg")))

    print(f"  Done.  Saved={saved}  Skipped/deleted={skipped}")
    print(f"  Dataset now: {total_images} images, {total_labels} labels")


if __name__ == "__main__":
    main()
