import sys
import os
import glob
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.obelisk.pipeline import label_all

def generate_previews(yolo_dir: str) -> None:

    images_dir= os.path.join(yolo_dir, "images")
    labels_dir = os.path.join(yolo_dir, "labels")
    previews_dir = os.path.join(yolo_dir, "previews")
    os.makedirs(previews_dir, exist_ok=True)

    img_paths = (glob.glob(os.path.join(images_dir, "*.jpg")) +glob.glob(os.path.join(images_dir, "*.png")))

    if not img_paths:
        print(f"No images found in '{images_dir}'.")
        return

    saved = 0
    for img_path in sorted(img_paths):
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        lbl_path = os.path.join(labels_dir, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        vis  = img.copy()

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, bw, bh = map(float, parts)
                    x1 = int((cx - bw / 2) * W)
                    y1 = int((cy - bh / 2) * H)
                    x2 = int((cx + bw / 2) * W)
                    y2 = int((cy + bh / 2) * H)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(vis, "obelisk",(x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "NO LABEL", (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        out_path = os.path.join(previews_dir, fname)
        cv2.imwrite(out_path, vis)
        saved += 1

    print(f"Previews generated")


if __name__ == "__main__":
    YOLO_DIR = "data/yolo/obelisk"

    label_all(input_dir="data/raw/obelisk", output_dir=YOLO_DIR, debug_dir="data/processed/obelisk/debug", save_debug=True, verbose=True)
    generate_previews(YOLO_DIR)
    print("\nDone.")
