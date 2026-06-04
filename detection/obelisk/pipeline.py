import cv2
import os
import shutil
import numpy as np
from pathlib import Path


INPUT_DIR  = "data/raw/obelisk"
OUTPUT_DIR = "data/yolo/obelisk"
DEBUG_DIR  = "data/processed/obelisk/debug"

RESIZE_WIDTH = 1024     
MAX_PEAK_WIDTH_FRAC = 0.25 
PEAK_DELTA_Y_FRAC = 0.07   
MAX_TIP_Y_FRAC = 0.55  
EDGE_MARGIN_FRAC = 0.08   
MIN_RAW_OBELISK_WIDTH = 20
MAX_ATTEMPTS = 8 
FILL_THRESH  = 0.45        
FLOOR_MARGIN = 0.20        
MIN_STONE = 4   
HOUGH_THRESH = 30
HOUGH_MIN_LEN = 50
HOUGH_MAX_GAP = 20
HOUGH_VERT_TOL = 22    
CLASS_ID = 0         

# Step 1: Non Stone Mask 

def build_non_stone_mask(img_bgr):

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    high_blue_mask = np.array([140, 255, 255], np.uint8)
    low_blue_mask = np.array([100,  50, 130], np.uint8)
    low_white_mask = np.array([  0,   0, 225], np.uint8)
    high_white_mask = np.array([180,  25, 255], np.uint8)
    low_garden_mask = np.array([33, 45, 25], np.uint8)
    high_garden_mask = np.array([90, 255, 210], np.uint8)

    sky_blue = cv2.inRange(hsv, low_blue_mask,  high_blue_mask)
    sky_white = cv2.inRange(hsv, low_white_mask, high_white_mask)
    sky = cv2.bitwise_or(sky_blue, sky_white)

    sky_d = cv2.dilate(sky, np.ones((3, 3), np.uint8), iterations=1)
    veg = cv2.inRange(hsv, low_garden_mask, high_garden_mask)
    vivid = (hsv[:, :, 1] > 150).astype(np.uint8) * 255

    stone_object = cv2.inRange(hsv, np.array([ 5,  0,  60], np.uint8), np.array([50, 100, 220], np.uint8))
    vivid = cv2.bitwise_and(vivid, cv2.bitwise_not(stone_object))

    artifacts = cv2.inRange(hsv, np.array([ 0, 30, 226], np.uint8), np.array([60, 255, 255], np.uint8))   # mask for lamps or flowers
    non_stone_object = cv2.bitwise_or(sky_d, veg)
    non_stone_object = cv2.bitwise_or(non_stone_object, vivid)
    non_stone_object = cv2.bitwise_or(non_stone_object, artifacts)

    return non_stone_object, sky


# Step 2: Stone Boundaries 

def compute_top_profile(non_stone, Height, Width):

    top_y = np.full(Width, float(Height), dtype=np.float32)
    ns = non_stone   
    for x in range(Width):
        col_ns = ns[:, x]
        stone_rows = np.where(col_ns == 0)[0]  
        if len(stone_rows) > 0:
            top_y[x] = float(stone_rows[0])
    return top_y


def smooth_profile(top_y, Width):

    sigma = max(3, Width // 80)
    ksize = 6 * sigma + 1
    profile = cv2.GaussianBlur(top_y.reshape(1, -1).astype(np.float32), (ksize, 1), sigma).reshape(-1)
    return profile

# Step 3: Find tip of the obelisk 

def raw_shaft_width(top_y_raw, tip_x, tip_y, Height, Width):

    delta = int(Height * 0.03)      
    low = max(0, tip_x - 200)
    high = min(Width - 1, tip_x + 200)
    thresh = tip_y + delta
    count = int(np.sum(top_y_raw[low:high + 1] <= thresh))
    return count


def measure_peak_width(profile, tip_x, tip_y, Height, Width):

    threshold = tip_y + Height * PEAK_DELTA_Y_FRAC

    x_left = tip_x
    while x_left > 0 and profile[x_left - 1] < threshold:
        x_left -= 1

    x_right = tip_x
    while x_right < Width - 1 and profile[x_right + 1] < threshold:
        x_right += 1

    return x_left, x_right


def find_obelisk_tip(profile, top_y_raw, Height, Width):

    margin     = int(Width * EDGE_MARGIN_FRAC)
    max_peak_w = int(Width * MAX_PEAK_WIDTH_FRAC)

    search = profile.copy()
    search[:margin]     = float(Height)
    search[Width - margin:] = float(Height)

    for _ in range(MAX_ATTEMPTS):
        tip_x = int(np.argmin(search))
        tip_y = float(search[tip_x])

        if tip_y >= float(Height) * 0.99:
            break   # no more valid candidates

        if tip_y > Height * MAX_TIP_Y_FRAC:
            break   # structure doesn't reach high enough

        x_left, x_right = measure_peak_width(profile, tip_x, tip_y, Height, Width)

        pw = x_right - x_left
        at_left_edge = (x_left  <= margin + 3)
        at_right_edge = (x_right >= Width - margin - 4)

        if pw <= max_peak_w and not at_left_edge and not at_right_edge:
            rw = raw_shaft_width(top_y_raw, tip_x, int(tip_y), Height, Width)
            if rw >= MIN_RAW_OBELISK_WIDTH:
                return tip_x, int(tip_y)

        excl_x1 = max(0,x_left - 3)
        excl_x2 = min(Width - 1, x_right + 3)
        search[excl_x1:excl_x2 + 1] = float(Height)

    return None

# Step 4: Finding body of the obelisk

def refine_shaft_bottom(non_stone_object, shaft_left_edge, shaft_right_edge, tip_y, Height):

    shaft_width = max(1, shaft_right_edge - shaft_left_edge)
    shaft_cx = (shaft_left_edge + shaft_right_edge) // 2
    Width = non_stone_object.shape[1]

    half_band = min(shaft_width * 4, Width // 3)
    sx1 = max(0, shaft_cx - half_band)
    sx2 = min(Width - 1, shaft_cx + half_band)
    band_w = sx2 - sx1 + 1

    shaft_occ    = shaft_width / band_w
    floor_thresh = min(0.90, max(FILL_THRESH, shaft_occ + FLOOR_MARGIN))
    last_y = tip_y  

    for y in range(tip_y + 5, Height):
        row = non_stone_object[y, sx1:sx2]
        stone = row == 0    
        count = int(stone.sum())

        if count < MIN_STONE:
            break

        fill = count / band_w
        if fill > floor_thresh:
            break

        last_y = y

    return last_y


def find_shaft_extents(edges_clean, tip_y, peak_left, peak_right, Height, Width):

    pad = max(15, (peak_right - peak_left) // 2)
    rx1 = max(0, peak_left - pad)
    rx2 = min(Width, peak_right + pad)
    roi = edges_clean[:, rx1:rx2]

    lines = cv2.HoughLinesP(roi, rho = 1, theta = np.pi / 180, threshold = HOUGH_THRESH, minLineLength = HOUGH_MIN_LEN, maxLineGap = HOUGH_MAX_GAP,)

    shaft_left_edge  = peak_left
    shaft_right_edge  = peak_right
    shaft_bot = min(Height - 1, int(Height * 0.95)) 

    if lines is None:
        return shaft_left_edge, shaft_right_edge, shaft_bot

    center_x  = (peak_left + peak_right) / 2
    left_xs   = []
    right_xs  = []
    max_bot_y = tip_y

    for seg in lines:
        lx1, ly1, lx2, ly2 = seg[0]
        dy = abs(ly2 - ly1)
        dx = abs(lx2 - lx1)
        if dy < 5:
            continue
        angle = np.degrees(np.arctan2(dx, dy + 1e-6))
        if angle > HOUGH_VERT_TOL:
            continue

        abs_x1 = rx1 + lx1
        abs_x2 = rx1 + lx2
        avg_abs_x = (abs_x1 + abs_x2) / 2

        if avg_abs_x < center_x:
            left_xs.append(min(abs_x1, abs_x2))
        else:
            right_xs.append(max(abs_x1, abs_x2))

        max_bot_y = max(max_bot_y, ly1, ly2)

    if left_xs:
        shaft_left_edge = max(0, min(left_xs) - 5)
    if right_xs:
        shaft_right_edge = min(Width - 1, max(right_xs) + 5)
    if max_bot_y > tip_y:
        shaft_bot = min(Height - 1, max_bot_y + int(Height * 0.04))

    return shaft_left_edge, shaft_right_edge, shaft_bot

# Detection pipeline

def detect(img_bgr, verbose=False):

    H_orig, W_orig = img_bgr.shape[:2]
    scale = RESIZE_WIDTH / W_orig
    Width = RESIZE_WIDTH
    Height = int(H_orig * scale)
    img_width  = cv2.resize(img_bgr, (Width, Height))

    gray = cv2.cvtColor(img_width, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    debug = dict(img_w=img_width, non_stone=None, sky=None, edges=None, profile_raw=None, profile_smooth=None, tip=None, peak=None, final_bbox=None)

    non_stone_object, sky = build_non_stone_mask(img_width)
    edges_clean = cv2.bitwise_and(edges, cv2.bitwise_not(sky))
    debug.update(non_stone=non_stone_object, sky=sky, edges=edges_clean)

    top_y = compute_top_profile(non_stone_object, Height, Width)
    profile = smooth_profile(top_y, Width)
    debug.update(profile_raw=top_y, profile_smooth=profile)

    if verbose:
        min_y = profile.min()
        min_x = int(np.argmin(profile))
        print(f"Profile min: x={min_x}  y={min_y:.1f}. (y_frac={min_y/Height:.2f})")

    tip_result = find_obelisk_tip(profile, top_y, Height, Width)
    if tip_result is None:
        if verbose:
            print("Fail: no narrow tip found in central region")
        return False, None, debug

    tip_x, tip_y = tip_result
    debug["tip"] = (tip_x, tip_y)

    peak_left, peak_right = measure_peak_width(profile, tip_x, tip_y, Height, Width)
    peak_width = peak_right - peak_left
    debug["peak"] = (peak_left, tip_y, peak_right)

    rw_check = raw_shaft_width(top_y, tip_x, tip_y, Height, Width)
    if verbose:
        print(f"Tip:  ({tip_x}, {tip_y}). Peak_width={peak_width} px ({peak_width/Width:.2f} of W). Raw_shaft_w={rw_check}")

    sx1, sx2, _ = find_shaft_extents( edges_clean, tip_y, peak_left, peak_right, Height, Width)

    max_hough_w = int(peak_width * 2.5)
    if (sx2 - sx1) > max_hough_w:
        half = max_hough_w // 2
        sx1 = max(0, tip_x - half)
        sx2 = min(Width - 1, tip_x + half)

    shaft_w_px = max(1, sx2 - sx1)
    s_bot_fill = refine_shaft_bottom(non_stone_object, sx1, sx2, tip_y, Height)

    SHAFT_CAP_PX = int(Width * 0.15) 
    capped_shaft_w = min(shaft_w_px, SHAFT_CAP_PX)
    min_bot = tip_y + capped_shaft_w * 5
    s_bot = max(s_bot_fill, min_bot)
    s_bot = min(s_bot, Height - 1)

    if verbose:
        print(f" Shaft: x=[{sx1},{sx2}]  shaft_w={shaft_w_px}, fill_bot={s_bot_fill}  min_bot={min_bot}  final_bot={s_bot}")

    # Bounding Box
    bx1 = max(0, sx1)
    by1 = max(0, tip_y - 4)
    bx2 = min(Width - 1, sx2)
    by2 = min(Height - 1, s_bot)
    debug["final_bbox"] = (bx1, by1, bx2, by2)

    # Convert to original image
    x1_o = bx1 / scale;  y1_o = by1 / scale
    x2_o = bx2 / scale;  y2_o = by2 / scale

    cx_n = float(np.clip((x1_o + x2_o) / 2 / W_orig, 0.0, 1.0))
    cy_n = float(np.clip((y1_o + y2_o) / 2 / H_orig, 0.0, 1.0))
    w_n = float(np.clip((x2_o - x1_o) / W_orig, 0.001, 1.0))
    h_n = float(np.clip((y2_o - y1_o) / H_orig, 0.001, 1.0))

    if verbose:
        print(f" YOLO: {CLASS_ID} {cx_n:.4f} {cy_n:.4f} "
              f"{w_n:.4f} {h_n:.4f}")

    return True, (cx_n, cy_n, w_n, h_n), debug

# Functions for debuging 

def save_debug_images(img_bgr, debug, stem, debug_dir):
    Height_orig, Width_orig = img_bgr.shape[:2]
    scale  = RESIZE_WIDTH / Width_orig
    Width = RESIZE_WIDTH
    Height = int(Height_orig * scale)
    img_w  = cv2.resize(img_bgr, (Width, Height))

    os.makedirs(debug_dir, exist_ok=True)

    if debug.get("non_stone") is not None:
        cv2.imwrite(f"{debug_dir}/{stem}_non_stone.jpg", debug["non_stone"])

    if debug.get("edges") is not None:
        cv2.imwrite(f"{debug_dir}/{stem}_edges.jpg", debug["edges"])

    if debug.get("profile_smooth") is not None:
        prof_vis = cv2.cvtColor(img_w.copy(), cv2.COLOR_BGR2GRAY)
        prof_vis = cv2.cvtColor(prof_vis, cv2.COLOR_GRAY2BGR)
        p = debug["profile_smooth"]
        for x in range(Width - 1):
            y1 = int(np.clip(p[x], 0, Height - 1))
            y2 = int(np.clip(p[x + 1], 0, Height - 1))
            cv2.line(prof_vis, (x, y1), (x + 1, y2), (0, 255, 255), 1)
        cv2.imwrite(f"{debug_dir}/{stem}_profile.jpg", prof_vis)

    vis = img_w.copy()

    if debug.get("tip"):
        tx, ty = debug["tip"]
        cv2.circle(vis, (tx, ty), 10, (0, 0, 255), -1)
        cv2.putText(vis, "TIP", (tx + 14, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if debug.get("peak"):
        px1, py, px2 = debug["peak"]
        cv2.line(vis, (px1, py), (px2, py), (255, 165, 0), 2)

    if debug.get("final_bbox"):
        x1, y1, x2, y2 = debug["final_bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(vis, "OBELISK", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    cv2.imwrite(f"{debug_dir}/{stem}_result.jpg", vis)

def label_all(input_dir = INPUT_DIR, output_dir = OUTPUT_DIR, debug_dir = DEBUG_DIR, save_debug = True, verbose = False):

    for sub in ("images", "labels"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    img_paths = sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if not img_paths:
        print(f"Error: No images found in '{input_dir}'")
        return
    
    ok_count   = 0
    fail_names = []

    for img_path in img_paths:
        print(f"\n[{img_path.name}]")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: cannot read '{img_path}',  skipped")
            fail_names.append(img_path.name)
            continue

        success, yolo, dbg = detect(img, verbose=verbose)

        if save_debug:
            save_debug_images(img, dbg, img_path.stem, debug_dir)

        if success:
            cx, cy, w, h = yolo
            print(f"Success: {CLASS_ID} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")

            shutil.copy(img_path, os.path.join(output_dir, "images", img_path.name))
            with open(os.path.join(output_dir, "labels", img_path.stem + ".txt"), "w") as f:
                f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            ok_count += 1
        else:
            print(f" Miss : no obelisk detected")
            fail_names.append(img_path.name)

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n" "train: images\n" "val: images\n" "nc: 1\n" "names: ['obelisk']\n")

    print(f"Detected: {ok_count} / {len(img_paths)}")
    if fail_names:
        print(f"Missed : {len(fail_names)}")

if __name__ == "__main__":
    label_all(save_debug=True, verbose=True)
