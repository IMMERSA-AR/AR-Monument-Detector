import cv2
from matplotlib.pyplot import hsv
from matplotlib.pyplot import hsv
import numpy as np
import os
import time

# ── Paths ──────────────────────────────────────────────────────
INPUT_FOLDER  = "photos"
OUTPUT_FOLDER = "output"
STEPS_FOLDER  = "steps_ip"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STEPS_FOLDER,  exist_ok=True)

# ── Settings ───────────────────────────────────────────────────
OBELISK_COLORS = [
    "3F3B3C",   # shadow / dark side
    "453C33",   # mid-tone
    "979088", 
    "969082",
    "706E6F"
]
COLOUR_THRESHOLD = 30         # Euclidean RGB distance tolerance


# ══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════

def x_at_y(pt1, pt2, y):
    x1, y1 = pt1
    x2, y2 = pt2
    if y2 == y1:
        return float(x1)
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1;  x2, y2 = p2
    x3, y3 = p3;  x4, y4 = p4
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    det = dx1 * dy2 - dy1 * dx2
    if abs(det) < 1e-8:
        return None
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
    return (x1 + t * dx1, y1 + t * dy1)


def hex_to_bgr(hex_str):
    h  = hex_str.lstrip('#')
    r  = int(h[0:2], 16)
    g  = int(h[2:4], 16)
    b  = int(h[4:6], 16)
    return float(b), float(g), float(r)


def save(sf, name, img):
    cv2.imwrite(os.path.join(sf, name), img)


# ══════════════════════════════════════════════════════════════
#  STEP 1 — PREPROCESSING
# ══════════════════════════════════════════════════════════════

def step1_preprocess(img, h, sf):
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    k = max(3, int(h * 0.0015))
    if k % 2 == 0:
        k += 1
    k = min(k, 9)
    blurred = cv2.GaussianBlur(enhanced, (k, k), 0)

    save(sf, "s1a_gray.jpg",     gray)
    save(sf, "s1b_enhanced.jpg", enhanced)
    save(sf, "s1c_blurred.jpg",  blurred)
    return blurred


# ══════════════════════════════════════════════════════════════
#  STEP 2 — EDGE DETECTION (Canny)
# ══════════════════════════════════════════════════════════════

def step2_canny(blurred, sf):
    v     = np.median(blurred)
    lower = int(max(0,   (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blurred, lower, upper)

    save(sf, "s2_edges.jpg", edges)
    print(f"    Canny thresholds: lower={lower}  upper={upper}")
    return edges


# ══════════════════════════════════════════════════════════════
#  STEP 2b — VEGETATION SUPPRESSION
# ══════════════════════════════════════════════════════════════

BUILDING_COLOR_HEX  = "EFE5CA"
BUILDING_THRESHOLD  = 40

def step2b_suppress_vegetation(edges, img, sf):
    kernel = np.ones((3, 3), np.uint8)

    hsv_img    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── Vegetation mask ────────────────────────────────────────
    green_mask = cv2.inRange(hsv_img,
                             np.array([25,  40,  40]),
                             np.array([90, 255, 255]))
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    # ── Tree trunk mask ────────────────────────────────────────
    # FIX: raised saturation floor 30→50, lowered value ceiling 160→130
    # to avoid matching the obelisk's low-saturation dark granite.
    trunk_mask = cv2.inRange(hsv_img,
                             np.array([ 8,  50,  20]),
                             np.array([22, 255, 130]))
    trunk_kernel = np.ones((2, 2), np.uint8)
    trunk_mask   = cv2.dilate(trunk_mask, trunk_kernel, iterations=1)

    # ── Building mask ──────────────────────────────────────────
    b_hex = BUILDING_COLOR_HEX.lstrip('#')
    B_b = float(int(b_hex[4:6], 16))
    G_b = float(int(b_hex[2:4], 16))
    R_b = float(int(b_hex[0:2], 16))

    f    = img.astype(np.float64)
    dist = np.sqrt((R_b - f[:, :, 2])**2 +
                   (G_b - f[:, :, 1])**2 +
                   (B_b - f[:, :, 0])**2)
    building_mask = (dist < BUILDING_THRESHOLD).astype(np.uint8) * 255
    building_mask = cv2.dilate(building_mask, kernel, iterations=1)

    # ── Combine ────────────────────────────────────────────────
    combined_mask = cv2.bitwise_or(green_mask, trunk_mask)
    combined_mask = cv2.bitwise_or(combined_mask, building_mask)

    # FIX: protect ±15% centre band — obelisk is always near centre.
    # Prevents trunk/building suppression from erasing the obelisk's
    # own edges even when a lamp post or building is immediately adjacent.
    h_img, w_img = edges.shape[:2]
    cx = w_img // 2
    band = int(w_img * 0.15)
    combined_mask[:, max(0, cx - band) : min(w_img, cx + band)] = 0

    edges_clean = edges.copy()
    edges_clean[combined_mask > 0] = 0

    save(sf, "s2b_green_mask.jpg",    green_mask)
    save(sf, "s2b_trunk_mask.jpg",    trunk_mask)
    save(sf, "s2c_building_mask.jpg", building_mask)
    save(sf, "s2d_combined_mask.jpg", combined_mask)
    save(sf, "s2e_edges_clean.jpg",   edges_clean)
    pct = int((combined_mask > 0).sum() / combined_mask.size * 100)
    print(f"    Suppression: {pct}% of image area masked (vegetation + trunks + building)")
    return edges_clean


# ══════════════════════════════════════════════════════════════
#  STEP 3 — LINE DETECTION (Hough + LSD merged)
# ══════════════════════════════════════════════════════════════

def step3_detect_lines(edges, img, h, w, sf):
    """
    Run BOTH Hough and LSD, then merge their outputs into one list.

    Why both?
    - LSD produces clean, precise segments with few duplicates but can
      MISS edges that are slightly blurred or low-contrast (e.g. the
      obelisk's darker face against a building).
    - Hough is noisier and produces many overlapping segments, but
      catches edges LSD misses because it votes across many pixels.
    - Union of both = maximum recall. Step 4's quality filters then
      remove the Hough noise, leaving the best lines from either detector.

    Deduplication: after merging, remove any line from one detector
    that is within DEDUP_PX pixels of a line from the other (keep the
    LSD version as it is more precise).
    """
    DEDUP_PX = 8   # lines whose midpoints are within 8px are considered duplicates

    # ── Hough ──────────────────────────────────────────────────
    hough_raw = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=35,
        minLineLength=min(h * 0.08, 300),
        maxLineGap=80,
    )
    hough_lines = []
    if hough_raw is not None:
        for ln in hough_raw:
            x1, y1, x2, y2 = ln[0]
            hough_lines.append((int(x1), int(y1), int(x2), int(y2)))

    # ── LSD ────────────────────────────────────────────────────
    lsd_lines = []
    try:
        lsd = cv2.createLineSegmentDetector(0)
        lsd_raw = lsd.detect(edges)
        if lsd_raw[0] is not None:
            min_len = min(h * 0.08, 300)
            for ln in lsd_raw[0]:
                x1, y1, x2, y2 = float(ln[0,0]), float(ln[0,1]), float(ln[0,2]), float(ln[0,3])
                if np.hypot(x2 - x1, y2 - y1) >= min_len:
                    lsd_lines.append((int(round(x1)), int(round(y1)),
                                      int(round(x2)), int(round(y2))))
    except Exception as e:
        print(f"    LSD error: {e} — using Hough only")

    # ── Merge: start with all LSD lines, add Hough lines that are
    #    not already covered by a nearby LSD line ─────────────────
    def _midpoint(seg):
        return ((seg[0] + seg[2]) / 2.0, (seg[1] + seg[3]) / 2.0)

    merged = list(lsd_lines)
    lsd_mids = [_midpoint(s) for s in lsd_lines]

    for hseg in hough_lines:
        hx, hy = _midpoint(hseg)
        # Check if any LSD line midpoint is within DEDUP_PX
        duplicate = any(
            abs(hx - lx) < DEDUP_PX and abs(hy - ly) < DEDUP_PX
            for lx, ly in lsd_mids
        )
        if not duplicate:
            merged.append(hseg)

    # Convert to numpy array in Hough-compatible format [[x1,y1,x2,y2]]
    if merged:
        lines = np.array([[[x1, y1, x2, y2]] for x1, y1, x2, y2 in merged],
                         dtype=np.int32)
    else:
        lines = None

    # ── Visualise ──────────────────────────────────────────────
    vis = img.copy()
    if lsd_lines:
        for x1, y1, x2, y2 in lsd_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)   # green = LSD
    for hseg in hough_lines:
        x1, y1, x2, y2 = hseg
        hx, hy = _midpoint(hseg)
        duplicate = any(
            abs(hx - lx) < DEDUP_PX and abs(hy - ly) < DEDUP_PX
            for lx, ly in lsd_mids
        )
        if not duplicate:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)   # red = Hough-only
    save(sf, "s3_merged_lines.jpg", vis)

    print(f"    Line detection: Hough={len(hough_lines)}  "
          f"LSD={len(lsd_lines)}  merged={len(merged)}")
    return lines




# ══════════════════════════════════════════════════════════════
#  STEP 4 — FILTER VERTICAL LINES
# ══════════════════════════════════════════════════════════════

def step4_filter_verticals(lines, img, h, w, sf):
    if lines is None:
        print("    Vertical filter: 0 lines kept (no input lines)")
        return []

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── Sky mask ───────────────────────────────────────────────
    sky_blue  = cv2.inRange(hsv_img, np.array([85,  0, 130]), np.array([150, 110, 255]))
    sky_white = cv2.inRange(hsv_img, np.array([0,   0, 200]), np.array([180,  40, 255]))
    sky_mask  = cv2.bitwise_or(sky_blue, sky_white)
    # Dilate only vertically so the sky region extends downward slightly —
    # this lets lines whose top is just at the sky/building boundary be
    # counted as sky_top=True. Using a UNIFORM kernel was causing mid-image
    # lines to incorrectly get sky_top=True via horizontal bleed.
    sky_dilate_kernel = np.ones((max(3, int(h * 0.04)), 1), np.uint8)
    sky_mask_dilated  = cv2.dilate(sky_mask, sky_dilate_kernel, iterations=1)

    # ── Stone mask ─────────────────────────────────────────────
    # Obelisk granite: low saturation (0-55), value 70-195.
    # Upper value 195 (was 185) to catch the sun-lit face of the obelisk
    # in direct Cairo sunlight while still excluding bright cream building
    # walls (value typically 200-240).
    stone_mask = cv2.inRange(hsv_img,
                             np.array([0,   0,  70]),
                             np.array([180, 55, 195]))
    stone_dilate = np.ones((5, 5), np.uint8)
    stone_mask   = cv2.dilate(stone_mask, stone_dilate, iterations=2)
    save(sf, "s4b_stone_mask.jpg", stone_mask)

    kept = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy == 0:
            continue

        angle  = np.degrees(np.arctan2(dy, dx))
        length = np.hypot(dx, dy)
        x_mid  = (x1 + x2) / 2.0

        if angle  < 71:        continue
        if length < h * 0.10:  continue
        if x_mid  < w * 0.03:  continue
        if x_mid  > w * 0.97:  continue

        y_top_line = min(y1, y2)
        if y_top_line > h * 0.60:
            continue

        # Top endpoint
        if y1 <= y2:
            x_top, y_top = x1, y1
        else:
            x_top, y_top = x2, y2
        x_top_c = int(np.clip(x_top, 0, w - 1))
        y_top_c = int(np.clip(y_top, 0, h - 1))

        # ── Unified acceptance gate ────────────────────────────
        #
        # Condition A: top endpoint is in sky → keep
        in_sky = sky_mask_dilated[y_top_c, x_top_c] > 0

        # Condition B: line body runs alongside stone-coloured pixels.
        # CRITICAL FIX: sample only the INWARD-FACING side of each line
        # (toward image centre), not a symmetric ±band.
        # For the LEFT obelisk edge: the left side is building wall → excluded
        # from stone_mask. The right side (inward) is obelisk stone → matches.
        # For the RIGHT obelisk edge: the right side may be building/sky,
        # the left side (inward) is obelisk stone.
        # Using a symmetric band was causing the outer-side (building/sky)
        # pixels to dilute the stone fraction below the 0.25 threshold.
        def _stone_at(py_f, px_f):
            py = int(np.clip(py_f, 0, h - 1))
            # Inward direction: left-edge lines (x_mid < cx) sample rightward;
            # right-edge lines (x_mid > cx) sample leftward.
            if x_mid < w / 2.0:
                bx0 = int(np.clip(px_f,       0, w - 1))   # start at line
                bx1 = int(np.clip(px_f + 20,  0, w - 1))   # sample 20px inward (right)
            else:
                bx0 = int(np.clip(px_f - 20,  0, w - 1))   # sample 20px inward (left)
                bx1 = int(np.clip(px_f,        0, w - 1))   # end at line
            if bx0 >= bx1:
                return False
            return (stone_mask[py, bx0:bx1 + 1] > 0).mean() > 0.20

        frac_t = 0.25
        frac_b = 0.75
        px_t = x1 + (x2 - x1) * frac_t;  py_t = y1 + (y2 - y1) * frac_t
        px_b = x1 + (x2 - x1) * frac_b;  py_b = y1 + (y2 - y1) * frac_b
        stone_hits    = sum([_stone_at(py_t, px_t),
                             _stone_at((y1 + y2) / 2.0, x_mid),
                             _stone_at(py_b, px_b)])
        stone_adjacent = stone_hits >= 2

        # Condition C: pure-sky midpoint rejection — discard sky artifacts
        # regardless of A or B.
        x_mid_c = int(np.clip(x_mid, 0, w - 1))
        y_mid_c = int(np.clip((y1 + y2) / 2.0, 0, h - 1))
        r = 5
        y0c = max(0, y_mid_c - r);  y1c_w = min(h - 1, y_mid_c + r)
        x0c = max(0, x_mid_c - r);  x1c_w = min(w - 1, x_mid_c + r)
        window_sky   = sky_mask_dilated[y0c:y1c_w, x0c:x1c_w]
        window_stone = stone_mask[y0c:y1c_w, x0c:x1c_w]
        sky_ratio   = (window_sky   > 0).mean()
        stone_ratio = (window_stone > 0).mean()
        pure_sky_artifact = (sky_ratio > 0.90 and stone_ratio < 0.05)

        if pure_sky_artifact:
            continue
        if not (in_sky or stone_adjacent):
            continue

        kept.append(dict(x1=x1, y1=y1, x2=x2, y2=y2,
                         x_mid=x_mid, length=length,
                         sky_top=in_sky,
                         stone_adj=stone_adjacent))

    vert_vis = img.copy()
    for ln in kept:
        cv2.line(vert_vis, (ln['x1'], ln['y1']),
                 (ln['x2'], ln['y2']), (0, 255, 0), 4)
    save(sf, "s4_vertical_lines.jpg", vert_vis)
    print(f"    Vertical filter: {len(kept)} lines kept")
    return kept


# ══════════════════════════════════════════════════════════════
#  STEP 5 — FIND OBELISK LINE PAIR
# ══════════════════════════════════════════════════════════════

def step5_find_pair(vlines, img, h, w, sf):
    cx = w / 2.0
    min_ob_width = w * 0.01   # at least 1% — obelisk can be very far
    max_ob_width = 400        # hard pixel cap — obelisk never wider than 400px

    # Pre-compute stone mask once for between-lines check
    hsv_img    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    stone_mask = cv2.inRange(hsv_img,
                             np.array([0,   0,  70]),
                             np.array([180, 55, 195]))
    stone_mask = cv2.dilate(stone_mask, np.ones((5, 5), np.uint8), iterations=2)

    def _between_stone_ratio(left, right, n_samples=5):
        """
        Sample n_samples horizontal slices between the two lines.
        At each slice y, measure what fraction of pixels between xl and xr
        are stone-coloured. Returns the average fraction across all slices.
        An obelisk body is stone → high ratio.
        A building window gap is dark glass → low ratio.
        Garden gap is green → low ratio (green excluded from stone_mask).
        """
        y_top = min(left['y1'], left['y2'], right['y1'], right['y2'])
        y_bot = max(left['y1'], left['y2'], right['y1'], right['y2'])
        if y_bot <= y_top:
            return 0.0
        ratios = []
        for frac in np.linspace(0.2, 0.8, n_samples):
            y = int(y_top + (y_bot - y_top) * frac)
            xl = int(np.clip(x_at_y((left['x1'],  left['y1']),
                                    (left['x2'],  left['y2']),  y), 0, w - 1))
            xr = int(np.clip(x_at_y((right['x1'], right['y1']),
                                    (right['x2'], right['y2']), y), 0, w - 1))
            if xr <= xl:
                continue
            strip = stone_mask[y, xl:xr]
            ratios.append((strip > 0).mean())
        return float(np.mean(ratios)) if ratios else 0.0

    best_score = -1
    best_pair  = None

    for i in range(len(vlines)):
        for j in range(i + 1, len(vlines)):
            a = vlines[i]
            b = vlines[j]

            left  = a if a['x_mid'] < b['x_mid'] else b
            right = b if a['x_mid'] < b['x_mid'] else a

            # ── Hard gate 1: width ─────────────────────────────────
            y_top_left  = min(left['y1'],  left['y2'])
            y_top_right = min(right['y1'], right['y2'])
            y_bot_left  = max(left['y1'],  left['y2'])
            y_bot_right = max(right['y1'], right['y2'])

            y_overlap_top = max(y_top_left, y_top_right)
            y_overlap_bot = min(y_bot_left, y_bot_right)
            y_measure = ((y_overlap_top + y_overlap_bot) / 2
                         if y_overlap_top < y_overlap_bot else y_overlap_top)

            xl = x_at_y((left['x1'],  left['y1']),  (left['x2'],  left['y2']),  y_measure)
            xr = x_at_y((right['x1'], right['y1']), (right['x2'], right['y2']), y_measure)
            width = xr - xl

            if width < min_ob_width or width > max_ob_width:
                continue

            # ── Hard gate 2: pair top in upper 70% of image ────────
            pair_y_top = min(y_top_left, y_top_right)
            pair_y_bot = max(y_bot_left, y_bot_right)
            if pair_y_top > h * 0.70:
                continue

            # ══ SIGNAL 1: CONVERGENCE (weight 0.35) ════════════════
            # The pyramidion makes lines converge above the body.
            # This is the single strongest universal discriminator.
            apex = line_intersection(
                (left['x1'],  left['y1']),  (left['x2'],  left['y2']),
                (right['x1'], right['y1']), (right['x2'], right['y2']),
            )
            if apex is not None:
                ax, ay = apex
                body_h = max(1, pair_y_bot - pair_y_top)
                if ay < pair_y_top:
                    convergence = 1.0          # apex above body — perfect
                elif ay < pair_y_top + body_h * 0.2:
                    convergence = 0.75         # apex just inside top
                elif ay < h * 0.5:
                    convergence = 0.30         # apex in upper half of image
                elif ay < h:
                    convergence = 0.10         # apex in lower half
                else:
                    convergence = 0.02         # diverging downward
            else:
                convergence = 0.10             # parallel

            # ══ SIGNAL 2: SKY-TOP BONUS (weight 0.25) ══════════════
            # Both obelisk edges always emerge from sky at their top.
            # Garden stems and building frames do NOT both have sky tops.
            sky_l = left.get('sky_top',  False)
            sky_r = right.get('sky_top', False)
            sky_bonus = 1.0 if (sky_l and sky_r) else (0.4 if (sky_l or sky_r) else 0.0)

            # ══ SIGNAL 3: BETWEEN-LINES STONE DENSITY (weight 0.25) =
            # The obelisk body between the two edges is stone-coloured.
            # Building window gaps are dark glass; garden gaps are green.
            # Both are excluded from stone_mask → ratio near 0.
            between_stone = _between_stone_ratio(left, right)

            # ══ SIGNAL 4: STONE-EDGE ADJACENCY (weight 0.10) ═══════
            stone_l = left.get('stone_adj',  False)
            stone_r = right.get('stone_adj', False)
            stone_bonus = 1.0 if (stone_l and stone_r) else (0.5 if (stone_l or stone_r) else 0.0)

            # ══ SIGNAL 5: CENTRALITY (weight 0.05) ═════════════════
            pair_cx    = (left['x_mid'] + right['x_mid']) / 2.0
            centrality = 1.0 - abs(pair_cx - cx) / cx

            score = (0.35 * convergence
                   + 0.25 * sky_bonus
                   + 0.25 * between_stone
                   + 0.10 * stone_bonus
                   + 0.05 * centrality)

            if score > best_score:
                best_score = score
                best_pair  = (left, right, apex)

    # ── Mirroring fallback ──────────────────────────────────────
    if best_pair is None and len(vlines) >= 1:
        print("    Pair search: no two-line pair found — trying mirror fallback")

        def single_score(ln):
            centrality  = 1.0 - abs(ln['x_mid'] - cx) / cx
            length_norm = ln['length'] / h
            return centrality * length_norm

        best_single = max(vlines, key=single_score)

        if abs(best_single['x_mid'] - cx) < w * 0.30:
            offset = best_single['x_mid'] - cx

            mirrored = dict(
                x1    = int(round(2 * cx - best_single['x1'])),
                y1    = best_single['y1'],
                x2    = int(round(2 * cx - best_single['x2'])),
                y2    = best_single['y2'],
                x_mid = cx - offset,
                length= best_single['length'],
            )

            if best_single['x_mid'] > cx:
                left, right = mirrored, best_single
            else:
                left, right = best_single, mirrored

            apex = line_intersection(
                (left['x1'],  left['y1']),  (left['x2'],  left['y2']),
                (right['x1'], right['y1']), (right['x2'], right['y2']),
            )

            best_pair = (left, right, apex)
            print(f"    Mirror fallback used: detected_x={best_single['x_mid']:.0f}  "
                  f"mirrored_x={mirrored['x_mid']:.0f}  offset={offset:.0f}px")
        else:
            print(f"    Mirror fallback skipped: best line too far from centre "
                  f"(x_mid={best_single['x_mid']:.0f}, cx={cx:.0f})")

    if best_pair is None:
        print("    No valid obelisk line pair found.")
        return None

    left, right, apex = best_pair
    pair_vis = img.copy()
    cv2.line(pair_vis,
             (left['x1'],  left['y1']),  (left['x2'],  left['y2']),
             (255, 50, 50), 3)
    cv2.line(pair_vis,
             (right['x1'], right['y1']), (right['x2'], right['y2']),
             (50, 50, 255), 3)
    if apex is not None:
        ax, ay = int(apex[0]), int(apex[1])
        if -h < ay < 2 * h and 0 < ax < w:
            cv2.circle(pair_vis, (ax, ay), 12, (0, 255, 255), -1)
            cv2.putText(pair_vis, "apex", (ax + 14, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    save(sf, "s5_obelisk_pair.jpg", pair_vis)

    print(f"    Best pair score: {best_score:.4f}  "
          f"left_x={left['x_mid']:.0f}  right_x={right['x_mid']:.0f}  "
          f"width={abs(left['x_mid'] - right['x_mid']):.0f}px")
    return best_pair


# ══════════════════════════════════════════════════════════════
#  STEP 6 — BUILD POLYGON MASK
# ══════════════════════════════════════════════════════════════

def step6_polygon(best_pair, h, w, sf):
    left, right, apex = best_pair

    y_top = min(left['y1'], left['y2'], right['y1'], right['y2'])
    y_bot = max(left['y1'], left['y2'], right['y1'], right['y2'])
    y_bot = min(y_bot + int(h * 0.05), h - 1)

    xl_top = x_at_y((left['x1'],  left['y1']),  (left['x2'],  left['y2']),  y_top)
    xr_top = x_at_y((right['x1'], right['y1']), (right['x2'], right['y2']), y_top)
    xl_bot = x_at_y((left['x1'],  left['y1']),  (left['x2'],  left['y2']),  y_bot)
    xr_bot = x_at_y((right['x1'], right['y1']), (right['x2'], right['y2']), y_bot)

    xl_top = max(0, min(w - 1, xl_top))
    xr_top = max(0, min(w - 1, xr_top))
    xl_bot = max(0, min(w - 1, xl_bot))
    xr_bot = max(0, min(w - 1, xr_bot))

    use_apex = (apex is not None and
                apex[1] < y_top and
                0 < apex[0] < w)

    if use_apex:
        polygon = np.array([
            [int(apex[0]),  int(apex[1])],
            [int(xr_top),   int(y_top)],
            [int(xr_bot),   int(y_bot)],
            [int(xl_bot),   int(y_bot)],
            [int(xl_top),   int(y_top)],
        ], dtype=np.int32)
        print(f"    Polygon: PENTAGON (apex+body)  apex=({int(apex[0])},{int(apex[1])})  "
              f"bot=({int(xl_bot)},{int(y_bot)})-({int(xr_bot)},{int(y_bot)})")
    else:
        polygon = np.array([
            [int(xl_top), int(y_top)],
            [int(xr_top), int(y_top)],
            [int(xr_bot), int(y_bot)],
            [int(xl_bot), int(y_bot)],
        ], dtype=np.int32)
        print(f"    Polygon: TRAPEZOID  "
              f"top=({int(xl_top)},{int(y_top)})-({int(xr_top)},{int(y_top)})  "
              f"bot=({int(xl_bot)},{int(y_bot)})-({int(xr_bot)},{int(y_bot)})")

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    save(sf, "s6_polygon_mask.jpg", mask)
    return mask


# ══════════════════════════════════════════════════════════════
#  STEP 7 — COLOUR REFINEMENT
# ══════════════════════════════════════════════════════════════

def step7_colour_refine(img, polygon_mask, sf):
    f = img.astype(np.float64)

    min_dist = np.full(f.shape[:2], np.inf)
    for hex_color in OBELISK_COLORS:
        B_w, G_w, R_w = hex_to_bgr(hex_color)
        dist = np.sqrt((R_w - f[:, :, 2])**2 +
                    (G_w - f[:, :, 1])**2 +
                    (B_w - f[:, :, 0])**2)
        min_dist = np.minimum(min_dist, dist)

    colour_mask = (min_dist < COLOUR_THRESHOLD).astype(np.uint8) * 255

    hsv_img     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask  = cv2.inRange(hsv_img,
                              np.array([25,  40,  40]),
                              np.array([90, 255, 255]))
    not_green   = cv2.bitwise_not(green_mask)
    save(sf, "s7b_vegetation_mask.jpg", green_mask)

    refined = cv2.bitwise_and(colour_mask, polygon_mask)
    refined = cv2.bitwise_and(refined,     not_green)

    save(sf, "s7c_colour_mask.jpg",  colour_mask)
    save(sf, "s7d_refined_mask.jpg", refined)
    return refined


# ══════════════════════════════════════════════════════════════
#  STEP 8 — MORPHOLOGICAL CLEANUP
# ══════════════════════════════════════════════════════════════

def step8_morphology(mask, sf):
    k      = np.ones((9, 9), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    closed = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, k, iterations=3)

    save(sf, "s8a_closed.jpg", closed)
    save(sf, "s8b_opened.jpg", opened)
    return opened


# ══════════════════════════════════════════════════════════════
#  STEP 9 — OUTPUT
# ══════════════════════════════════════════════════════════════

def step9_output(img, mask, filename, sf):
    overlay     = img.copy()
    green_layer = np.zeros_like(img)
    green_layer[mask > 0] = (0, 200, 0)
    overlay = cv2.addWeighted(overlay, 0.6, green_layer, 0.4, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)

    save(sf, "s9_final_overlay.jpg", overlay)
    name = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, name + "_result.jpg"), overlay)

    pts = cv2.findNonZero(mask)
    if pts is not None:
        h, w = img.shape[:2]
        rx, ry, rw, rh = cv2.boundingRect(pts)
        pad = max(8, min(rw, rh) // 15)
        x0, y0 = max(0, rx - pad), max(0, ry - pad)
        x1, y1 = min(w, rx + rw + pad), min(h, ry + rh + pad)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, name + "_crop.jpg"),
                    img[y0:y1, x0:x1])

    return overlay


# ══════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def process_image(img_path, filename):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  SKIP (cannot read): {filename}")
        return False

    h, w = img.shape[:2]
    name = os.path.splitext(filename)[0]
    sf   = os.path.join(STEPS_FOLDER, name)
    os.makedirs(sf, exist_ok=True)
    start_time = time.time()

    save(sf, "s0_original.jpg", img)
    print(f"\n{'─'*55}")
    print(f"Processing: {filename}  ({w}x{h})")

    print("  Step 1: preprocessing (grayscale + CLAHE + blur) ...")
    blurred = step1_preprocess(img, h, sf)

    print("  Step 2: Canny edge detection ...")
    edges = step2_canny(blurred, sf)

    print("  Step 2b: suppressing vegetation edges ...")
    edges = step2b_suppress_vegetation(edges, img, sf)

    print("  Step 3: line detection (Hough + LSD merged) ...")
    lines = step3_detect_lines(edges, img, h, w, sf)
    if lines is None:
        print("  No lines detected — skipping.")
        return False

    print("  Step 4: filtering vertical lines ...")
    vlines = step4_filter_verticals(lines, img, h, w, sf)
    if len(vlines) < 2:
        print("  Not enough vertical lines — skipping.")
        return False

    print("  Step 5: finding obelisk line pair ...")
    best_pair = step5_find_pair(vlines, img, h, w, sf)
    if best_pair is None:
        print("  No valid obelisk pair found.")
        return False

    print("  Step 6: building polygon mask ...")
    polygon_mask = step6_polygon(best_pair, h, w, sf)

    print("  Step 7: colour refinement ...")
    refined = step7_colour_refine(img, polygon_mask, sf)

    print("  Step 8: morphological cleanup ...")
    clean = step8_morphology(refined, sf)

    print("  Step 9: output ...")
    step9_output(img, clean, filename, sf)

    end_time = time.time()
    print(f"  Done — steps saved in '{sf}/' (took {end_time - start_time:.2f}s)")
    return True


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    files = sorted(f for f in os.listdir(INPUT_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    print(f"Found {len(files)} images.")

    ok = 0
    for filename in files:
        if process_image(os.path.join(INPUT_FOLDER, filename), filename):
            ok += 1

    print(f"\n{'='*55}")
    print(f"DONE.  Detected {ok} / {len(files)} images.")
    print(f"  Step images : '{STEPS_FOLDER}/<image_name>/'")
    print(f"  Final output: '{OUTPUT_FOLDER}/'")
    print(f"{'='*55}")