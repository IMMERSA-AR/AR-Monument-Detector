"""
panel_detector.py
=================
Two-stage classical image processing pipeline for detecting and identifying
exhibition panels at the Faculty of Engineering.

Stage 1  -  Localization
    Find the black metal frame in the camera frame using dark-pixel
    masking + contour detection + rotated bounding rect.
    Apply a perspective warp (homography) to produce a canonical
    frontal crop of the poster content.

Stage 2  -  Identification
    Run SIFT feature matching against every reference image.
    Use Lowe's ratio test + RANSAC homography to count geometric
    inliers per reference. The reference with the most inliers
    (above a minimum threshold) wins.

Fallback
    If Stage 1 finds no valid frame in the shot, Stage 2 runs directly
    on the full camera frame. SIFT + RANSAC is robust enough to still
    identify the panel from a distance or a partial view.

Reference images
    Place one clean photo of each panel in  panel_references/
    Name them  panel_0.jpg, panel_1.jpg, ... (alphabetical order = panel order).

Requirements
    pip install opencv-contrib-python numpy
"""

import cv2
import os
import numpy as np


class PanelDetector:
    """
    Detects and identifies one of N exhibition panels.

    Usage
    -----
        detector = PanelDetector("panel_references", panel_names=[...])
        result   = detector.detect(frame)   # frame: BGR numpy array
        vis      = detector.draw_result(frame, result)
    """

    # Canonical size for deskewed crops - portrait aspect (~1 : 1.6)
    CANONICAL_W = 600
    CANONICAL_H = 960

    def __init__(self,
                 references_folder: str,
                 panel_names: list = None,
                 min_inliers: int  = 12,
                 ratio_threshold: float = 0.75):
        """
        Parameters
        ----------
        references_folder : path to folder with panel_0.jpg ... panel_N.jpg
        panel_names       : display name per panel (same order as filenames).
                            Falls back to "Panel 0", "Panel 1", ... if None.
        min_inliers       : RANSAC inliers required to accept identification.
                            Lower  -> more detections, more false positives.
                            Higher -> stricter, fewer false positives.
                            Recommended range: 10 - 20.
        ratio_threshold   : Lowe's ratio test (0.70 - 0.80 is typical).
        """
        self.min_inliers     = min_inliers
        self.ratio_threshold = ratio_threshold

        # -- SIFT detector ------------------------------------------
        # Requires opencv-contrib-python.
        # nfeatures=0 -> keep ALL detected keypoints (max recall).
        self._sift = cv2.SIFT_create(
            nfeatures         = 0,
            contrastThreshold = 0.04,
            edgeThreshold     = 10,
        )

        # -- BFMatcher ----------------------------------------------
        # Brute-force L2 matching.
        # We use BF (not FLANN) because we have N independent descriptor
        # databases - one per reference image - and BF is simpler to
        # use independently against each one.
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

        # -- Load reference images ----------------------------------
        self._refs = []

        if not os.path.isdir(references_folder):
            raise FileNotFoundError(
                f"References folder not found: '{references_folder}'\n"
                f"Create it and place panel_0.jpg ... panel_N.jpg inside."
            )

        ref_files = sorted(
            f for f in os.listdir(references_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )

        if not ref_files:
            raise FileNotFoundError(
                f"No images found in '{references_folder}/'. "
                f"Add panel_0.jpg ... panel_N.jpg as reference photos."
            )

        for idx, fname in enumerate(ref_files):
            path = os.path.join(references_folder, fname)
            img  = cv2.imread(path)

            if img is None:
                print(f"  [PanelDetector] WARN: cannot read '{path}' - skipped")
                continue

            kp, desc = self._sift.detectAndCompute(img, None)

            if desc is None or len(kp) < 10:
                print(f"  [PanelDetector] WARN: too few keypoints in '{fname}' "
                      f"({len(kp)}) - skipped")
                continue

            name = (panel_names[idx]
                    if panel_names and idx < len(panel_names)
                    else f"Panel {idx}")

            self._refs.append({
                "id"   : idx,
                "name" : name,
                "file" : fname,
                "kp"   : kp,
                "desc" : desc,
            })
            print(f"  [PanelDetector] '{fname}'  ->  '{name}'  ({len(kp)} keypoints)")

        if not self._refs:
            raise ValueError(
                "No valid reference images could be loaded. "
                "Ensure the images are readable and well-lit."
            )

        print(f"[PanelDetector] Ready - {len(self._refs)} panels registered.\n")

    # ==============================================================
    #  STAGE 1  -  Panel Localization
    # ==============================================================

    def _locate_panels(self, frame: np.ndarray) -> list:
        """
        Locate the main (central) exhibition panel using a brightness
        column profile.

        WHY column profile instead of blob detection
        ---------------------------------------------
        Both the dark-frame and bright-poster blob approaches fail here
        because all dark/bright regions merge across panels.

        The key observation: the thick black metal frame borders appear
        as sharp VERTICAL DARK STRIPES in the column brightness profile.
        These stripes naturally separate the main panel from its
        neighbours — no morphology or thresholding ambiguity.

        Steps
        -----
        1. Resize to 800 px wide for speed.
        2. Compute column-wise mean brightness  ->  1-D profile (width).
        3. Smooth the profile to remove noise from text/drawings.
        4. Find "bright runs" = consecutive columns above a dark threshold.
           Each bright run = one panel's poster region.
        5. Score runs by width x centrality; pick the main panel.
        6. Find vertical extent by repeating the same idea on rows,
           but only inside the selected column range.
        7. Scale back to original resolution, add margin, deskew.

        Returns
        -------
        List with one (deskewed_crop, corners) pair, or empty list.
        corners : np.ndarray shape (4, 2) float32
                  order  [top-left, top-right, bottom-right, bottom-left]
        """
        fh, fw = frame.shape[:2]

        # -- Step 1: work at reduced resolution ----------------------
        WORK_W = 800
        scale  = WORK_W / fw
        WORK_H = int(fh * scale)
        small  = cv2.resize(frame, (WORK_W, WORK_H))
        gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # -- Step 2: column brightness profile -----------------------
        # Average grayscale per column -> dark dips = frame borders.
        col_profile = gray.mean(axis=0).astype(float)   # shape (WORK_W,)

        # -- Step 3: smooth to handle text/drawings inside poster ----
        k = 15
        col_smooth = np.convolve(col_profile, np.ones(k) / k, mode='same')

        # -- Step 4: find bright column runs -------------------------
        # Columns darker than DARK_THRESHOLD belong to a frame border.
        # The frame borders are near-black ( < 50 average ) so 100 is
        # a safe threshold that doesn't accidentally cut bright wall.
        DARK_THRESHOLD = 100
        is_dark = col_smooth < DARK_THRESHOLD

        bright_runs = []     # list of (col_start, col_end)
        start = None
        for i, dark in enumerate(is_dark):
            if not dark and start is None:
                start = i
            elif dark and start is not None:
                bright_runs.append((start, i))
                start = None
        if start is not None:
            bright_runs.append((start, WORK_W))

        if not bright_runs:
            return []

        # -- Step 5: score runs by width x centrality ----------------
        cx_frame = WORK_W / 2.0
        scored   = []
        for x0, x1 in bright_runs:
            width = x1 - x0
            if width < WORK_W * 0.10:        # skip very thin slivers
                continue
            cx_run     = (x0 + x1) / 2.0
            centrality = 1.0 - abs(cx_run - cx_frame) / cx_frame
            score      = width * (0.5 + 0.5 * centrality)
            scored.append((score, x0, x1))

        if not scored:
            return []

        scored.sort(reverse=True)
        _, col_x0, col_x1 = scored[0]        # winning panel column range

        # -- Step 6: row extent using dark frame border detection -------
        # Strategy: find the dark metal frame borders (near-black horizontal
        # bands) that define the top and bottom of the panel.
        #
        # WHY this is better than finding "bright content":
        # The metal frame is reliably very dark (<60 grayscale) no matter
        # what artwork is printed on the panel. Panels with dark photos or
        # engineering drawings have mean row brightness of 70-120, which
        # caused the old "bright content > 130" approach to miss them and
        # fall back to full-frame height.
        #
        # Structure (top to bottom in the row profile):
        #   - Wall / ceiling (bright, ~120-160) -- NOT dark, ignored
        #   - Top metal frame (very dark, ~20-50) -- DETECTED as border
        #   - Panel content (variable, 50-250)
        #   - Bottom metal frame (very dark, ~20-50) -- DETECTED as border
        #   - Fabric skirt / floor (very dark, ~20-50) -- DETECTED as border
        #
        # By finding the first and last significant dark runs, we bracket
        # the panel content regardless of how light or dark that content is.
        col_slice   = gray[:, col_x0:col_x1]
        row_profile = col_slice.mean(axis=1).astype(float)  # shape (WORK_H,)
        k_row       = 5     # small kernel: keep frame borders sharp
        row_smooth  = np.convolve(row_profile, np.ones(k_row) / k_row, mode='same')

        FRAME_DARK = 60     # metal frame + fabric skirt are reliably below this
        MIN_BORDER = 8      # min consecutive rows to be a real border, not noise

        is_dark_row = row_smooth < FRAME_DARK

        # Collect contiguous dark runs
        dark_runs = []
        run_start = None
        for i, dark in enumerate(is_dark_row):
            if dark and run_start is None:
                run_start = i
            elif not dark and run_start is not None:
                dark_runs.append((run_start, i))
                run_start = None
        if run_start is not None:
            dark_runs.append((run_start, WORK_H))

        # Keep only significant dark runs (wide enough to be a real border)
        sig_dark = [(y0, y1) for y0, y1 in dark_runs if y1 - y0 >= MIN_BORDER]

        if len(sig_dark) >= 2:
            # Panel content sits between first and last significant dark runs
            row_y0 = sig_dark[0][1]     # just below the top frame border
            row_y1 = sig_dark[-1][0]    # just above the bottom frame / skirt
        elif len(sig_dark) == 1:
            y0d, y1d = sig_dark[0]
            if (y0d + y1d) / 2.0 < WORK_H / 2:  # dark run in top half = top border
                row_y0 = y1d
                row_y1 = WORK_H
            else:                                   # dark run in bottom half = skirt
                row_y0 = 0
                row_y1 = y0d
        else:
            # No dark borders detected: fall back to brightness-based search
            bright_row_idx = np.where(row_smooth > 100)[0]
            if len(bright_row_idx) == 0:
                row_y0, row_y1 = 0, WORK_H
            else:
                row_y0 = int(bright_row_idx[0])
                row_y1 = int(bright_row_idx[-1])

        # Sanity guard
        row_y0 = max(0, row_y0)
        row_y1 = min(WORK_H, row_y1)
        if row_y1 <= row_y0:
            row_y0, row_y1 = 0, WORK_H

        # -- Step 7: scale back and add margin -----------------------
        pad_x = int((col_x1 - col_x0) * 0.03)
        pad_y = int((row_y1 - row_y0) * 0.02)

        x0_orig = max(0,  int((col_x0 - pad_x) / scale))
        x1_orig = min(fw, int((col_x1 + pad_x) / scale))
        y0_orig = max(0,  int((row_y0 - pad_y) / scale))
        y1_orig = min(fh, int((row_y1 + pad_y) / scale))

        corners = self._order_corners(np.array(
            [[x0_orig, y0_orig], [x1_orig, y0_orig],
             [x1_orig, y1_orig], [x0_orig, y1_orig]],
            dtype=np.float32
        ))

        crop = self._deskew(frame, corners)
        return [(crop, corners)]

    # -- Geometry helpers -------------------------------------------

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """
        Reorder 4 arbitrary points as:
            [top-left, top-right, bottom-right, bottom-left]

        Trick: for an axis-aligned or gently rotated rectangle -
          TL has the smallest (x + y) sum
          BR has the largest  (x + y) sum
          TR has the smallest (y - x) difference
          BL has the largest  (y - x) difference
        """
        pts = pts.reshape(4, 2)
        s   = pts.sum(axis=1)           # x + y
        d   = np.diff(pts, axis=1).ravel()  # y - x
        tl  = pts[np.argmin(s)]
        br  = pts[np.argmax(s)]
        tr  = pts[np.argmin(d)]
        bl  = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _deskew(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Warp the quadrilateral defined by `corners` (TL, TR, BR, BL)
        into a canonical CANONICAL_W × CANONICAL_H upright rectangle.

        This removes perspective distortion so Stage 2 sees a frontal
        view of the poster regardless of camera angle.
        """
        dst = np.array([
            [0,                    0                   ],
            [self.CANONICAL_W - 1, 0                   ],
            [self.CANONICAL_W - 1, self.CANONICAL_H - 1],
            [0,                    self.CANONICAL_H - 1],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(corners, dst)

        if H is None:
            # Degenerate case - just resize the whole frame
            return cv2.resize(frame, (self.CANONICAL_W, self.CANONICAL_H))

        return cv2.warpPerspective(
            frame, H, (self.CANONICAL_W, self.CANONICAL_H)
        )

    # ==============================================================
    #  STAGE 2  -  Panel Identification
    # ==============================================================

    def _identify_panel(self, img: np.ndarray):
        """
        Match `img` against every loaded reference.

        For each reference:
          1. kNN-match (k=2): reference descriptors -> image descriptors.
          2. Lowe's ratio test: keep only unambiguous matches.
          3. RANSAC homography: count geometrically consistent inliers.

        The reference with the most RANSAC inliers wins.
        Returns (panel_id, inlier_count, ref_dict) or (None, 0, None).

        Why reference -> image (not image -> reference)?
        ------------------------------------------------
        The reference image is clean and complete. Matching each
        reference descriptor to the image lets us find the subset of
        the reference that is visible in the (possibly partial) frame.
        This is the same direction used in obelisk_detection_fm.py.
        """
        kp_img, desc_img = self._sift.detectAndCompute(img, None)

        if desc_img is None or len(kp_img) < 4:
            return None, 0, None

        best_id      = None
        best_inliers = 0
        best_ref     = None

        for ref in self._refs:

            # -- kNN match ------------------------------------------
            try:
                raw_matches = self._matcher.knnMatch(
                    ref["desc"], desc_img, k=2
                )
            except cv2.error:
                continue

            # -- Lowe's ratio test -----------------------------------
            # Keep only matches where the best neighbour is significantly
            # closer than the second-best - removes ambiguous matches.
            good = [
                m for pair in raw_matches
                if len(pair) == 2
                for m, n in [(pair[0], pair[1])]
                if m.distance < self.ratio_threshold * n.distance
            ]

            # Need at least half of min_inliers good matches before
            # we bother with the (expensive) RANSAC step.
            if len(good) < max(4, self.min_inliers // 2):
                continue

            # -- RANSAC homography -----------------------------------
            # src_pts: where those keypoints sit in the REFERENCE image
            # dst_pts: where the matching keypoints sit in the QUERY image
            src_pts = np.float32(
                [ref["kp"][m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            dst_pts = np.float32(
                [kp_img[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            _, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC, ransacReprojThreshold=5.0
            )

            if mask is None:
                continue

            # Inliers = matches that are geometrically consistent
            # with the estimated homography
            inliers = int(mask.ravel().sum())

            if inliers > best_inliers:
                best_inliers = inliers
                best_id      = ref["id"]
                best_ref     = ref

        if best_inliers >= self.min_inliers:
            return best_id, best_inliers, best_ref

        return None, best_inliers, None

    # ==============================================================
    #  FULL PIPELINE
    # ==============================================================

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run the full two-stage pipeline on a BGR camera frame.

        Returns
        -------
        On success:
            {
                "detected"   : True,
                "panel_id"   : 2,
                "panel_name" : "Mechanical Engineering Drawings",
                "score"      : 47,            # RANSAC inlier count
                "stage"      : "localized",   # or "full_frame_fallback"
                "corners"    : [[x,y], ...],  # 4 corners in original frame
                "bbox": {
                    "x"      : 0.12,   # all normalized 0 - 1
                    "y"      : 0.05,
                    "width"  : 0.38,
                    "height" : 0.90,
                    "cx"     : 0.31,
                    "cy"     : 0.50,
                }
            }
        On failure:
            {"detected": False, "reason": "..."}
        """
        fh, fw = frame.shape[:2]

        best_id      = None
        best_score   = 0
        best_corners = None
        best_ref     = None
        stage        = "full_frame_fallback"

        # -- Stage 1: find framed panel candidates ------------------
        candidates = self._locate_panels(frame)

        for crop, corners in candidates:
            pid, score, ref = self._identify_panel(crop)
            if score > best_score:
                best_score   = score
                best_id      = pid
                best_corners = corners
                best_ref     = ref
                stage        = "localized"

        # -- Fallback: match on the raw full frame ------------------
        # Runs when:
        #   (a) Stage 1 found no dark frame shape at all, OR
        #   (b) Stage 1 found a shape but no reference matched it.
        # SIFT + RANSAC is robust enough to identify the panel even
        # without deskewing, especially at moderate viewing distances.
        if best_id is None:
            pid, score, ref = self._identify_panel(frame)
            if score > best_score:
                best_score   = score
                best_id      = pid
                best_corners = None
                best_ref     = ref
                stage        = "full_frame_fallback"

        if best_id is None:
            return {
                "detected" : False,
                "reason"   : (
                    f"no panel matched above threshold "
                    f"(best score={best_score}, need {self.min_inliers})"
                ),
            }

        # -- Build bounding box -------------------------------------
        if best_corners is not None:
            xs = best_corners[:, 0]
            ys = best_corners[:, 1]
        else:
            # Full-frame fallback: return whole-frame bbox
            xs = np.array([0.0, float(fw), float(fw), 0.0])
            ys = np.array([0.0, 0.0, float(fh), float(fh)])

        rx,  ry  = float(xs.min()), float(ys.min())
        rx2, ry2 = float(xs.max()), float(ys.max())
        rw,  rh  = rx2 - rx,        ry2 - ry

        return {
            "detected"   : True,
            "panel_id"   : best_id,
            "panel_name" : best_ref["name"],
            "score"      : best_score,
            "stage"      : stage,
            "corners"    : (best_corners.tolist()
                            if best_corners is not None else None),
            "bbox": {
                "x"      : rx  / fw,
                "y"      : ry  / fh,
                "width"  : rw  / fw,
                "height" : rh  / fh,
                "cx"     : (rx + rw / 2) / fw,
                "cy"     : (ry + rh / 2) / fh,
            },
        }

    # -- Visualization ----------------------------------------------

    def draw_result(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw the detection result on a copy of the frame.
        Useful for debugging - save the output to disk or show with cv2.imshow.

        Green quad  = Stage 1 frame corners
        Blue rect   = bounding box
        Yellow dots = quad corner points
        """
        vis = frame.copy()
        fh, fw = vis.shape[:2]

        if not result.get("detected"):
            cv2.putText(
                vis,
                f"NOT DETECTED: {result.get('reason', '')}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            return vis

        # -- Quad outline (Stage 1 result) -------------------------
        corners = result.get("corners")
        if corners:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            for pt in corners:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 255), -1)

        # -- Bounding box -------------------------------------------
        bbox = result["bbox"]
        x  = int(bbox["x"]  * fw)
        y  = int(bbox["y"]  * fh)
        x2 = int((bbox["x"] + bbox["width"])  * fw)
        y2 = int((bbox["y"] + bbox["height"]) * fh)
        cv2.rectangle(vis, (x, y), (x2, y2), (255, 120, 0), 2)

        # -- Label --------------------------------------------------
        label = (
            f"{result['panel_name']}  "
            f"score={result['score']}  "
            f"[{result['stage']}]"
        )
        cv2.putText(
            vis, label,
            (x, max(0, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        return vis


# ======================================================================
#  QUICK TEST - run this file directly to test on a folder of photos
# ======================================================================

if __name__ == "__main__":
    import sys

    REFS_FOLDER = "data/raw/panel"      # folder with IMG_7066.jpg ... IMG_7074.jpg
    TEST_FOLDER = "data/raw/panel"      # using same folder to test detection on itself

    # -- Panel names (must match alphabetical order of reference files) --
    # Files: IMG_7066.jpg ... IMG_7074.jpg  (alphabetical = numerical order)
    # Edit the names below to match each panel's actual history content.
    PANEL_NAMES = [
        "College History Text",        
        "University Dome Building",   
        "Mechanical Engineering",      
        "Gate and Mining Cart",        
        "Hydraulics and Tile Work",    
        "Architecture and Ornament Studies",       
        "Survey Instrument and Classical Facades", 
        "Corinthian Capital and Industrial Machinery", 
        "Steam Engine and Polytechnique Relief",  
    ]

    if not os.path.isdir(REFS_FOLDER):
        print(f"ERROR: Folder '{REFS_FOLDER}/' not found.")
        sys.exit(1)

    # -- Load detector ----------------------------------------------
    detector = PanelDetector(
        references_folder = REFS_FOLDER,
        panel_names       = PANEL_NAMES,
        min_inliers       = 12,
        ratio_threshold   = 0.75,
    )

    # -- Run on all test photos -------------------------------------
    files = sorted(
        f for f in os.listdir(TEST_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    print(f"Testing on {len(files)} images in '{TEST_FOLDER}/'...\n")

    os.makedirs("data/processed/panel", exist_ok=True)
    detected_count = 0

    for fname in files:
        frame = cv2.imread(os.path.join(TEST_FOLDER, fname))
        if frame is None:
            print(f"  SKIP  {fname}")
            continue

        result  = detector.detect(frame)
        vis     = detector.draw_result(frame, result)
        cv2.imwrite(os.path.join("data/processed/panel", fname), vis)

        if result["detected"]:
            detected_count += 1
            print(
                f"  FOUND  {fname:<30} -> "
                f"{result['panel_name']:<30}  "
                f"score={result['score']:3d}  [{result['stage']}]"
            )
        else:
            print(f"  MISS   {fname:<30} -> {result['reason']}")

    print(f"\n{'='*65}")
    print(f"Detected {detected_count} / {len(files)} test images.")
    print(f"Annotated results saved to 'data/processed/panel/'")
    print(f"{'='*65}")
