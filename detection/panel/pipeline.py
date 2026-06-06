import cv2
import os
import numpy as np
import sys

REFS_FOLDER = "data/raw/panel"    
TEST_FOLDER = "data/raw/panel"     
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


class PanelDetector:
    CANONICAL_WIDTH = 600
    CANONICAL_HEIGHT = 960

    def __init__(self,references_folder: str,panel_names: list = None,min_inliers: int  = 12,ratio_threshold: float = 0.75):
        self.min_inliers     = min_inliers
        self.ratio_threshold = ratio_threshold

        self._sift = cv2.SIFT_create(nfeatures= 0, contrastThreshold = 0.04, edgeThreshold= 10,)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._refs = []

        if not os.path.isdir(references_folder):
            raise FileNotFoundError(f"References folder not found. Create another one")

        ref_files = sorted( f for f in os.listdir(references_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

        if not ref_files:
            raise FileNotFoundError(f"No images found")

        for idx, fname in enumerate(ref_files):
            path = os.path.join(references_folder, fname)
            img  = cv2.imread(path)

            if img is None:
                print(f" cannot read '{path}', skipped")
                continue

            kp, desc = self._sift.detectAndCompute(img, None)
            if desc is None or len(kp) < 10:
                print(f"too few keypoints in '{fname}', skipped")
                continue

            name = (panel_names[idx]
                    if panel_names and idx < len(panel_names)
                    else f"Panel {idx}")

            self._refs.append({"id": idx,"name" : name,"file" : fname,"kp": kp,"desc" : desc,})

        if not self._refs:
            raise ValueError("No valid reference images could be loaded. ")

        print("panels registered")

    # Panel Localization

    def _locate_panels(self, frame: np.ndarray) -> list:
        fh, fw = frame.shape[:2]

        WORK_WIDTH = 800
        scale  = WORK_WIDTH / fw
        WORK_HEIGHT = int(fh * scale)
        small  = cv2.resize(frame, (WORK_WIDTH, WORK_HEIGHT))
        gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        col_profile = gray.mean(axis=0).astype(float)   
        k = 15
        col_smooth = np.convolve(col_profile, np.ones(k) / k, mode='same')

        DARK_THRESHOLD = 100
        is_dark = col_smooth < DARK_THRESHOLD

        bright_runs = []  
        start = None
        for i, dark in enumerate(is_dark):
            if not dark and start is None:
                start = i
            elif dark and start is not None:
                bright_runs.append((start, i))
                start = None
        if start is not None:
            bright_runs.append((start, WORK_WIDTH))

        if not bright_runs:
            return []

        cx_frame = WORK_WIDTH / 2.0
        scored   = []
        for x0, x1 in bright_runs:
            width = x1 - x0
            if width < WORK_WIDTH * 0.10:  
                continue
            cx_run     = (x0 + x1) / 2.0
            centrality = 1.0 - abs(cx_run - cx_frame) / cx_frame
            score      = width * (0.5 + 0.5 * centrality)
            scored.append((score, x0, x1))

        if not scored:
            return []

        scored.sort(reverse=True)
        _, col_x0, col_x1 = scored[0]   
        col_slice = gray[:, col_x0:col_x1]
        row_profile = col_slice.mean(axis=1).astype(float) 
        k_row = 5    
        row_smooth  = np.convolve(row_profile, np.ones(k_row) / k_row, mode='same')

        FRAME_DARK = 60  
        MIN_BORDER = 8  

        is_dark_row = row_smooth < FRAME_DARK

        dark_runs = []
        run_start = None
        for i, dark in enumerate(is_dark_row):
            if dark and run_start is None:
                run_start = i
            elif not dark and run_start is not None:
                dark_runs.append((run_start, i))
                run_start = None
        if run_start is not None:
            dark_runs.append((run_start, WORK_HEIGHT))

        sig_dark = [(y0, y1) for y0, y1 in dark_runs if y1 - y0 >= MIN_BORDER]

        if len(sig_dark) >= 2:
            row_y0 = sig_dark[0][1]     
            row_y1 = sig_dark[-1][0]    
        elif len(sig_dark) == 1:
            y0d, y1d = sig_dark[0]
            if (y0d + y1d) / 2.0 < WORK_HEIGHT / 2:  
                row_y0 = y1d
                row_y1 = WORK_HEIGHT
            else:                             
                row_y0 = 0
                row_y1 = y0d
        else:
            bright_row_idx = np.where(row_smooth > 100)[0]
            if len(bright_row_idx) == 0:
                row_y0, row_y1 = 0, WORK_HEIGHT
            else:
                row_y0 = int(bright_row_idx[0])
                row_y1 = int(bright_row_idx[-1])

        row_y0 = max(0, row_y0)
        row_y1 = min(WORK_HEIGHT, row_y1)
        if row_y1 <= row_y0:
            row_y0, row_y1 = 0, WORK_HEIGHT

        pad_x = int((col_x1 - col_x0) * 0.03)
        pad_y = int((row_y1 - row_y0) * 0.02)

        x0_orig = max(0,  int((col_x0 - pad_x) / scale))
        x1_orig = min(fw, int((col_x1 + pad_x) / scale))
        y0_orig = max(0,  int((row_y0 - pad_y) / scale))
        y1_orig = min(fh, int((row_y1 + pad_y) / scale))

        corners = self._order_corners(np.array([[x0_orig, y0_orig], [x1_orig, y0_orig],[x1_orig, y1_orig], [x0_orig, y1_orig]],dtype=np.float32))

        crop = self._deskew(frame, corners)
        return [(crop, corners)]


    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        pts = pts.reshape(4, 2)
        s   = pts.sum(axis=1)         
        d   = np.diff(pts, axis=1).ravel()  
        tl  = pts[np.argmin(s)]
        br  = pts[np.argmax(s)]
        tr  = pts[np.argmin(d)]
        bl  = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _deskew(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        dst = np.array([[0, 0], [self.CANONICAL_WIDTH - 1, 0],[self.CANONICAL_WIDTH - 1, self.CANONICAL_HEIGHT - 1],[0,self.CANONICAL_HEIGHT - 1],], dtype=np.float32)

        H, _ = cv2.findHomography(corners, dst)
        if H is None:
            return cv2.resize(frame, (self.CANONICAL_WIDTH, self.CANONICAL_HEIGHT))

        return cv2.warpPerspective(frame, H, (self.CANONICAL_WIDTH, self.CANONICAL_HEIGHT))

    #  Panel Identification

    def _identify_panel(self, img: np.ndarray):
        kp_img, desc_img = self._sift.detectAndCompute(img, None)

        if desc_img is None or len(kp_img) < 4:
            return None, 0, None

        best_id = None
        best_inliers = 0
        best_ref= None

        for ref in self._refs:
            try:
                raw_matches = self._matcher.knnMatch(ref["desc"], desc_img, k=2)
            except cv2.error:
                continue
            good = [m for pair in raw_matches if len(pair) == 2 for m, n in [(pair[0], pair[1])] if m.distance < self.ratio_threshold * n.distance]

            if len(good) < max(4, self.min_inliers // 2):
                continue

            src_pts = np.float32([ref["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts,cv2.RANSAC, ransacReprojThreshold=5.0)

            if mask is None:
                continue

            inliers = int(mask.ravel().sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best_id = ref["id"]
                best_ref = ref

        if best_inliers >= self.min_inliers:
            return best_id, best_inliers, best_ref

        return None, best_inliers, None

    #  FULL PIPELINE

    def detect(self, frame: np.ndarray) -> dict:

        fh, fw = frame.shape[:2]
        best_id = None
        best_score= 0
        best_corners = None
        best_ref = None
        stage = "full_frame_fallback"

        candidates = self._locate_panels(frame)

        for crop, corners in candidates:
            pid, score, ref = self._identify_panel(crop)
            if score > best_score:
                best_score = score
                best_id = pid
                best_corners = corners
                best_ref = ref
                stage = "localized"

        if best_id is None:
            pid, score, ref = self._identify_panel(frame)
            if score > best_score:
                best_score = score
                best_id = pid
                best_corners = None
                best_ref = ref
                stage= "full_frame_fallback"

        if best_id is None:
            return {"detected" : False,"reason" : ( f"no panel matched above threshold, (best score={best_score}, need {self.min_inliers})"),}

        # Build bounding box 
        if best_corners is not None:
            xs = best_corners[:, 0]
            ys = best_corners[:, 1]
        else:
            xs = np.array([0.0, float(fw), float(fw), 0.0])
            ys = np.array([0.0, 0.0, float(fh), float(fh)])

        rx,  ry  = float(xs.min()), float(ys.min())
        rx2, ry2 = float(xs.max()), float(ys.max())
        rw,  rh  = rx2 - rx, ry2 - ry

        return {"detected": True, "panel_id": best_id, "panel_name": best_ref["name"], "score": best_score, "stage": stage, "corners": (best_corners.tolist() if best_corners is not None else None), "bbox": {"x": rx / fw, "y": ry / fh, "width": rw / fw, "height": rh / fh, "cx": (rx + rw / 2) / fw, "cy": (ry + rh / 2) / fh}}

    def draw_result(self, frame: np.ndarray, result: dict) -> np.ndarray:
        vis = frame.copy()
        fh, fw = vis.shape[:2]

        if not result.get("detected"):
            cv2.putText(vis,f"NOT DETECTED: {result.get('reason', '')}",(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return vis
        
        corners = result.get("corners")
        if corners:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            for pt in corners:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 255), -1)

        bbox = result["bbox"]
        x  = int(bbox["x"]  * fw)
        y  = int(bbox["y"]  * fh)
        x2 = int((bbox["x"] + bbox["width"])  * fw)
        y2 = int((bbox["y"] + bbox["height"]) * fh)
        cv2.rectangle(vis, (x, y), (x2, y2), (255, 120, 0), 2)

        label = (f"{result['panel_name']}, score={result['score']},[{result['stage']}]")
        cv2.putText(vis, label,(x, max(0, y - 12)),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return vis

if __name__ == "__main__":

    if not os.path.isdir(REFS_FOLDER):
        print(f"ERROR: Folder '{REFS_FOLDER}/' not found.")
        sys.exit(1)

    detector = PanelDetector(references_folder = REFS_FOLDER,panel_names= PANEL_NAMES,min_inliers= 12,ratio_threshold = 0.75,)

    files = sorted(f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    os.makedirs("data/processed/panel", exist_ok=True)
    detected_count = 0

    for fname in files:
        frame = cv2.imread(os.path.join(TEST_FOLDER, fname))
        if frame is None:
            print(f"Skip{fname}")
            continue

        result = detector.detect(frame)
        vis = detector.draw_result(frame, result)
        cv2.imwrite(os.path.join("data/processed/panel", fname), vis)

        if result["detected"]:
            detected_count += 1
            print(f"Found  {fname:<30}, {result['panel_name']:<30}")
        else:
            print(f" Miss{fname:<30}, {result['reason']}")

    print(f"Detected {detected_count} / {len(files)} test images.")