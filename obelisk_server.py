"""
Obelisk Detection Server
Receives a camera frame from Unity (Quest), runs the IP pipeline,
returns a JSON result with detection info.

Run:  python obelisk_server.py
Then set SERVER_URL in Unity's ObeliskDetectionClient.cs to your PC's IP.
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import traceback

# Import all detection steps from your existing script
from obelisk_detection import (
    step1_preprocess,
    step2_canny,
    step2b_suppress_vegetation,
    step3_detect_lines,
    step4_filter_verticals,
    step5_find_pair,
    step6_polygon,
    step7_colour_refine,
    step8_morphology,
)

app = Flask(__name__)


def detect_obelisk(img):
    """
    Runs the full IP pipeline on an in-memory image (numpy array).
    Returns a dict with detection result and bounding box.
    """
    h, w = img.shape[:2]

    # Use a dummy steps folder (we don't save step images in server mode)
    import tempfile, os
    sf = tempfile.mkdtemp()

    try:
        blurred     = step1_preprocess(img, h, sf)
        edges       = step2_canny(blurred, sf)
        edges       = step2b_suppress_vegetation(edges, img, sf)
        lines       = step3_detect_lines(edges, img, h, w, sf)

        if lines is None:
            return {"detected": False, "reason": "no lines found"}

        vlines = step4_filter_verticals(lines, img, h, w, sf)
        if len(vlines) < 1:
            return {"detected": False, "reason": "no vertical lines"}

        best_pair = step5_find_pair(vlines, img, h, w, sf)
        if best_pair is None:
            return {"detected": False, "reason": "no obelisk pair"}

        polygon_mask = step6_polygon(best_pair, h, w, sf)
        refined      = step7_colour_refine(img, polygon_mask, sf)
        clean        = step8_morphology(refined, sf)

        # Get bounding box from final mask
        pts = cv2.findNonZero(clean)
        if pts is None:
            return {"detected": False, "reason": "empty mask after refinement"}

        rx, ry, rw, rh = cv2.boundingRect(pts)

        # Normalize to 0-1 so Unity can use them regardless of resolution
        return {
            "detected"  : True,
            "bbox": {
                "x"      : rx / w,
                "y"      : ry / h,
                "width"  : rw / w,
                "height" : rh / h,
                "cx"     : (rx + rw / 2) / w,   # center x normalized
                "cy"     : (ry + rh / 2) / h,   # center y normalized
            },
            "image_width" : w,
            "image_height": h,
        }

    except Exception as e:
        traceback.print_exc()
        return {"detected": False, "reason": str(e)}

    finally:
        # Clean up temp step files
        import shutil
        shutil.rmtree(sf, ignore_errors=True)


@app.route('/detect', methods=['POST'])
def detect():
    """
    POST /detect
    Body: raw JPEG bytes
    Returns: JSON detection result
    """
    try:
        img_bytes = request.data
        if not img_bytes:
            return jsonify({"detected": False, "reason": "no image data"}), 400

        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"detected": False, "reason": "could not decode image"}), 400

        print(f"[Server] Received frame: {img.shape[1]}x{img.shape[0]}")
        result = detect_obelisk(img)
        print(f"[Server] Result: {result}")
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"detected": False, "reason": str(e)}), 500


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    print("=" * 50)
    print("Obelisk Detection Server starting...")
    print("Find your PC IP: run 'ipconfig' in CMD")
    print("Set that IP in Unity's ObeliskDetectionClient.cs")
    print("=" * 50)
    # host='0.0.0.0' makes it accessible from Quest over WiFi
    app.run(host='0.0.0.0', port=5000, debug=False)
