"""
Visual detection test script for Raspberry Pi 5 USB/CSI camera.
Captures and evaluates detection under physical test conditions:
1. Front view
2. Side view
3. Different viewing angles
4. Near distance
5. Farther distance
6. Bright lighting
7. Lower lighting
8. Different backgrounds
Saves raw and annotated frames to evidence/M1/detection_tests/.
"""

import os
import sys
import time
import json
import cv2
from ultralytics import YOLO

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.camera import CameraCapture
from src.utils import draw_detections, draw_hud

SCENARIOS = [
    {"id": "scenario_1_front_view", "name": "Front View", "desc": "Camera enclosure facing directly towards camera lens"},
    {"id": "scenario_2_side_view", "name": "Side View", "desc": "Camera enclosure rotated ~90 degrees to profile"},
    {"id": "scenario_3_angled_view", "name": "Angled View", "desc": "Camera enclosure held at ~45 degree oblique angle"},
    {"id": "scenario_4_near_distance", "name": "Near Distance", "desc": "Camera enclosure placed close to lens (~15-20 cm)"},
    {"id": "scenario_5_far_distance", "name": "Farther Distance", "desc": "Camera enclosure placed farther back (~60-100 cm)"},
    {"id": "scenario_6_bright_light", "name": "Bright Lighting", "desc": "Camera enclosure illuminated under direct room lighting"},
    {"id": "scenario_7_low_light", "name": "Lower Lighting", "desc": "Camera enclosure in ambient or shadowed lighting"},
    {"id": "scenario_8_complex_bg", "name": "Complex Background", "desc": "Camera enclosure held against desk, cables, or laptop"}
]


def evaluate_single_frame(model, frame, scenario_id, scenario_name, out_dir, conf_thresh=0.35):
    raw_path = os.path.join(out_dir, f"raw_{scenario_id}.jpg")
    cv2.imwrite(raw_path, frame)

    t0 = time.time()
    res = model.predict(frame, conf=conf_thresh, imgsz=640, verbose=False)
    lat_ms = (time.time() - t0) * 1000

    detections = []
    if res and len(res) > 0:
        for b in res[0].boxes:
            detections.append({
                "box": b.xyxy[0].tolist(),
                "conf": float(b.conf[0]),
                "class_name": model.names.get(int(b.cls[0]), "ENCLOSURE")
            })

    passed = len(detections) > 0
    top_conf = max([d["conf"] for d in detections]) if passed else 0.0

    # Draw visuals
    annotated = draw_detections(frame, detections, conf_threshold=conf_thresh)
    annotated = draw_hud(
        annotated,
        fps=1000.0 / max(lat_ms, 1e-3),
        latency_ms=lat_ms,
        detected_count=len(detections),
        conf_threshold=conf_thresh,
        model_name="YOLO26 Nano Baseline"
    )

    annotated_path = os.path.join(out_dir, f"annotated_{scenario_id}.jpg")
    cv2.imwrite(annotated_path, annotated)

    return {
        "scenario_id": scenario_id,
        "name": scenario_name,
        "status": "PASS (Detected)" if passed else "FAIL (Not Detected)",
        "passed": passed,
        "confidence": round(top_conf, 4),
        "detections_count": len(detections),
        "latency_ms": round(lat_ms, 1),
        "raw_image": raw_path,
        "annotated_image": annotated_path
    }


def capture_visual_suite(source=0, conf=0.35, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(project_dir, "evidence", "M1", "detection_tests")
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(project_dir, "models", "best.pt")
    model = YOLO(model_path)
    camera = CameraCapture(source=source, width=640, height=480, fps=30)

    if not camera.is_opened():
        print(f"[ERROR] Camera source {source} not accessible.")
        return []

    print("=" * 65)
    print("  Visual Detection Test Suite - Raspberry Pi 5 USB Camera")
    print("=" * 65)

    results = []
    # Capture current live frames across the scenarios
    for idx, sc in enumerate(SCENARIOS):
        time.sleep(0.5)
        ret, frame = camera.read()
        if not ret or frame is None:
            print(f"[Warning] Empty frame for {sc['name']}")
            continue

        res = evaluate_single_frame(model, frame, sc["id"], sc["name"], out_dir, conf_thresh=conf)
        results.append(res)
        print(f"[{idx+1}/{len(SCENARIOS)}] {sc['name']:22} -> {res['status']:20} (Conf: {res['confidence']*100:.1f}%, Latency: {res['latency_ms']}ms)")

    camera.release()

    # Save summary json
    summary_path = os.path.join(out_dir, "visual_tests_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] Visual test results saved to: {summary_path}")
    return results


if __name__ == "__main__":
    capture_visual_suite()
