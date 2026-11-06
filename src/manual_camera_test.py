"""
M2 Manual Camera Testing Script for Raspberry Pi 5.
Supports model selection (ONNX Runtime best.onnx or PyTorch best.pt).
Interactive Keyboard Controls:
  - 's' or 'S': Capture and save synchronized pair:
      1. evidence/M2/manual_camera_tests/annotated/test_<seq>_<timestamp>.jpg
      2. evidence/M2/manual_camera_tests/original/test_<seq>_<timestamp>.jpg (clean raw frame)
  - 'q' or 'Q' or ESC: Clean exit and release camera.
Also supports MJPEG web viewer at http://localhost:5000/ with a one-click 'Capture Evidence' button.
"""

import os
import sys
import time
import argparse
import datetime
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.camera import CameraCapture
from src.utils import get_system_info, draw_detections, draw_hud

CLASS_NAMES = {0: "RASPBERY PI CAM ENCLOSURE"}


class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.raw_frame = None
        self.detections = []
        self.condition = threading.Condition()
        self.capture_requested = False

    def set_frame(self, frame_bytes, raw_frame=None, detections=None):
        with self.condition:
            self.frame = frame_bytes
            self.raw_frame = raw_frame
            self.detections = detections or []
            self.condition.notify_all()

    def get_frame(self):
        with self.condition:
            self.condition.wait()
            return self.frame

    def request_capture(self):
        with self.condition:
            self.capture_requested = True


streaming_output = None
global_capture_callback = None


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global streaming_output, global_capture_callback
        if self.path in ['/', '/index.html']:
            html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>M2 Manual Camera Testing - Raspberry Pi 5</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #38bdf8; margin-bottom: 8px; }
        p { color: #94a3b8; margin-top: 0; }
        .box { display: inline-block; background: #1e293b; padding: 12px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        img { width: 100%; max-width: 800px; border-radius: 8px; display: block; }
        .controls { margin-top: 16px; }
        button { background: #0284c7; color: white; border: none; padding: 12px 24px; font-size: 1rem; font-weight: 600; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #0369a1; }
        .status { margin-top: 10px; font-size: 0.9rem; color: #4ade80; min-height: 24px; }
    </style>
</head>
<body>
    <h1>M2 Manual Camera Verification</h1>
    <p>Synchronized Paired Capture: Clean Original Frame + Inference Annotated Frame</p>
    <div class="box"><img src="/stream.mjpg" alt="Live Camera Feed"></div>
    <div class="controls">
        <button onclick="triggerCapture()">&#128247; Capture Paired Test Images (S)</button>
        <div id="status" class="status"></div>
    </div>
    <script>
        function triggerCapture() {
            document.getElementById('status').innerText = 'Capturing...';
            fetch('/capture')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').innerText = 'Saved: ' + data.annotated + ' & ' + data.original;
                })
                .catch(err => {
                    document.getElementById('status').innerText = 'Capture error: ' + err;
                });
        }
    </script>
</body>
</html>"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    frame = streaming_output.get_frame()
                    if frame is None:
                        continue
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception:
                pass
        elif self.path == '/capture':
            if global_capture_callback:
                res = global_capture_callback()
                import json
                resp = json.dumps(res).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            else:
                self.send_error(500, "Capture callback not ready")
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_img, r, (dw, dh)


def preprocess_onnx(frame, input_shape=(640, 640)):
    padded_img, scale, (pad_w, pad_h) = letterbox(frame, new_shape=input_shape)
    rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    return np.ascontiguousarray(tensor), scale, (pad_w, pad_h)


def postprocess_onnx(ort_output, scale, pads, orig_shape, conf_thresh=0.35, iou_thresh=0.70):
    predictions = np.squeeze(ort_output[0]).T
    boxes_xywh = []
    scores = []
    pad_w, pad_h = pads
    h_orig, w_orig = orig_shape[:2]

    score_col = predictions[:, 4]
    mask = score_col >= conf_thresh
    filtered_preds = predictions[mask]

    for row in filtered_preds:
        cx, cy, w, h, score = row
        x1 = (cx - w / 2 - pad_w) / scale
        y1 = (cy - h / 2 - pad_h) / scale
        x2 = (cx + w / 2 - pad_w) / scale
        y2 = (cy + h / 2 - pad_h) / scale

        x1 = max(0.0, min(float(w_orig), x1))
        y1 = max(0.0, min(float(h_orig), y1))
        x2 = max(0.0, min(float(w_orig), x2))
        y2 = max(0.0, min(float(h_orig), y2))

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        boxes_xywh.append([int(x1), int(y1), int(bw), int(bh)])
        scores.append(float(score))

    if not boxes_xywh:
        return []

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=conf_thresh, nms_threshold=iou_thresh)
    detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            bx, by, bw, bh = boxes_xywh[idx]
            detections.append({
                "box": [bx, by, bx + bw, by + bh],
                "conf": scores[idx],
                "class_name": CLASS_NAMES.get(0, "RASPBERY PI CAM ENCLOSURE")
            })
    return detections


def parse_args():
    parser = argparse.ArgumentParser(description="M2 Manual Camera Test with Paired Capture")
    parser.add_argument("--model", type=str, default=os.path.join(project_dir, "models", "onnx", "best.onnx"),
                        help="Path to model weights (.onnx or .pt)")
    parser.add_argument("--source", type=str, default="/dev/video0",
                        help="Video capture source (/dev/video0 or 0 or csi)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold (default: 0.70)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input inference size (default: 640)")
    parser.add_argument("--web", action="store_true", help="Launch web server on port 5000")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--no-view", dest="view", action="store_false", help="Run without OpenCV GUI window")
    parser.add_argument("--auto-capture-count", type=int, default=0,
                        help="Automatically capture N pairs on start for headless testing")
    parser.add_argument("--auto-capture-interval", type=float, default=2.0,
                        help="Interval in seconds between auto captures")
    return parser.parse_args()


def main():
    global streaming_output, global_capture_callback
    args = parse_args()

    # Create target evidence directories
    annotated_dir = os.path.join(project_dir, "evidence", "M2", "manual_camera_tests", "annotated")
    original_dir = os.path.join(project_dir, "evidence", "M2", "manual_camera_tests", "original")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)

    print("=" * 65)
    print("  M2 Manual Camera Verification & Paired Evidence Capture")
    print("=" * 65)
    print(f"Model path:    {args.model}")
    print(f"Camera source: {args.source}")
    print(f"Confidence:    {args.conf} | IoU: {args.iou}")
    print(f"Annotated dir: {annotated_dir}")
    print(f"Original dir:  {original_dir}")

    # Determine backend
    is_onnx = args.model.endswith(".onnx")
    if is_onnx:
        import onnxruntime as ort
        print(f"[Backend] Initializing ONNX Runtime CPU Provider for {args.model}...")
        session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        model_tag = "YOLO26 Nano (ONNX Runtime)"
    else:
        from ultralytics import YOLO
        print(f"[Backend] Initializing PyTorch Ultralytics for {args.model}...")
        pt_model = YOLO(args.model)
        model_tag = "YOLO26 Nano (PyTorch)"

    # Open camera
    print(f"[Camera] Connecting to {args.source}...")
    camera = CameraCapture(source=args.source, width=640, height=480, fps=30)
    if not camera.is_opened():
        print(f"[ERROR] Could not open camera {args.source}. Aborting.")
        sys.exit(1)

    streaming_output = StreamingOutput()

    # Web server if requested
    if args.web:
        server_address = ('0.0.0.0', args.port)
        server = ThreadingHTTPServer(server_address, StreamingHandler)
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        print(f"[Web] Streaming active at: http://localhost:{args.port}/")

    can_gui = args.view and bool(os.environ.get("DISPLAY"))
    window_name = f"M2 Manual Test - {model_tag}"
    if can_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)
        except Exception:
            can_gui = False

    seq_counter = 1
    # Check existing sequence numbers
    existing_annotated = [f for f in os.listdir(annotated_dir) if f.startswith("test_")]
    if existing_annotated:
        seq_nums = []
        for f in existing_annotated:
            parts = f.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                seq_nums.append(int(parts[1]))
        if seq_nums:
            seq_counter = max(seq_nums) + 1

    last_raw_frame = None
    last_annotated_frame = None
    capture_lock = threading.Lock()

    def do_save_pair():
        nonlocal seq_counter, last_raw_frame, last_annotated_frame
        with capture_lock:
            if last_raw_frame is None or last_annotated_frame is None:
                return {"status": "error", "message": "No frame captured yet"}

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            annotated_fname = f"test_{seq_counter:03d}_{ts}.jpg"
            original_fname = f"test_{seq_counter:03d}_{ts}.jpg"

            annotated_path = os.path.join(annotated_dir, annotated_fname)
            original_path = os.path.join(original_dir, original_fname)

            # Save clean raw original (no bounding boxes, no HUD)
            cv2.imwrite(original_path, last_raw_frame)
            # Save annotated frame with detections & HUD
            cv2.imwrite(annotated_path, last_annotated_frame)

            print("\n" + "#" * 65)
            print(f" [CAPTURE SAVED] Sequence #{seq_counter:03d}")
            print(f"  Annotated: {annotated_path}")
            print(f"  Original:  {original_path}")
            print("#" * 65 + "\n")

            seq_counter += 1
            return {
                "status": "success",
                "annotated": annotated_path,
                "original": original_path,
                "sequence": seq_counter - 1
            }

    global_capture_callback = do_save_pair

    fps = 0.0
    prev_time = time.time()
    auto_captures_done = 0
    last_auto_capture_time = time.time()

    print("\n[Controls]")
    print("  's' or 'S' : Save paired evidence images (annotated + raw original)")
    print("  'q' or ESC : Exit\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            clean_raw = frame.copy()
            t0 = time.time()

            if is_onnx:
                tensor, scale, pads = preprocess_onnx(clean_raw, input_shape=(args.imgsz, args.imgsz))
                t_infer_0 = time.time()
                ort_out = session.run([output_name], {input_name: tensor})
                infer_ms = (time.time() - t_infer_0) * 1000
                detections = postprocess_onnx(ort_out, scale, pads, clean_raw.shape,
                                              conf_thresh=args.conf, iou_thresh=args.iou)
            else:
                t_infer_0 = time.time()
                results = pt_model(clean_raw, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
                infer_ms = (time.time() - t_infer_0) * 1000
                detections = []
                for r in results:
                    for b in r.boxes:
                        bx = [int(v) for v in b.xyxy[0].tolist()]
                        detections.append({
                            "box": bx,
                            "conf": float(b.conf[0]),
                            "class_name": CLASS_NAMES.get(int(b.cls[0]), "ENCLOSURE")
                        })

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            # Create annotated copy
            annotated = draw_detections(clean_raw.copy(), detections, conf_threshold=args.conf)
            annotated = draw_hud(
                annotated,
                fps=fps,
                latency_ms=infer_ms,
                detected_count=len(detections),
                conf_threshold=args.conf,
                model_name=model_tag
            )

            with capture_lock:
                last_raw_frame = clean_raw
                last_annotated_frame = annotated

            # Update web stream buffer
            _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            streaming_output.set_frame(buffer.tobytes(), raw_frame=clean_raw, detections=detections)

            # Auto-capture logic if requested
            if args.auto_capture_count > 0 and auto_captures_done < args.auto_capture_count:
                if (curr_time - last_auto_capture_time) >= args.auto_capture_interval:
                    do_save_pair()
                    auto_captures_done += 1
                    last_auto_capture_time = curr_time
                    if auto_captures_done >= args.auto_capture_count:
                        print(f"[Auto-Capture] Finished capturing {args.auto_capture_count} test pairs.")
                        break

            # Check web capture request
            if streaming_output.capture_requested:
                streaming_output.capture_requested = False
                do_save_pair()

            # GUI display & key handling
            if can_gui:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("[User] Exit key received. Quitting...")
                    break
                elif key in [ord('s'), ord('S')]:
                    do_save_pair()
            else:
                # Small pause to avoid pegging loop if headless without auto-capture
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[User] Keyboard interrupt received.")
    finally:
        camera.release()
        if can_gui:
            cv2.destroyAllWindows()
        print("[Camera] Camera released. Exiting clean.")


if __name__ == "__main__":
    main()
