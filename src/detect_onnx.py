"""
Standalone real-time object detection using optimized ONNX Runtime on Raspberry Pi 5.
Loads models/onnx/best.onnx directly via onnxruntime.InferenceSession.
Performs manual image letterboxing, tensor normalization, anchor-free box decoding,
and Non-Maximum Suppression (NMS) with zero dependency on PyTorch during inference.
"""

import os
import sys
import time
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import cv2
import onnxruntime as ort

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.camera import CameraCapture
from src.utils import get_system_info, draw_detections, draw_hud

CLASS_NAMES = {0: "RASPBERY PI CAM ENCLOSURE"}


class StreamingOutput:
    """Thread-safe frame buffer for optional web streaming."""
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def set_frame(self, frame_bytes):
        with self.condition:
            self.frame = frame_bytes
            self.condition.notify_all()

    def get_frame(self):
        with self.condition:
            self.condition.wait()
            return self.frame


streaming_output = None


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/index.html']:
            content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Raspberry Pi 5 - ONNX Runtime Detection</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #38bdf8; margin: 0 0 6px 0; font-size: 1.5rem; }
        p { color: #94a3b8; margin: 0 0 16px 0; font-size: 0.9rem; }
        .box { display: inline-block; background: #1e293b; padding: 12px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
        img { width: 100%; max-width: 800px; border-radius: 8px; display: block; }
        .badge { display: inline-block; padding: 4px 12px; margin: 6px 4px; border-radius: 16px; font-size: 0.8rem; font-weight: 600; background: #1e293b; color: #38bdf8; border: 1px solid #334155; }
        .badge-green { color: #4ade80; border-color: #166534; }
    </style>
</head>
<body>
    <h1>Raspberry Pi 5 - Optimized ONNX Runtime Stream</h1>
    <p>Real-Time Edge Inference: <strong>YOLO26 Nano (models/onnx/best.onnx)</strong></p>
    <div class="box"><img src="/stream.mjpg" alt="Live ONNX Feed"></div>
    <div>
        <span class="badge badge-green">&#9679; ONNX ACTIVE</span>
        <span class="badge">Engine: ONNX Runtime 1.31.1</span>
        <span class="badge">Provider: CPUExecutionProvider</span>
        <span class="badge">Resolution: 640x640</span>
    </div>
</body>
</html>"""
            body = content.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes image to target resolution with aspect-ratio preserving letterbox padding.
    Returns:
        padded_img, scale, (pad_left, pad_top)
    """
    shape = img.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_img, r, (dw, dh)


def preprocess(frame, input_shape=(640, 640)):
    """Prepares OpenCV BGR frame for ONNX Runtime BCHW float32 input."""
    padded_img, scale, (pad_w, pad_h) = letterbox(frame, new_shape=input_shape)
    # BGR to RGB
    rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    # HWC to CHW and normalize to [0, 1]
    tensor = rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension -> [1, 3, 640, 640]
    return np.ascontiguousarray(tensor), scale, (pad_w, pad_h)


def postprocess(ort_output, scale, pads, orig_shape, conf_thresh=0.35, iou_thresh=0.70):
    """
    Decodes YOLO26 Nano output [1, 5, 8400] to scaled bounding boxes and applies NMS.
    Returns list of dicts: {'box': [x1, y1, x2, y2], 'conf': float, 'class_name': str}
    """
    predictions = np.squeeze(ort_output[0]).T  # shape: (8400, 5)

    boxes_xywh = []
    scores = []

    pad_w, pad_h = pads
    h_orig, w_orig = orig_shape[:2]

    # Vectorized filter by confidence threshold
    score_col = predictions[:, 4]
    mask = score_col >= conf_thresh
    filtered_preds = predictions[mask]

    for row in filtered_preds:
        cx, cy, w, h, score = row
        # Map center-xywh back to original unpadded frame
        x1 = (cx - w / 2 - pad_w) / scale
        y1 = (cy - h / 2 - pad_h) / scale
        x2 = (cx + w / 2 - pad_w) / scale
        y2 = (cy + h / 2 - pad_h) / scale

        # Clip to image boundary
        x1 = max(0.0, min(float(w_orig), x1))
        y1 = max(0.0, min(float(h_orig), y1))
        x2 = max(0.0, min(float(w_orig), x2))
        y2 = max(0.0, min(float(h_orig), y2))

        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)

        boxes_xywh.append([int(x1), int(y1), int(box_w), int(box_h)])
        scores.append(float(score))

    if not boxes_xywh:
        return []

    # Non-Maximum Suppression via OpenCV
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=conf_thresh, nms_threshold=iou_thresh)

    detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            bx, by, bw, bh = boxes_xywh[idx]
            detections.append({
                "box": [bx, by, bx + bw, by + bh],
                "conf": scores[idx],
                "class_name": CLASS_NAMES.get(0, "ENCLOSURE")
            })

    return detections


def parse_args():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 ONNX Runtime Detection")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(project_dir, "models", "onnx", "best.onnx"),
        help="Path to optimized ONNX model"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera source (0 for USB webcam, 'picam' for CSI camera module)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS IoU threshold (default: 0.70)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference resolution (default: 640)"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        default=True,
        help="Display GUI window"
    )
    parser.add_argument(
        "--no-view",
        dest="view",
        action="store_false",
        help="Run in headless mode"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start local MJPEG web streaming server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for web stream (default: 5000)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(project_dir, "evidence", "M2", "detections"),
        help="Directory to save snapshot captures"
    )
    return parser.parse_args()


def main():
    global streaming_output
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 65)
    print("  Raspberry Pi 5: Optimized ONNX Runtime Detection Pipeline")
    print("=" * 65)

    sys_info = get_system_info()
    print(f"[System] Platform: {sys_info['rpi_model']} ({sys_info['machine']})")
    print(f"[System] ONNX Runtime Version: {ort.__version__}")

    # 1. Initialize ONNX Runtime Session
    if not os.path.exists(args.model):
        print(f"[ERROR] ONNX model not found at {args.model}")
        sys.exit(1)

    print(f"[ONNX] Loading model from: {args.model}...")
    t_load_0 = time.time()
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    load_time_ms = (time.time() - t_load_0) * 1000
    active_providers = session.get_providers()
    print(f"[ONNX] Model loaded in {load_time_ms:.1f}ms | Providers: {active_providers}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[ONNX] Node mapping: input='{input_name}', output='{output_name}'")

    # 2. Connect to Camera
    print(f"[Camera] Connecting to video source: '{args.source}'...")
    camera = CameraCapture(source=args.source, width=640, height=480, fps=30)
    if not camera.is_opened():
        print(f"[ERROR] Could not open camera source '{args.source}'. Exiting.")
        sys.exit(1)

    # 3. Web Streamer Setup if requested
    server = None
    if args.web:
        streaming_output = StreamingOutput()
        server_address = ('0.0.0.0', args.port)
        server = ThreadingHTTPServer(server_address, StreamingHandler)
        web_thread = threading.Thread(target=server.serve_forever, daemon=True)
        web_thread.start()
        print(f"\n[Web Stream] Live ONNX feed active at: http://localhost:{args.port}/")

    # 4. GUI & Execution Loop
    can_gui = args.view and bool(os.environ.get("DISPLAY"))
    window_name = "Edge AI - Raspberry Pi 5 (ONNX Runtime)"
    if can_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)
        except Exception:
            can_gui = False

    fps = 0.0
    prev_time = time.time()
    consecutive_empty = 0
    last_status_print = time.time()
    live_frame_path = os.path.join(args.save_dir, "live_frame_onnx.jpg")

    print("[Inference] Starting optimized ONNX detection loop...")
    print("  Controls: [Q] Quit | [S] Save Snapshot | Auto-saving frames\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                consecutive_empty += 1
                if consecutive_empty >= 10:
                    print("\n[Camera] Stream ended or disconnected.")
                    break
                time.sleep(0.05)
                continue
            consecutive_empty = 0

            # Step A: Preprocess
            t_pre_0 = time.time()
            tensor_input, scale, pads = preprocess(frame, input_shape=(args.imgsz, args.imgsz))

            # Step B: Model Inference (ONNX Runtime)
            t_infer_0 = time.time()
            ort_outputs = session.run([output_name], {input_name: tensor_input})
            t_infer_1 = time.time()
            inference_ms = (t_infer_1 - t_infer_0) * 1000

            # Step C: Decode & NMS
            detections = postprocess(ort_outputs, scale, pads, frame.shape,
                                     conf_thresh=args.conf, iou_thresh=args.iou)

            # Pipeline FPS calculation
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            # Step D: Visual HUD Rendering
            annotated = draw_detections(frame, detections, conf_threshold=args.conf)
            annotated = draw_hud(
                annotated,
                fps=fps,
                latency_ms=inference_ms,
                detected_count=len(detections),
                conf_threshold=args.conf,
                model_name="YOLO26 Nano (ONNX Runtime)"
            )

            # Update live image buffer
            cv2.imwrite(live_frame_path, annotated)

            # Web stream update
            if args.web and streaming_output is not None:
                ret_enc, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret_enc:
                    streaming_output.set_frame(jpeg.tobytes())

            # Native display or terminal heartbeat
            if can_gui:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("\n[Inference] Quit requested by user.")
                    break
                elif key in [ord('s'), ord('S')]:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    snap_name = f"onnx_det_{timestamp}.jpg"
                    snap_path = os.path.join(args.save_dir, snap_name)
                    cv2.imwrite(snap_path, annotated)
                    print(f"[Snapshot] Saved ONNX detection frame to: {snap_path}")
            else:
                if len(detections) > 0:
                    top_conf = max([d['conf'] for d in detections])
                    print(f"[ONNX DETECTION] Enclosure detected! Conf: {top_conf*100:.1f}%, Latency: {inference_ms:.1f}ms, Pipeline FPS: {fps:.1f}")
                elif time.time() - last_status_print > 3.0:
                    print(f"[ONNX Running] Latency: {inference_ms:.1f}ms | Pipeline FPS: {fps:.1f} | Detections: {len(detections)}")
                    last_status_print = time.time()

    except KeyboardInterrupt:
        print("\n[Inference] Interrupted by keyboard.")
    finally:
        camera.release()
        if can_gui:
            cv2.destroyAllWindows()
        if server:
            server.shutdown()
        print("[Inference] ONNX pipeline shutdown complete.")


if __name__ == '__main__':
    main()
