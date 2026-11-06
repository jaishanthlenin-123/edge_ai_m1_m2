"""
Real-time object detection inference script for Raspberry Pi 5 and Desktop.
Loads trained YOLO26 Nano weights (best.pt or best.onnx) and runs detection.
Supports native GUI windows, MJPEG web browser streaming, and headless console logging.
"""

import os
import sys
import time
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
from ultralytics import YOLO

# Add parent directory to path so relative imports work seamlessly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.camera import CameraCapture
from src.utils import get_system_info, draw_detections, draw_hud


class StreamingOutput:
    """Thread-safe frame buffer for MJPEG web streaming."""
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
    """HTTP Request Handler serving live MJPEG stream and responsive HTML viewer."""
    def do_GET(self):
        if self.path in ['/', '/index.html']:
            content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Raspberry Pi 5 - Camera Enclosure Detection</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #111418;
            color: #e0e0e0;
            text-align: center;
            margin: 0;
            padding: 24px 16px;
        }
        h1 { margin: 0 0 8px 0; font-size: 1.6rem; color: #38bdf8; font-weight: 700; }
        p { color: #94a3b8; font-size: 0.95rem; margin: 0 0 20px 0; }
        .feed-card {
            display: inline-block;
            background: #1e293b;
            padding: 12px;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            border: 1px solid #334155;
            max-width: 96vw;
        }
        img {
            width: 100%;
            max-width: 800px;
            height: auto;
            border-radius: 8px;
            display: block;
            background: #0f172a;
        }
        .badges { margin-top: 16px; }
        .badge {
            display: inline-block;
            padding: 6px 14px;
            margin: 4px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            background: #0f172a;
            color: #38bdf8;
            border: 1px solid #1e293b;
        }
        .badge-green { color: #4ade80; border-color: #14532d; }
    </style>
</head>
<body>
    <h1>Raspberry Pi 5 - Edge AI Live Camera Feed</h1>
    <p>Real-Time Object Detection: <strong>Raspberry Pi Camera Enclosure</strong></p>
    <div class="feed-card">
        <img src="/stream.mjpg" alt="Live Camera Feed with YOLO26 Nano Detection">
    </div>
    <div class="badges">
        <span class="badge badge-green">&#9679; LIVE STREAM</span>
        <span class="badge">Model: YOLO26 Nano</span>
        <span class="badge">Class: RASPBERY PI CAM ENCLOSURE</span>
        <span class="badge">Platform: Raspberry Pi 5 CPU</span>
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
        # Suppress HTTP access logging to keep terminal output clean
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Raspberry Pi Camera Enclosure Real-Time Detection")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(project_dir, "models", "best.pt"),
        help="Path to trained model weights (.pt or .onnx)"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: camera index (0, 1), 'picam' for RPi Camera Module, or video path"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        default=True,
        help="Display live GUI window"
    )
    parser.add_argument(
        "--no-view",
        dest="view",
        action="store_false",
        help="Run in headless mode (no GUI window)"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start local MJPEG web streaming server to view feed in browser"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Web streaming server port (default: 5000)"
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
    print("  Raspberry Pi Camera Enclosure Detection - Edge Inference")
    print("=" * 65)

    sys_info = get_system_info()
    print(f"[System] OS: {sys_info['os']} ({sys_info['machine']})")
    if sys_info['is_raspberry_pi']:
        print(f"[System] Platform: {sys_info['rpi_model']}")

    # 1. Verify and Load Model
    if not os.path.exists(args.model):
        alt_model = args.model.replace(".pt", ".onnx") if args.model.endswith(".pt") else args.model.replace(".onnx", ".pt")
        if os.path.exists(alt_model):
            print(f"[Model] Specified model '{args.model}' not found. Using alternative: '{alt_model}'")
            args.model = alt_model
        else:
            print(f"[ERROR] Model file not found at: {args.model}")
            sys.exit(1)

    print(f"[Model] Loading model from: {args.model}...")
    t_load = time.time()
    try:
        model = YOLO(args.model)
        print(f"[Model] Model loaded in {(time.time() - t_load)*1000:.1f}ms")
        print(f"[Model] Target Classes: {model.names}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # 2. Open Camera Source
    print(f"[Camera] Connecting to source: '{args.source}'...")
    camera = CameraCapture(source=args.source, width=640, height=480, fps=30)
    if not camera.is_opened():
        print("\n[CAMERA NOTICE] No active video stream or camera device could be opened.")
        print("  - Raspberry Pi Camera Module (CSI): Check 22-pin ribbon cable in CAM/DISP 0 or 1.")
        print("  - USB Camera: Check USB connection and device index with 'v4l2-ctl --list-devices'.")
        print("  - Video File: Pass '--source <video_file>' to test with a pre-recorded clip.")
        print("[Inference] Exiting because no video source is available.\n")
        sys.exit(1)

    # 3. Setup Web Server if requested
    server = None
    if args.web:
        streaming_output = StreamingOutput()
        server_address = ('0.0.0.0', args.port)
        server = ThreadingHTTPServer(server_address, StreamingHandler)
        web_thread = threading.Thread(target=server.serve_forever, daemon=True)
        web_thread.start()
        print(f"\n[Web Stream] Live browser feed active at: http://localhost:{args.port}/")
        print(f"[Web Stream] (Or open http://<your-pi-ip>:{args.port}/ on any device on your network)\n")

    # 4. GUI Setup & Environment Check
    can_gui = args.view and bool(os.environ.get("DISPLAY"))
    window_name = "Edge AI - Raspberry Pi Camera Enclosure Detection"
    if can_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)
        except Exception as e:
            print(f"[GUI NOTICE] Graphical display window could not be opened ({e}). Running headless.")
            can_gui = False
    elif args.view and not args.web:
        print("[GUI NOTICE] Graphical display server not detected ($DISPLAY is empty).")
        print("  -> Live detection running in console mode.")
        print("  -> Latest frame is saved continuously to: evidence/M2/detections/live_frame.jpg")
        print("  -> Tip: Run with '--web' to view live video in your browser.\n")

    conf_thresh = args.conf
    fps = 0.0
    prev_time = time.time()
    consecutive_errors = 0
    last_status_time = time.time()
    live_frame_path = os.path.join(args.save_dir, "live_frame.jpg")

    print("[Inference] Starting live detection loop...")
    print("  Controls: [Ctrl+C] Quit | Auto-saving frames to evidence/M2/detections/\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    print(f"\n[CameraCapture] Reached {consecutive_errors} consecutive empty frames. Stream ended.")
                    break
                time.sleep(0.05)
                continue
            consecutive_errors = 0

            # Compute FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            # Run YOLO inference
            t_infer = time.time()
            results = model(frame, conf=conf_thresh, imgsz=args.imgsz, verbose=False)
            latency_ms = (time.time() - t_infer) * 1000

            # Extract detections
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, "ENCLOSURE")
                    detections.append({
                        "box": xyxy,
                        "conf": conf,
                        "class_name": cls_name
                    })

            # Render visuals
            annotated = draw_detections(frame, detections, conf_threshold=conf_thresh)
            annotated = draw_hud(
                annotated,
                fps=fps,
                latency_ms=latency_ms,
                detected_count=len(detections),
                conf_threshold=conf_thresh,
                model_name="YOLO26 Nano (Roboflow v3)"
            )

            # Update live frame file for workspace view
            cv2.imwrite(live_frame_path, annotated)

            # Update web stream if enabled
            if args.web and streaming_output is not None:
                ret_enc, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret_enc:
                    streaming_output.set_frame(jpeg.tobytes())

            # Display GUI window if supported
            if can_gui:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), ord('Q'), 27]:
                    print("\n[Inference] Quit requested by user.")
                    break
                elif key in [ord('s'), ord('S')]:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    snap_name = f"detection_{timestamp}.jpg"
                    snap_path = os.path.join(args.save_dir, snap_name)
                    cv2.imwrite(snap_path, annotated)
                    print(f"[Snapshot] Saved detection frame to: {snap_path}")
                elif key in [ord('+'), ord('=')]:
                    conf_thresh = min(0.95, conf_thresh + 0.05)
                    print(f"[Threshold] Confidence threshold increased to: {conf_thresh:.2f}")
                elif key in [ord('-'), ord('_')]:
                    conf_thresh = max(0.10, conf_thresh - 0.05)
                    print(f"[Threshold] Confidence threshold decreased to: {conf_thresh:.2f}")
            else:
                # Terminal output for headless/web mode
                if len(detections) > 0:
                    best_conf = max([d['conf'] for d in detections])
                    print(f"[DETECTION] Enclosure detected! Conf: {best_conf*100:.1f}%, Latency: {latency_ms:.1f}ms, FPS: {fps:.1f}")
                elif time.time() - last_status_time > 4.0:
                    print(f"[Scanning] Camera feed active | Latency: {latency_ms:.1f}ms | FPS: {fps:.1f} | Detections: {len(detections)}")
                    last_status_time = time.time()

    except KeyboardInterrupt:
        print("\n[Inference] Interrupted by keyboard.")
    finally:
        camera.release()
        if can_gui:
            cv2.destroyAllWindows()
        if server:
            server.shutdown()
        print("[Inference] Shutdown complete.")


if __name__ == '__main__':
    main()
