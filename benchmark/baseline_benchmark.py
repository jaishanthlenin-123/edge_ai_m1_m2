"""
Milestone 1 (M1) Baseline Performance Benchmark for Raspberry Pi 5.
Runs the original trained YOLO26 Nano model (best.pt) on CPU.
Measures:
- Model inference latency (Avg, Median, Min, Max, P95) separate from pipeline overhead
- Pure model inference FPS vs complete camera-to-display pipeline FPS
- Sustained duration tracking (frames, wall-clock time)
- Real-time CPU utilization (total and per-core)
- Process RAM and system memory dynamics
- SoC temperature and hardware thermal throttling flags
Saves per-frame telemetry to CSV and writes comprehensive summary.
"""

import os
import sys
import time
import argparse
import csv
import json
from datetime import datetime
import numpy as np
import psutil
import cv2
from ultralytics import YOLO

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.camera import CameraCapture
from src.utils import get_system_info, draw_detections, draw_hud
from benchmark.system_monitor import get_system_telemetry, get_cpu_temperature, get_throttled_state


def parse_args():
    parser = argparse.ArgumentParser(description="M1 Baseline Performance Benchmark on Raspberry Pi 5")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(project_dir, "models", "best.pt"),
        help="Path to trained PyTorch weights"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera source (0 for USB camera, 'picam' for CSI camera module)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)"
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
        "--duration",
        type=int,
        default=300,
        help="Sustained test duration in seconds (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up frames before data collection (default: 10)"
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=os.path.join(project_dir, "evidence", "M1", "csv", "baseline_m1_measurements.csv"),
        help="Output CSV path for frame measurements"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(project_dir, "evidence", "M1", "benchmark_logs"),
        help="Directory to save benchmark summary logs"
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default=os.path.join(project_dir, "evidence", "M1", "detection_tests"),
        help="Directory to save representative detection frames"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.snapshot_dir, exist_ok=True)

    process = psutil.Process()

    print("=" * 70)
    print("  Milestone 1 (M1): Baseline Performance Benchmark - YOLO26 Nano")
    print("  Platform: Raspberry Pi 5 (ARM Cortex-A76 @ 2.4 GHz) | PyTorch CPU")
    print("=" * 70)

    # Initial hardware & environment state
    sys_info = get_system_info()
    init_telemetry = get_system_telemetry(process)
    ram_before_mb = init_telemetry["system_ram_available_mb"]
    temp_before_c = init_telemetry["temperature_c"]

    print(f"[Hardware] Board: {sys_info['rpi_model']} ({sys_info['machine']})")
    print(f"[Hardware] Initial CPU Temp: {temp_before_c:.1f}°C | Throttling: {init_telemetry['throttled_hex']}")
    print(f"[Hardware] System RAM Available: {ram_before_mb:.1f} MB / {init_telemetry['system_ram_total_mb']:.1f} MB")

    # 1. Load Original Model
    print(f"\n[Model] Loading original weights: {args.model}")
    t_load_start = time.time()
    try:
        model = YOLO(args.model)
        load_time_ms = (time.time() - t_load_start) * 1000
        print(f"[Model] Successfully loaded in {load_time_ms:.1f} ms")
        print(f"[Model] Target classes: {model.names}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # 2. Initialize Camera
    print(f"[Camera] Initializing video source: {args.source}")
    camera = CameraCapture(source=args.source, width=640, height=480, fps=30)
    if not camera.is_opened():
        print(f"[ERROR] Camera source '{args.source}' could not be opened.")
        sys.exit(1)

    # 3. Warm-Up Phase
    print(f"\n[Warm-up] Executing {args.warmup} warm-up frames to stabilize caches...")
    for w in range(args.warmup):
        ret, frame = camera.read()
        if ret and frame is not None:
            _ = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
    print("[Warm-up] Complete. Commencing timed benchmark collection.\n")

    # 4. Sustained Benchmark Execution
    records = []
    saved_snapshots = 0
    start_time = time.time()
    frame_counter = 0

    csv_headers = [
        "frame_id", "timestamp_iso",
        "capture_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "render_ms", "pipeline_total_ms",
        "instant_pipeline_fps", "instant_inference_fps", "detections_count",
        "cpu_total_pct", "cpu_core0_pct", "cpu_core1_pct", "cpu_core2_pct", "cpu_core3_pct",
        "proc_ram_rss_mb", "sys_ram_avail_mb", "temp_c", "throttled_hex"
    ]

    with open(args.csv_out, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)

        print(f"[Benchmark] Running sustained evaluation for {args.duration} seconds...")
        print("  Tracking: Inference Latency, Pipeline FPS, Core Loads, RAM, and Temperature")

        last_progress_print = time.time()

        try:
            while (time.time() - start_time) < args.duration:
                t_frame_start = time.time()
                ts_iso = datetime.now().isoformat()

                # Step A: Camera frame capture
                t_cap_0 = time.time()
                ret, frame = camera.read()
                capture_ms = (time.time() - t_cap_0) * 1000

                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame_counter += 1

                # Step B: Model inference (isolated measurement)
                t_infer_0 = time.time()
                results = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
                t_infer_1 = time.time()
                total_infer_wall_ms = (t_infer_1 - t_infer_0) * 1000

                # Extract Ultralytics detailed latency breakdown
                speed_dict = results[0].speed if results and len(results) > 0 else {}
                preprocess_ms = speed_dict.get("preprocess", 0.0)
                inference_ms = speed_dict.get("inference", total_infer_wall_ms)
                postprocess_ms = speed_dict.get("postprocess", 0.0)

                # Detections
                detections = []
                if results and len(results) > 0:
                    for box in results[0].boxes:
                        detections.append({
                            "box": box.xyxy[0].tolist(),
                            "conf": float(box.conf[0]),
                            "class_name": model.names.get(int(box.cls[0]), "ENCLOSURE")
                        })

                # Step C: Rendering and HUD overhead
                t_rend_0 = time.time()
                annotated = draw_detections(frame, detections, conf_threshold=args.conf)
                # Compute instantaneous FPS
                t_frame_end = time.time()
                pipeline_total_ms = (t_frame_end - t_frame_start) * 1000
                instant_pipe_fps = 1000.0 / max(pipeline_total_ms, 1e-3)
                instant_infer_fps = 1000.0 / max(inference_ms, 1e-3)

                annotated = draw_hud(
                    annotated,
                    fps=instant_pipe_fps,
                    latency_ms=inference_ms,
                    detected_count=len(detections),
                    conf_threshold=args.conf,
                    model_name="YOLO26 Nano (Roboflow v3)"
                )
                render_ms = (time.time() - t_rend_0) * 1000

                # Update live snapshot image
                live_path = os.path.join(project_dir, "evidence", "M1", "screenshots", "live_frame.jpg")
                cv2.imwrite(live_path, annotated)

                # Save up to 5 representative detection snapshots
                if len(detections) > 0 and saved_snapshots < 5:
                    saved_snapshots += 1
                    snap_name = f"baseline_det_{saved_snapshots}_conf_{int(detections[0]['conf']*100)}.jpg"
                    snap_path = os.path.join(args.snapshot_dir, snap_name)
                    cv2.imwrite(snap_path, annotated)

                # Step D: System telemetry sampling
                telem = get_system_telemetry(process)
                cores = telem["cpu_per_core_percent"]
                c0 = cores[0] if len(cores) > 0 else 0.0
                c1 = cores[1] if len(cores) > 1 else 0.0
                c2 = cores[2] if len(cores) > 2 else 0.0
                c3 = cores[3] if len(cores) > 3 else 0.0

                row = [
                    frame_counter, ts_iso,
                    round(capture_ms, 2), round(preprocess_ms, 2), round(inference_ms, 2),
                    round(postprocess_ms, 2), round(render_ms, 2), round(pipeline_total_ms, 2),
                    round(instant_pipe_fps, 2), round(instant_infer_fps, 2), len(detections),
                    round(telem["cpu_total_percent"], 1), round(c0, 1), round(c1, 1), round(c2, 1), round(c3, 1),
                    round(telem["process_rss_mb"], 2), round(telem["system_ram_available_mb"], 2),
                    round(telem["temperature_c"], 2) if telem["temperature_c"] else 0.0,
                    telem["throttled_hex"]
                ]
                writer.writerow(row)
                records.append({
                    "infer_ms": inference_ms,
                    "pipe_ms": pipeline_total_ms,
                    "cpu_pct": telem["cpu_total_percent"],
                    "cores": cores,
                    "ram_proc": telem["process_rss_mb"],
                    "ram_avail": telem["system_ram_available_mb"],
                    "temp": telem["temperature_c"],
                    "throttled": telem["throttled_hex"],
                    "detections": len(detections)
                })

                # Terminal progress heartbeat every 10 seconds
                if (time.time() - last_progress_print) >= 10.0:
                    elapsed = time.time() - start_time
                    recent_inf = np.mean([r["infer_ms"] for r in records[-15:]])
                    recent_pipe_fps = 1000.0 / np.mean([r["pipe_ms"] for r in records[-15:]])
                    print(f"  [Progress] {elapsed:.0f}/{args.duration}s | Frames: {frame_counter} | "
                          f"Infer Latency: {recent_inf:.1f}ms | Pipeline: {recent_pipe_fps:.2f} FPS | "
                          f"Temp: {telem['temperature_c']:.1f}°C | CPU: {telem['cpu_total_percent']:.1f}%")
                    last_progress_print = time.time()

        except KeyboardInterrupt:
            print("\n[Benchmark] Early interruption requested by user.")
        finally:
            camera.release()

    total_duration_s = time.time() - start_time
    final_telemetry = get_system_telemetry(process)
    ram_after_mb = final_telemetry["system_ram_available_mb"]
    temp_after_c = final_telemetry["temperature_c"]

    # 5. Statistical Calculations
    if not records:
        print("[ERROR] No benchmark frames collected.")
        sys.exit(1)

    infer_arr = np.array([r["infer_ms"] for r in records])
    pipe_arr = np.array([r["pipe_ms"] for r in records])
    cpu_arr = np.array([r["cpu_pct"] for r in records])
    proc_ram_arr = np.array([r["ram_proc"] for r in records])
    temp_arr = np.array([r["temp"] for r in records if r["temp"] is not None])

    avg_infer_ms = float(np.mean(infer_arr))
    med_infer_ms = float(np.median(infer_arr))
    min_infer_ms = float(np.min(infer_arr))
    max_infer_ms = float(np.max(infer_arr))
    p95_infer_ms = float(np.percentile(infer_arr, 95))

    infer_fps = 1000.0 / avg_infer_ms
    pipeline_fps = float(len(records) / total_duration_s)

    avg_cpu = float(np.mean(cpu_arr))
    peak_cpu = float(np.max(cpu_arr))

    # Per-core averages
    all_cores = [r["cores"] for r in records if len(r["cores"]) >= 4]
    core_avgs = [float(np.mean([c[i] for c in all_cores])) for i in range(4)] if all_cores else [0, 0, 0, 0]

    peak_proc_ram = float(np.max(proc_ram_arr))
    final_proc_ram = float(proc_ram_arr[-1])

    min_temp = float(np.min(temp_arr)) if len(temp_arr) else temp_before_c
    max_temp = float(np.max(temp_arr)) if len(temp_arr) else temp_after_c
    delta_temp = max_temp - min_temp

    throttling_events = sum(1 for r in records if r["throttled"] != "0x0")

    summary = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "model_name": "YOLO26 Nano (Roboflow Version 3)",
        "model_weights": args.model,
        "model_size_bytes": os.path.getsize(args.model),
        "input_resolution": f"{args.imgsz}x{args.imgsz}",
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "runtime_backend": "PyTorch ARM64 CPU (FP32)",
        "total_frames_processed": len(records),
        "total_duration_seconds": round(total_duration_s, 2),
        "inference_latency": {
            "average_ms": round(avg_infer_ms, 2),
            "median_ms": round(med_infer_ms, 2),
            "min_ms": round(min_infer_ms, 2),
            "max_ms": round(max_infer_ms, 2),
            "p95_ms": round(p95_infer_ms, 2)
        },
        "fps_metrics": {
            "pure_inference_fps": round(infer_fps, 2),
            "complete_pipeline_fps": round(pipeline_fps, 2)
        },
        "cpu_utilization": {
            "average_percent": round(avg_cpu, 2),
            "peak_percent": round(peak_cpu, 2),
            "per_core_average_percent": [round(c, 2) for c in core_avgs]
        },
        "ram_utilization": {
            "available_before_mb": round(ram_before_mb, 2),
            "process_used_mb": round(final_proc_ram, 2),
            "process_peak_mb": round(peak_proc_ram, 2),
            "available_after_mb": round(ram_after_mb, 2)
        },
        "temperature_celsius": {
            "start": round(temp_before_c, 2) if temp_before_c else None,
            "min": round(min_temp, 2),
            "peak": round(max_temp, 2),
            "end": round(temp_after_c, 2) if temp_after_c else None,
            "delta": round(delta_temp, 2)
        },
        "thermal_throttling": {
            "occurred": throttling_events > 0,
            "throttling_events_count": throttling_events,
            "final_throttled_hex": final_telemetry["throttled_hex"],
            "final_throttled_flags": final_telemetry["throttled_flags"]
        }
    }

    # Save summary JSON and TXT logs
    summary_json_path = os.path.join(args.log_dir, "m1_baseline_summary.json")
    with open(summary_json_path, "w") as jf:
        json.dump(summary, jf, indent=2)

    summary_txt_path = os.path.join(args.log_dir, "m1_baseline_summary.txt")
    with open(summary_txt_path, "w") as tf:
        tf.write("=" * 65 + "\n")
        tf.write("  M1 BASELINE PERFORMANCE BENCHMARK - RASPBERRY PI 5\n")
        tf.write("=" * 65 + "\n\n")
        tf.write(f"Model:                {summary['model_name']}\n")
        tf.write(f"Weights File:         {summary['model_weights']} ({summary['model_size_bytes']} bytes)\n")
        tf.write(f"Runtime Backend:      {summary['runtime_backend']}\n")
        tf.write(f"Input Resolution:     {summary['input_resolution']}\n")
        tf.write(f"Frames Evaluated:     {summary['total_frames_processed']} frames\n")
        tf.write(f"Test Duration:        {summary['total_duration_seconds']} seconds\n\n")
        tf.write("--- Inference Latency (Model-Only) ---\n")
        tf.write(f"Average Latency:      {summary['inference_latency']['average_ms']} ms\n")
        tf.write(f"Median Latency:       {summary['inference_latency']['median_ms']} ms\n")
        tf.write(f"Minimum Latency:      {summary['inference_latency']['min_ms']} ms\n")
        tf.write(f"Maximum Latency:      {summary['inference_latency']['max_ms']} ms\n")
        tf.write(f"95th Percentile:      {summary['inference_latency']['p95_ms']} ms\n\n")
        tf.write("--- Frame Rate (FPS) ---\n")
        tf.write(f"Model Inference FPS:  {summary['fps_metrics']['pure_inference_fps']} FPS\n")
        tf.write(f"Complete Pipeline:    {summary['fps_metrics']['complete_pipeline_fps']} FPS\n\n")
        tf.write("--- CPU Utilization ---\n")
        tf.write(f"Average CPU Load:     {summary['cpu_utilization']['average_percent']}%\n")
        tf.write(f"Peak CPU Load:        {summary['cpu_utilization']['peak_percent']}%\n")
        tf.write(f"Per-Core Loads:       {summary['cpu_utilization']['per_core_average_percent']}%\n\n")
        tf.write("--- RAM Utilization ---\n")
        tf.write(f"Available Before:     {summary['ram_utilization']['available_before_mb']} MB\n")
        tf.write(f"Process Memory:       {summary['ram_utilization']['process_used_mb']} MB\n")
        tf.write(f"Process Peak:         {summary['ram_utilization']['process_peak_mb']} MB\n")
        tf.write(f"Available After:      {summary['ram_utilization']['available_after_mb']} MB\n\n")
        tf.write("--- Thermal & Throttling ---\n")
        tf.write(f"Start Temp:           {summary['temperature_celsius']['start']}°C\n")
        tf.write(f"Peak Temp:            {summary['temperature_celsius']['peak']}°C\n")
        tf.write(f"Temp Delta:           +{summary['temperature_celsius']['delta']}°C\n")
        tf.write(f"Thermal Throttling:   {'YES' if summary['thermal_throttling']['occurred'] else 'NO'} ({summary['thermal_throttling']['final_throttled_flags']})\n")
        tf.write("=" * 65 + "\n")

    # Console Report
    with open(summary_txt_path, "r") as f:
        print("\n" + f.read())

    print(f"[Evidence Saved] CSV raw logs:       {args.csv_out}")
    print(f"[Evidence Saved] JSON summary:       {summary_json_path}")
    print(f"[Evidence Saved] Text report:        {summary_txt_path}")
    print(f"[Evidence Saved] Live screenshot:    {live_path}")


if __name__ == "__main__":
    main()
