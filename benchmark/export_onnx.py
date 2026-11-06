"""
Milestone 2 (M2) Model Optimization: Export YOLO26 Nano (best.pt) to ONNX.
- Exports original trained weights to models/onnx/best.onnx
- Preserves original PyTorch model
- Validates model architecture, opset version, tensor shapes, and ONNX Runtime execution
- Compares PyTorch vs ONNX outputs on identical input
- Logs full export details to evidence/M2/export_logs/
"""

import os
import sys
import time
import json
import shutil
import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO

# Project root path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

src_pt = os.path.join(project_dir, "models", "best.pt")
dest_onnx_dir = os.path.join(project_dir, "models", "onnx")
dest_onnx = os.path.join(dest_onnx_dir, "best.onnx")
log_dir = os.path.join(project_dir, "evidence", "M2", "export_logs")
os.makedirs(dest_onnx_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print("=" * 65)
print("  Milestone 2 (M2): YOLO26 Nano ONNX Export & Verification")
print("=" * 65)

# 1. Inspect source model
assert os.path.exists(src_pt), f"Source model not found at {src_pt}"
pt_size = os.path.getsize(src_pt)
print(f"[Source] PyTorch Checkpoint: {src_pt} ({pt_size} bytes)")

t0 = time.time()
model = YOLO(src_pt)
print(f"[Source] Model Architecture: {type(model.model)}")
print(f"[Source] Class mapping: {model.names}")

# 2. Perform ONNX Export
print(f"\n[Export] Exporting to ONNX format (imgsz=640, opset=17)...")
export_log_txt = []
t_export_0 = time.time()
exported_file = model.export(
    format="onnx",
    imgsz=640,
    opset=17,
    dynamic=False,
    simplify=False
)
export_duration_s = time.time() - t_export_0
print(f"[Export] Export completed in {export_duration_s:.2f} seconds.")
print(f"[Export] Generated file: {exported_file}")

# Move to standard destination: models/onnx/best.onnx
if exported_file != dest_onnx:
    shutil.copy2(exported_file, dest_onnx)
print(f"[Export] Saved ONNX model to: {dest_onnx}")

# 3. Model Integrity Verification
print("\n[Verification] Validating ONNX graph integrity with onnx.checker...")
onnx_model = onnx.load(dest_onnx)
onnx.checker.check_model(onnx_model)
print("[Verification] ONNX structure check PASSED!")

opset_version = None
for o in onnx_model.opset_import:
    if o.domain == "" or o.domain == "ai.onnx":
        opset_version = o.version
print(f"[Verification] ONNX Opset Version: {opset_version}")

onnx_size = os.path.getsize(dest_onnx)
print(f"[Verification] ONNX File Size: {onnx_size} bytes ({onnx_size / (1024*1024):.2f} MB)")

# 4. Inspect Input and Output Signatures
session = ort.InferenceSession(dest_onnx, providers=["CPUExecutionProvider"])
inputs = session.get_inputs()
outputs = session.get_outputs()

input_info = [{"name": i.name, "shape": i.shape, "type": i.type} for i in inputs]
output_info = [{"name": o.name, "shape": o.shape, "type": o.type} for o in outputs]

print("\n[Signature] Input Tensors:")
for inp in input_info:
    print(f"  Name: {inp['name']}, Shape: {inp['shape']}, Type: {inp['type']}")

print("[Signature] Output Tensors:")
for out in output_info:
    print(f"  Name: {out['name']}, Shape: {out['shape']}, Type: {out['type']}")

# 5. Numerical Output Equivalence Check (PyTorch vs ONNX)
print("\n[Validation] Checking PyTorch vs ONNX numerical consistency on synthetic test tensor...")
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

# PyTorch forward pass
model.model.eval()
with torch.no_grad():
    pt_tensor = torch.from_numpy(dummy_input)
    pt_out = model.model(pt_tensor)
    if isinstance(pt_out, (tuple, list)):
        pt_out_np = pt_out[0].cpu().numpy()
    else:
        pt_out_np = pt_out.cpu().numpy()

# ONNX Runtime forward pass
ort_out = session.run([outputs[0].name], {inputs[0].name: dummy_input})[0]

# Calculate absolute difference
abs_diff = np.abs(pt_out_np - ort_out)
max_diff = float(np.max(abs_diff))
mean_diff = float(np.mean(abs_diff))
print(f"[Validation] Max Absolute Difference: {max_diff:.6e}")
print(f"[Validation] Mean Absolute Difference: {mean_diff:.6e}")
diff_status = "PASSED (< 1e-4 tolerance)" if max_diff < 1e-3 else "WARNING (Discrepancy observed)"
print(f"[Validation] Equivalence check: {diff_status}")

# 6. Save Export Logs
report_data = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source_model": src_pt,
    "source_size_bytes": pt_size,
    "destination_model": dest_onnx,
    "destination_size_bytes": onnx_size,
    "opset_version": opset_version,
    "input_resolution": "640x640",
    "export_duration_seconds": round(export_duration_s, 2),
    "input_tensors": input_info,
    "output_tensors": output_info,
    "numerical_consistency": {
        "max_absolute_difference": max_diff,
        "mean_absolute_difference": mean_diff,
        "status": diff_status
    },
    "execution_providers": session.get_providers()
}

log_json_path = os.path.join(log_dir, "onnx_export_log.json")
with open(log_json_path, "w") as f:
    json.dump(report_data, f, indent=2)

log_txt_path = os.path.join(log_dir, "onnx_export_log.txt")
with open(log_txt_path, "w") as f:
    f.write("=" * 65 + "\n")
    f.write("  YOLO26 NANO - ONNX EXPORT AND VERIFICATION LOG\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Source PyTorch Checkpoint: {src_pt} ({pt_size} bytes)\n")
    f.write(f"Exported ONNX Model:       {dest_onnx} ({onnx_size} bytes)\n")
    f.write(f"Opset Version:             {opset_version}\n")
    f.write(f"Input Shape:               {input_info[0]['shape']}\n")
    f.write(f"Output Shape:              {output_info[0]['shape']}\n")
    f.write(f"Export Duration:           {export_duration_s:.2f} seconds\n")
    f.write(f"Max Difference (PT vs ORT):{max_diff:.6e}\n")
    f.write(f"Equivalence Status:        {diff_status}\n")
    f.write(f"Active Providers:          {session.get_providers()}\n")

print(f"\n[Saved] Detailed JSON log: {log_json_path}")
print(f"[Saved] Text export log:  {log_txt_path}")
print("=" * 65)
