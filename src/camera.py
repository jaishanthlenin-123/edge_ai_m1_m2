"""
Camera capture interface with dual backend support:
1. Raspberry Pi 5 native Camera Module (via Picamera2 / libcamera)
2. Standard USB / laptop webcam (via OpenCV cv2.VideoCapture)
"""

import sys
import time
import cv2
import numpy as np

# Try importing Picamera2 for Raspberry Pi 5
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class CameraCapture:
    def __init__(self, source=0, width=640, height=480, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.is_picam = False
        self.cap = None
        self.picam2 = None

        # Check if source specifies Raspberry Pi camera module
        if str(source).lower() in ['picam', 'picamera', 'csi']:
            if PICAMERA2_AVAILABLE:
                self._init_picam2()
            else:
                print("[WARNING] Picamera2 not installed or not running on Raspberry Pi. Falling back to OpenCV index 0.")
                self._init_opencv(0)
        else:
            # Numeric index or video path
            try:
                cam_idx = int(source)
                self._init_opencv(cam_idx)
            except ValueError:
                # Video file or stream URL
                self._init_opencv(source)

    def _init_picam2(self):
        print(f"[CameraCapture] Initializing Raspberry Pi Camera Module via Picamera2 ({self.width}x{self.height})...")
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.is_picam = True
            time.sleep(1.0) # Camera warm-up
            print("[CameraCapture] Picamera2 initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to start Picamera2: {e}")
            print("[CameraCapture] Falling back to OpenCV VideoCapture...")
            self.is_picam = False
            self._init_opencv(0)

    def _init_opencv(self, cam_idx):
        print(f"[CameraCapture] Opening OpenCV VideoCapture source: {cam_idx}...")
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            # Try camera index 0 if specific index failed
            if isinstance(cam_idx, int) and cam_idx != 0:
                print(f"[CameraCapture] Index {cam_idx} failed, trying camera index 0...")
                self.cap = cv2.VideoCapture(0)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            # Reduce buffering delay
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[CameraCapture] OpenCV camera opened: {actual_w}x{actual_h}")
        else:
            print(f"[ERROR] Could not open video source: {cam_idx}")

    def is_opened(self):
        """Returns True if the camera source is actively initialized and opened."""
        if self.is_picam and self.picam2 is not None:
            return True
        elif self.cap is not None and self.cap.isOpened():
            return True
        return False

    def read(self):
        """
        Reads a frame from the camera.
        Returns:
            (ret, frame): ret (bool), frame (numpy array BGR format)
        """
        if self.is_picam and self.picam2 is not None:
            try:
                frame_rgb = self.picam2.capture_array()
                # Picamera2 outputs RGB; convert to BGR for OpenCV / Ultralytics
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                return True, frame_bgr
            except Exception as e:
                print(f"[CameraCapture] Picamera2 read error: {e}")
                return False, None
        elif self.cap is not None and self.cap.isOpened():
            return self.cap.read()
        else:
            return False, None

    def release(self):
        """Releases camera hardware resources."""
        if self.is_picam and self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
                print("[CameraCapture] Picamera2 stopped.")
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
            print("[CameraCapture] OpenCV VideoCapture released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

