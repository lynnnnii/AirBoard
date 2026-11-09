"""MediaPipe Hands wrapper (contributor: Kim Hyunbo).

Model file: models/hand_landmarker.task
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""
from pathlib import Path
import os
import time

import cv2
import numpy as np

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "models", "hand_landmarker.task")


class HandTracker:
    """Extract hand landmarks as a list of pixel-coordinate (21, 3) arrays."""

    def __init__(self, model_path=DEFAULT_MODEL, detection_confidence=0.5):
        if not 0<detection_confidence<=1:
            raise ValueError("detection_confidence must be in (0, 1]")
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,                       # Track both hands for clap recognition
            min_hand_detection_confidence=detection_confidence,  # A lower threshold helps reacquire hands quickly after a clap
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.t0 = time.monotonic()
        self.last_ts = -1

    def detect(self, frame_bgr):
        """frame_bgr: BGR frame already mirrored horizontally."""
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self.t0) * 1000)
        ts = max(ts, self.last_ts + 1)          # VIDEO mode requires strictly increasing timestamps
        self.last_ts = ts
        res = self.landmarker.detect_for_video(image, ts)
        hands = []
        for hand in res.hand_landmarks:
            # The z coordinate uses the x scale, so multiply by frame width to express it in pixels
            hands.append(np.array([[p.x * w, p.y * h, p.z * w] for p in hand], dtype=np.float32))
        return hands

    def close(self):
        self.landmarker.close()
