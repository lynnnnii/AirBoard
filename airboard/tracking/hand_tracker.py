"""MediaPipe Hands 래퍼 (담당: Kim Hyunbo).

모델 파일: models/hand_landmarker.task
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
    """프레임에서 손 랜드마크를 픽셀 좌표 (21,3) 배열 리스트로 뽑는다."""

    def __init__(self, model_path=DEFAULT_MODEL):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,                       # 박수 인식을 위해 양손 추적
            min_hand_detection_confidence=0.5,  # 낮을수록 박수 후 손을 빠르게 다시 찾음
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.t0 = time.monotonic()
        self.last_ts = -1

    def detect(self, frame_bgr):
        """frame_bgr: 좌우 반전까지 끝난 BGR 프레임."""
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self.t0) * 1000)
        ts = max(ts, self.last_ts + 1)          # VIDEO 모드는 타임스탬프가 반드시 증가해야 함
        self.last_ts = ts
        res = self.landmarker.detect_for_video(image, ts)
        hands = []
        for hand in res.hand_landmarks:
            # z는 x와 같은 스케일이므로 w를 곱해 픽셀 단위로 맞춘다
            hands.append(np.array([[p.x * w, p.y * h, p.z * w] for p in hand], dtype=np.float32))
        return hands

    def close(self):
        self.landmarker.close()
