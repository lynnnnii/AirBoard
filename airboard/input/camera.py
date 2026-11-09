"""Webcam input."""

import sys

import cv2


def open_camera(index=0):
    """On Windows, prefer DirectShow because the default MSMF backend may hang or take a long time to open the camera."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None
