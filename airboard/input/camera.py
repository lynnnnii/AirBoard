"""웹캠 입력."""

import sys

import cv2


def open_camera(index=0):
    """윈도우에서는 기본 방식(MSMF)이 카메라를 여는 데 수십 초 걸리거나 멈추는 경우가 있어 DirectShow를 먼저 시도한다."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None
