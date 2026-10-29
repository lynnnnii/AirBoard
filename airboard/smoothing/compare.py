"""
떨림 보정 비교 화면 (담당: Lee Subin)

실행: python -m airboard.smoothing.compare
엄지·검지를 붙이면 쓰기, 떼면 멈춤. 같은 손 움직임을 세 가지 보정으로 동시에 그린다.
  빨강 = 보정 없음, 노랑 = EMA, 초록 = One Euro (초기값은 config/default.json)
손을 가만히 들고 있으면 왼쪽 위 jitter 값(최근 30프레임 흔들림, px)으로 보정 효과를 비교할 수 있다.
키: 1/2/3 선 보이기·숨기기, t 꼬리 제거 켜기·끄기, c 지우기, q/ESC 종료 (창을 닫아도 종료)
"""

from __future__ import annotations

import collections
import os
import time

import cv2
import numpy as np

from airboard.gestures.gesture import GestureRecognizer
from airboard.input.camera import open_camera
from airboard.smoothing.stroke import Point, SmoothResult, StrokeSmoother, load_config
from airboard.tracking.hand_tracker import DEFAULT_MODEL, HandTracker

WIN = "AirBoard smoothing compare"
JITTER_WINDOW = 30
METHODS = ("none", "ema", "one_euro")
LABELS = {"none": "raw", "ema": "EMA", "one_euro": "OneEuro"}
COLORS = {"none": (0, 0, 255), "ema": (0, 220, 255), "one_euro": (0, 200, 0)}


def jitter(points: collections.deque) -> float:
    """최근 점들이 평균 위치에서 얼마나 흩어져 있는지 (RMS, px)."""
    if len(points) < 2:
        return 0.0
    arr = np.array(points)
    return float(np.sqrt(((arr - arr.mean(axis=0)) ** 2).sum(axis=1).mean()))


def add_trackbars(cfg: dict) -> None:
    noop = lambda _: None
    cv2.createTrackbar("EMA alpha x100", WIN, int(cfg["ema_alpha"] * 100), 100, noop)
    cv2.createTrackbar("1e min_cutoff x10", WIN, int(cfg["one_euro_min_cutoff"] * 10), 100, noop)
    cv2.createTrackbar("1e beta x1000", WIN, int(cfg["one_euro_beta"] * 1000), 100, noop)


def read_trackbars(smoothers: dict[str, StrokeSmoother]) -> None:
    smoothers["ema"].filter.alpha = max(cv2.getTrackbarPos("EMA alpha x100", WIN), 1) / 100
    one_euro = smoothers["one_euro"].filter
    one_euro.min_cutoff = max(cv2.getTrackbarPos("1e min_cutoff x10", WIN), 1) / 10
    one_euro.beta = cv2.getTrackbarPos("1e beta x1000", WIN) / 1000


def draw(frame: np.ndarray, results: dict[str, SmoothResult], done: dict[str, list[list[Point]]],
         recent: dict[str, collections.deque], visible: dict[str, bool],
         smoothers: dict[str, StrokeSmoother], pen_down: bool, trim_on: bool) -> None:
    for m in METHODS:
        if not visible[m]:
            continue
        s = results[m]
        for stroke in done[m] + ([s.stroke] if s.stroke else []):
            if len(stroke) > 1:
                cv2.polylines(frame, [np.array(stroke, dtype=np.int32)], False, COLORS[m], 2, cv2.LINE_AA)
        if s.point is not None:
            cv2.circle(frame, tuple(int(v) for v in s.point), 6, COLORS[m], -1 if pen_down else 2)

    oe = smoothers["one_euro"].filter
    params = {"none": "", "ema": f"a={smoothers['ema'].filter.alpha:.2f}",
              "one_euro": f"mc={oe.min_cutoff:.1f} b={oe.beta:.3f}"}
    lines = [(f"{'WRITING' if pen_down else 'pen up'}   tail trim [t]: {'ON' if trim_on else 'OFF'}", (255, 255, 255))]
    lines += [(f"{LABELS[m]:8} jitter {jitter(recent[m]):5.1f}px  [{i + 1}]  {params[m]}", COLORS[m])
              for i, m in enumerate(METHODS)]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (10, 25 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, 25 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def main() -> None:
    if not os.path.exists(DEFAULT_MODEL):
        print("[오류] models/hand_landmarker.task 파일이 없습니다.")
        return
    cfg = load_config()
    print("모델 불러오는 중...")
    tracker = HandTracker()
    rec = GestureRecognizer()
    print("카메라 여는 중...")
    cap = open_camera(0)
    if cap is None:
        print("[오류] 카메라를 열 수 없습니다. 다른 프로그램이 쓰고 있는지 확인하세요.")
        return
    print(f"실행 중입니다. '{WIN}' 창에서 q 또는 ESC를 누르면 종료됩니다.")

    smoothers = {m: StrokeSmoother.from_config(cfg, method=m) for m in METHODS}
    done: dict[str, list[list[Point]]] = {m: [] for m in METHODS}
    recent = {m: collections.deque(maxlen=JITTER_WINDOW) for m in METHODS}
    visible = {m: True for m in METHODS}
    trim_on = True

    cv2.namedWindow(WIN)
    add_trackbars(cfg)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)      # 거울 모드
        t = time.monotonic()
        read_trackbars(smoothers)
        r = rec.update(tracker.detect(frame), frame.shape[1], frame.shape[0], t)

        results = {}
        for m in METHODS:
            s = smoothers[m].update(r.pen_point, r.pen_down, t)
            results[m] = s
            if s.finished:
                done[m].append(s.finished)
            if s.point is None:
                recent[m].clear()
            else:
                recent[m].append(s.point)

        draw(frame, results, done, recent, visible, smoothers, r.pen_down, trim_on)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        # 창의 X 버튼으로 닫아도 종료 (안 그러면 창만 사라지고 카메라를 계속 잡고 있음)
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("c"):
            for strokes in done.values():
                strokes.clear()
        if key == ord("t"):
            trim_on = not trim_on
            for sm in smoothers.values():
                sm.start_skip = cfg["start_skip"] if trim_on else 0
                sm.end_trim = cfg["end_trim"] if trim_on else 0
        for k, m in zip("123", METHODS):
            if key == ord(k):
                visible[m] = not visible[m]

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
