"""
제스처 판정 디버그 화면 (담당: Kim Hyunbo)

실행: python -m airboard.gestures.debug
웹캠으로 손 인식·제스처 판정 결과를 확인한다. q 또는 ESC로 종료.
"""

import os
import time

import cv2

from airboard.gestures.gesture import CFG, GestureRecognizer
from airboard.input.camera import open_camera
from airboard.tracking.hand_tracker import DEFAULT_MODEL, HandTracker

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

POSE_COLOR = {"PINCH": (0, 0, 255), "FIST": (255, 0, 255), "POINTER": (0, 200, 255),
              "OPEN": (0, 255, 0), "NONE": (0, 255, 0)}   # PEN=빨강, ERASER=보라, POINTER=주황, PEN STOP=초록


class ToolState:
    """현재 필기 도구 (PEN / HIGHLIGHTER). 박수 두 번(DOUBLE_CLAP)마다 서로 전환.
    렌더링 담당도 이 클래스를 그대로 가져다 쓰면 된다: tool.update(result.events) -> tool.name"""

    def __init__(self):
        self.name = "PEN"
        self.changed_at = -1e9

    def update(self, events, t=None):
        if "DOUBLE_CLAP" in events:
            self.name = "HIGHLIGHTER" if self.name == "PEN" else "PEN"
            self.changed_at = time.monotonic() if t is None else t
            return True
        return False


def display_label(r, tool_name="PEN", flash=None):
    """화면 표시용 이름.
    엄지·검지 붙임 = PEN, 뗌 = PEN STOP, 주먹 = ERASER, 검지만 = POINTER.
    박수 두 번으로 형광펜 모드가 되면 (주먹·포인터가 아닐 때) HIGHLIGHTER.
    스와이프 직후에는 잠깐 NEXT PAGE / PREV PAGE."""
    if flash:
        return flash
    if r.pose == "FIST":
        return "ERASER"
    if r.pose == "POINTER":
        return "POINTER"
    if tool_name == "HIGHLIGHTER" and r.main_hand is not None:
        return "HIGHLIGHTER"
    if r.pose == "PINCH":
        return "PEN"
    if r.main_hand is None:
        return "NO HAND"
    return "PEN STOP"


def draw_debug(frame, r, fps, flash=None, tool=None):
    tool_name = tool.name if tool else "PEN"
    for lm in r.hands:
        pts = lm[:, :2].astype(int)
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, (0, 128, 255), -1)
    if r.pen_point is not None:
        color = (0, 255, 255) if (r.pose == "PINCH" and tool_name == "HIGHLIGHTER") else POSE_COLOR[r.pose]
        cv2.circle(frame, r.pen_point, 10, color, -1 if r.pen_down else 2)

    y = 30
    for text in [f"FPS {fps:.0f}", f"MODE {display_label(r, tool_name, flash)}"] + \
                [f"{k} {v}" for k, v in r.debug.items()]:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 24


def main():
    if not os.path.exists(DEFAULT_MODEL):
        print("[오류] models/hand_landmarker.task 파일이 없습니다.")
        return
    print("모델 불러오는 중...")
    tracker = HandTracker()
    rec = GestureRecognizer()
    print("카메라 여는 중...")
    cap = open_camera(0)
    if cap is None:
        print("[오류] 카메라를 열 수 없습니다. 다른 프로그램(줌, 카메라 앱 등)이 쓰고 있는지,")
        print("       윈도우 설정 > 개인 정보 > 카메라에서 데스크톱 앱 접근이 켜져 있는지 확인하세요.")
        print("       외장 웹캠이면 open_camera(0)의 0을 1로 바꿔보세요.")
        return
    print("실행 중입니다. 'AirBoard gesture debug' 창에서 q 또는 ESC를 누르면 종료됩니다.")
    flash_text, flash_until = None, 0.0   # 스와이프 직후 MODE에 잠깐 표시할 글자
    tool = ToolState()
    prev = time.monotonic()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)      # 거울 모드
        now = time.monotonic()
        r = rec.update(tracker.detect(frame), frame.shape[1], frame.shape[0], now)

        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        for e in r.events:
            print(f"[{now:.2f}] EVENT {e}")      # 터미널에만 기록
            if e == "SWIPE_RIGHT":
                flash_text, flash_until = "NEXT PAGE", now + CFG["mode_flash"]
            elif e == "SWIPE_LEFT":
                flash_text, flash_until = "PREV PAGE", now + CFG["mode_flash"]
        flash = flash_text if now < flash_until else None

        tool.update(r.events)
        draw_debug(frame, r, fps, flash, tool)
        cv2.imshow("AirBoard gesture debug", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
