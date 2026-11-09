"""
Gesture recognition debug view (contributor: Kim Hyunbo)

Run: python -m airboard.gestures.debug
Inspect webcam hand tracking and gesture recognition; press q or Esc to quit.
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
              "OPEN": (0, 255, 0), "NONE": (0, 255, 0)}   # PEN=red, ERASER=purple, POINTER=orange, PEN STOP=green


class ToolState:
    """Current drawing tool (PEN / HIGHLIGHTER); toggle on DOUBLE_CLAP.
    Rendering modules can also use this class: tool.update(result.events) -> tool.name"""

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
    """Display name for the current gesture state.
    Pinch thumb and index = PEN; release = PEN STOP; fist = ERASER; index only = POINTER.
    After toggling by double clap, show HIGHLIGHTER unless fist/pointer is active.
    After swipes, briefly display NEXT PAGE / PREV PAGE."""
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
        print("[Error] Missing models/hand_landmarker.task.")
        return
    print("Loading model...")
    tracker = HandTracker()
    rec = GestureRecognizer()
    print("Opening camera...")
    cap = open_camera(0)
    if cap is None:
        print("[Error] Could not open the camera. Check whether Zoom or another camera app is using it,")
        print("       and enable desktop-app camera permissions in Windows Settings > Privacy > Camera.")
        print("       For an external webcam, try changing open_camera(0) to open_camera(1).")
        return
    print("Running. Press q or Esc in the 'AirBoard gesture debug' window to quit.")
    flash_text, flash_until = None, 0.0   # Text briefly displayed in MODE after a swipe
    tool = ToolState()
    prev = time.monotonic()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)      # Mirrored camera view
        now = time.monotonic()
        r = rec.update(tracker.detect(frame), frame.shape[1], frame.shape[0], now)

        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        for e in r.events:
            print(f"[{now:.2f}] EVENT {e}")      # Log only to the terminal
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
