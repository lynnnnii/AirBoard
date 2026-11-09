from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import json
import time
import cv2
import numpy as np
from airboard.render import AirBoardRenderer, RenderConfig
from airboard.render.session import RenderSession


def to_slide(point: tuple[float, float] | None, frame_size: tuple[int, int],
             slide_size: tuple[int, int]) -> tuple[float, float] | None:
    """Scale-based mapping helper for testing.
    
    This is not true homography calibration; replace it with the calibration module when integrating.
    """
    if point is None:
        return None
    fw, fh = frame_size
    sw, sh = slide_size
    return float(point[0]) / fw * (sw - 1), float(point[1]) / fh * (sh - 1)


def main() -> None:
    """Test handwriting using mouse or video input with the team's existing smoothing and tracking modules."""
    
    # 1. Configure command-line argument parsing
    parser = argparse.ArgumentParser(description='AirBoard render test')
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--config', default='config/default.json')
    parser.add_argument('--mapping', default='config/gestures.json')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--webcam', action='store_true')
    source.add_argument('--video')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--no-smoothing', action='store_true')
    args = parser.parse_args()
    
    # 2. Load the team's existing smoother without reimplementing or modifying it
    from airboard.smoothing.stroke import StrokeSmoother, load_config, make_filter
    cfg = load_config(args.config)
    if args.no_smoothing:
        cfg['method'] = 'none'  # Disable only filtering, preserving start_skip/end_trim settings
    
    # 3. Initialize renderer and session
    board = AirBoardRenderer(args.pdf, RenderConfig.from_json(args.config))
    session = RenderSession(board, StrokeSmoother.from_config(cfg), make_filter(cfg['method'], cfg))
    
    cap = tracker = recognizer = None
    win = 'AirBoard PDF - render test'
    camera_win = 'AirBoard camera - test mapping'
    toolbar = 64
    cw, ch = board.config.max_width, board.config.max_height + toolbar
    mouse = {'point': None, 'down': False}
    mapping = {}
    
    # 4. Load gesture mapping JSON
    if Path(args.mapping).exists():
        mapping = json.loads(Path(args.mapping).read_text(encoding='utf-8'))
        if not isinstance(mapping, dict):
            board.close()
            raise ValueError('Gesture mapping must be a JSON object. Webcam demo uses the raw event: action format.')
    
    notice = ''
    notice_until = 0.0
    input_failed = False
    mode = 'VIDEO' if args.video else 'WEBCAM' if args.webcam else 'MOUSE'

    def slide_rect() -> tuple[int, int, int, int]:
        """Compute slide-image bounds (x, y, w, h) within the current display."""
        w, h = board.slide_size
        return (cw - w) // 2, toolbar + (board.config.max_height - h) // 2, w, h

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        """Update mouse position and button state; the main loop handles drawing."""
        ox, oy, w, h = slide_rect()
        mouse['point'] = (float(x - ox), float(y - oy)) if ox <= x < ox + w and oy <= y < oy + h else None
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse['down'] = True
        elif event == cv2.EVENT_LBUTTONUP or (event == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_LBUTTON):
            mouse['down'] = False

    try:
        # 5. Configure webcam or video input
        if args.webcam or args.video:
            from airboard.input.camera import open_camera
            from airboard.tracking.hand_tracker import HandTracker, DEFAULT_MODEL
            from airboard.gestures.gesture import GestureRecognizer
            
            if not Path(DEFAULT_MODEL).is_file():
                raise FileNotFoundError(f'Hand model not found: {DEFAULT_MODEL}')
                
            tracker = HandTracker()
            recognizer = GestureRecognizer()
            cap = cv2.VideoCapture(args.video) if args.video else open_camera(args.camera)
            
            if cap is None or not cap.isOpened():
                raise RuntimeError('Could not open video input. Check camera access or video path.')
            cv2.namedWindow(camera_win, cv2.WINDOW_AUTOSIZE)
            
        # 6. Configure the OpenCV window and mouse callback
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, mouse_callback)
        previous = time.monotonic()
        fps = 0.0
        
        # 7. Main rendering and event-handling loop
        while True:
            now = time.monotonic()
            fps = .9 * fps + .1 / max(now - previous, 1e-6)
            previous = now
            events = []
            temporary = None
            
            # Handle coordinates and state for mouse versus webcam/video inputs
            if cap is None:
                # Mouse mode
                point = mouse['point']
                down = bool(mouse['down'])
                erasing = session.selected == 'eraser' and down
            else:
                # Webcam/video mode
                ok, frame = cap.read() if not input_failed else (False, None)
                if not ok:
                    if not input_failed:
                        session.flush(now)
                        print('Video ended or frame read failed. Save (S), arrow keys, and quit (Q) remain available.')
                        input_failed = True
                    point, down, erasing = None, False, False
                else:
                    # Mirror frames and run hand tracking/gesture recognition
                    frame = cv2.flip(frame, 1)
                    result = recognizer.update(tracker.detect(frame), frame.shape[1], frame.shape[0], now)
                    events = result.events
                    
                    # Select temporary tools from the current gesture pose
                    if result.pose == 'FIST':
                        temporary = 'eraser'
                        camera_point = result.pointer_point
                    elif result.pose == 'POINTER':
                        temporary = 'pointer'
                        camera_point = result.pointer_point
                    else:
                        camera_point = result.pen_point
                        
                    # Disable drawing/erasing when both hands are visible (e.g., clapping)
                    point = to_slide(camera_point, (frame.shape[1], frame.shape[0]), board.slide_size) if len(result.hands) == 1 else None
                    down = result.pen_down and temporary is None
                    erasing = result.pose == 'FIST' and len(result.hands) == 1
                    
                    # Determine status label from the current hand pose
                    if result.main_hand is None:
                        mode_label = "NO HAND"
                    elif result.pose == "FIST":
                        mode_label = "ERASER"
                    elif result.pose == "POINTER":
                        mode_label = "POINTER"
                    else:
                        tool_label = session.selected.upper()
                        mode_label = (
                            tool_label
                            if result.pen_down
                            else f"{tool_label} STOP"
                        )

                    # Draw the current mode label and pointer on the camera frame
                    cv2.putText(
                        frame,
                        f"MODE {mode_label}",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    
                    if camera_point is not None:
                        cv2.circle(frame, tuple(int(v) for v in camera_point), 7, (0, 255, 255), 2)
                    cv2.imshow(camera_win, frame)
                    
            # 8. Handle gesture events (highlighter toggle, navigation, etc.)
            transition = False
            for event in events:
                action = mapping.get(event)
                if action == 'toggle_highlighter':
                    session.select_tool('pen' if session.selected == 'highlighter' else 'highlighter', now)
                    transition = True
                elif action in ('page_next', 'page_prev') and not down:
                    session.change_page(1 if action == 'page_next' else -1, now)
                    transition = True
                    
            # Feed the current point and state to the handwriting session
            if not transition:
                session.feed(point, down, now, temporary_tool=temporary, eraser_active=erasing)
                
            # 9. Compose PDF slide and rendered display
            ox, oy, w, h = slide_rect()
            display = np.full((ch, cw, 3), 35, np.uint8)
            display[oy:oy + h, ox:ox + w] = board.render_page()
            
            # Draw cursor (eraser or pen)
            if session.cursor is not None:
                x, y = session.cursor
                radius = round(board.config.eraser_width / 2) if session.effective == 'eraser' else 6
                cv2.circle(display, (round(x + ox), round(y + oy)), radius, (0, 190, 255), 1, cv2.LINE_AA)
                
            # Render status and shortcut help text
            cv2.putText(display, f'{mode}  Page {board.current_page + 1}/{board.page_count}  {session.effective}  FPS {fps:.0f}',
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(display, 'P pen H highlight E erase L pointer | arrows/A/D page | Z undo S PDF I PNG Q exit',
                        (12, 49), cv2.FONT_HERSHEY_SIMPLEX, .45, (210, 210, 210), 1, cv2.LINE_AA)
                        
            # Check display timeout for notifications (e.g., save complete)
            if now < notice_until:
                cv2.putText(display, notice, (12, ch - 18), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 0), 2)
                
            cv2.imshow(win, display)
            
            # 10. Handle keyboard and window-close events
            key = cv2.waitKeyEx(16)
            closed = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
            if cap is not None:
                closed = closed or cv2.getWindowProperty(camera_win, cv2.WND_PROP_VISIBLE) < 1
                
            # Exit conditions (Q, Esc, or window close)
            if key in (ord('q'), 27) or closed:
                session.flush(time.monotonic())
                break
                
            # Keyboard shortcuts (tools, pages, undo, save)
            selected = {ord('p'): 'pen', ord('h'): 'highlighter', ord('e'): 'eraser', ord('l'): 'pointer'}
            if key in selected:
                session.select_tool(selected[key], time.monotonic())
            elif key in (2555904, 65363, ord('d')):  # Next page
                session.change_page(1, time.monotonic())
                mouse['point'] = None
            elif key in (2424832, 65361, ord('a')):  # Previous page
                session.change_page(-1, time.monotonic())
                mouse['point'] = None
            elif key == ord('z'):  # Undo
                session.flush(time.monotonic())
                board.undo()
                session.lift_required = True
            elif key in (ord('s'), ord('i')):  # Save as PDF (s) or image (i)
                session.flush(time.monotonic())
                session.lift_required = True
                suffix = '.pdf' if key == ord('s') else '.png'
                path = Path(args.output) / ('airboard_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + suffix)
                try:
                    saved = board.save_pdf(path) if suffix == '.pdf' else board.save_png(path)
                    print(f'Saved: {saved}')
                    notice = 'Saved (path in terminal)'
                except (OSError, ValueError, RuntimeError) as error:
                    print(f'Save failed: {error}')
                    notice = 'Save failed (see terminal)'
                notice_until = time.monotonic() + 2
                
    finally:
        # 11. Release resources and clean up
        if tracker is not None:
            tracker.close()
        if cap is not None:
            cap.release()
        board.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()