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
    """테스트용 비율 매핑 함수.
    
    실제 호모그래피 캘리브레이션이 아니며, 통합 시 실제 캘리브레이션 모듈로 교체됩니다.
    """
    if point is None:
        return None
    fw, fh = frame_size
    sw, sh = slide_size
    return float(point[0]) / fw * (sw - 1), float(point[1]) / fh * (sh - 1)


def main() -> None:
    """팀의 기존 보정/추적 모듈을 조합해 마우스 또는 영상 입력으로 필기를 시험하는 메인 함수."""
    
    # 1. 커맨드라인 인자 파싱 설정
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
    
    # 2. 팀원이 제공한 실제 보정기 로드 (재구현하거나 수정하지 않음)
    from airboard.smoothing.stroke import StrokeSmoother, load_config, make_filter
    cfg = load_config(args.config)
    if args.no_smoothing:
        cfg['method'] = 'none'  # 필터만 끄고, start_skip/end_trim 설정은 유지
    
    # 3. 렌더러 및 세션 초기화
    board = AirBoardRenderer(args.pdf, RenderConfig.from_json(args.config))
    session = RenderSession(board, StrokeSmoother.from_config(cfg), make_filter(cfg['method'], cfg))
    
    cap = tracker = recognizer = None
    win = 'AirBoard PDF - render test'
    camera_win = 'AirBoard camera - test mapping'
    toolbar = 64
    cw, ch = board.config.max_width, board.config.max_height + toolbar
    mouse = {'point': None, 'down': False}
    mapping = {}
    
    # 4. 제스처 매핑 JSON 파일 로드
    if Path(args.mapping).exists():
        mapping = json.loads(Path(args.mapping).read_text(encoding='utf-8'))
        if not isinstance(mapping, dict):
            board.close()
            raise ValueError('제스처 매핑은 JSON 객체여야 합니다. 웹캠 데모에서는 raw event: action 형식을 사용합니다.')
    
    notice = ''
    notice_until = 0.0
    input_failed = False
    mode = 'VIDEO' if args.video else 'WEBCAM' if args.webcam else 'MOUSE'

    def slide_rect() -> tuple[int, int, int, int]:
        """현재 화면에서 슬라이드 이미지의 위치(x, y)와 크기(w, h)를 계산."""
        w, h = board.slide_size
        return (cw - w) // 2, toolbar + (board.config.max_height - h) // 2, w, h

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        """마우스 좌표 및 버튼 상태 갱신 (실제 필기 처리는 메인 루프에서 수행)."""
        ox, oy, w, h = slide_rect()
        mouse['point'] = (float(x - ox), float(y - oy)) if ox <= x < ox + w and oy <= y < oy + h else None
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse['down'] = True
        elif event == cv2.EVENT_LBUTTONUP or (event == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_LBUTTON):
            mouse['down'] = False

    try:
        # 5. 웹캠 또는 동영상 입력 모드 설정
        if args.webcam or args.video:
            from airboard.input.camera import open_camera
            from airboard.tracking.hand_tracker import HandTracker, DEFAULT_MODEL
            from airboard.gestures.gesture import GestureRecognizer
            
            if not Path(DEFAULT_MODEL).is_file():
                raise FileNotFoundError(f'손 모델이 없습니다: {DEFAULT_MODEL}')
                
            tracker = HandTracker()
            recognizer = GestureRecognizer()
            cap = cv2.VideoCapture(args.video) if args.video else open_camera(args.camera)
            
            if cap is None or not cap.isOpened():
                raise RuntimeError('영상 입력을 열지 못했습니다. 카메라/영상 경로를 확인하세요.')
            cv2.namedWindow(camera_win, cv2.WINDOW_AUTOSIZE)
            
        # 6. OpenCV 윈도우 및 마우스 콜백 설정
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, mouse_callback)
        previous = time.monotonic()
        fps = 0.0
        
        # 7. 메인 렌더링 및 이벤트 처리 루프
        while True:
            now = time.monotonic()
            fps = .9 * fps + .1 / max(now - previous, 1e-6)
            previous = now
            events = []
            temporary = None
            
            # 입력 소스(마우스 vs 웹캠/비디오)에 따른 좌표 및 상태 처리
            if cap is None:
                # 마우스 모드
                point = mouse['point']
                down = bool(mouse['down'])
                erasing = session.selected == 'eraser' and down
            else:
                # 웹캠 / 비디오 모드
                ok, frame = cap.read() if not input_failed else (False, None)
                if not ok:
                    if not input_failed:
                        session.flush(now)
                        print('영상 종료 또는 읽기 실패. S 저장/방향키/Q 종료는 계속 사용할 수 있습니다.')
                        input_failed = True
                    point, down, erasing = None, False, False
                else:
                    # 화면 좌우 반전 및 손 추적/제스처 인식 수행
                    frame = cv2.flip(frame, 1)
                    result = recognizer.update(tracker.detect(frame), frame.shape[1], frame.shape[0], now)
                    events = result.events
                    
                    # 손 동작(포즈)에 따른 임시 도구 설정
                    if result.pose == 'FIST':
                        temporary = 'eraser'
                        camera_point = result.pointer_point
                    elif result.pose == 'POINTER':
                        temporary = 'pointer'
                        camera_point = result.pointer_point
                    else:
                        camera_point = result.pen_point
                        
                    # 양손이 모두 감지된 경우(박수 등) 필기/지우개 비활성화
                    point = to_slide(camera_point, (frame.shape[1], frame.shape[0]), board.slide_size) if len(result.hands) == 1 else None
                    down = result.pen_down and temporary is None
                    erasing = result.pose == 'FIST' and len(result.hands) == 1
                    
                    # 현재 손 상태에 따른 모드 레이블 결정
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

                    # 카메라 화면에 현재 모드 텍스트 및 포인터 표시
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
                    
            # 8. 제스처 이벤트 처리 (하이라이터 토글, 페이지 이동 등)
            transition = False
            for event in events:
                action = mapping.get(event)
                if action == 'toggle_highlighter':
                    session.select_tool('pen' if session.selected == 'highlighter' else 'highlighter', now)
                    transition = True
                elif action in ('page_next', 'page_prev') and not down:
                    session.change_page(1 if action == 'page_next' else -1, now)
                    transition = True
                    
            # 필기 세션에 현재 포인트와 상태 피드
            if not transition:
                session.feed(point, down, now, temporary_tool=temporary, eraser_active=erasing)
                
            # 9. PDF 슬라이드 및 렌더링 화면 구성
            ox, oy, w, h = slide_rect()
            display = np.full((ch, cw, 3), 35, np.uint8)
            display[oy:oy + h, ox:ox + w] = board.render_page()
            
            # 커서(지우개 또는 펜) 표시
            if session.cursor is not None:
                x, y = session.cursor
                radius = round(board.config.eraser_width / 2) if session.effective == 'eraser' else 6
                cv2.circle(display, (round(x + ox), round(y + oy)), radius, (0, 190, 255), 1, cv2.LINE_AA)
                
            # 상태 정보 및 단축키 안내 텍스트 출력
            cv2.putText(display, f'{mode}  Page {board.current_page + 1}/{board.page_count}  {session.effective}  FPS {fps:.0f}',
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(display, 'P pen H highlight E erase L pointer | arrows/A/D page | Z undo S PDF I PNG Q exit',
                        (12, 49), cv2.FONT_HERSHEY_SIMPLEX, .45, (210, 210, 210), 1, cv2.LINE_AA)
                        
            # 안내 메시지(저장 완료 등) 출력 유지 시간 확인
            if now < notice_until:
                cv2.putText(display, notice, (12, ch - 18), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 0), 2)
                
            cv2.imshow(win, display)
            
            # 10. 키보드 입력 및 창 닫기 이벤트 처리
            key = cv2.waitKeyEx(16)
            closed = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
            if cap is not None:
                closed = closed or cv2.getWindowProperty(camera_win, cv2.WND_PROP_VISIBLE) < 1
                
            # 종료 조건 (Q 키, ESC, 창 닫기 버튼)
            if key in (ord('q'), 27) or closed:
                session.flush(time.monotonic())
                break
                
            # 단축키 매핑 (도구 변경, 페이지 이동, 되돌리기, 저장)
            selected = {ord('p'): 'pen', ord('h'): 'highlighter', ord('e'): 'eraser', ord('l'): 'pointer'}
            if key in selected:
                session.select_tool(selected[key], time.monotonic())
            elif key in (2555904, 65363, ord('d')):  # 다음 페이지
                session.change_page(1, time.monotonic())
                mouse['point'] = None
            elif key in (2424832, 65361, ord('a')):  # 이전 페이지
                session.change_page(-1, time.monotonic())
                mouse['point'] = None
            elif key == ord('z'):  # 실행 취소 (Undo)
                session.flush(time.monotonic())
                board.undo()
                session.lift_required = True
            elif key in (ord('s'), ord('i')):  # PDF(s) 또는 이미지(i)로 저장
                session.flush(time.monotonic())
                session.lift_required = True
                suffix = '.pdf' if key == ord('s') else '.png'
                path = Path(args.output) / ('airboard_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + suffix)
                try:
                    saved = board.save_pdf(path) if suffix == '.pdf' else board.save_png(path)
                    print(f'저장 완료: {saved}')
                    notice = 'Saved (path in terminal)'
                except (OSError, ValueError, RuntimeError) as error:
                    print(f'저장 실패: {error}')
                    notice = 'Save failed (see terminal)'
                notice_until = time.monotonic() + 2
                
    finally:
        # 11. 자원 해제 및 정리
        if tracker is not None:
            tracker.close()
        if cap is not None:
            cap.release()
        board.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()