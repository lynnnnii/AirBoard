"""좌표 변환 및 UI 데모. --pdf로 팀 PDF 렌더러를 연결한다."""
import argparse
import math
import time
import json
from pathlib import Path
import cv2
import numpy as np
from airboard.calibration import Calibration
from .settings import Settings, ACTIONS, ROOT
from .controller import InputState, UIController, from_gesture_result
from .overlay import Overlay, bgr


class PreviewRenderer:
    """흰 배경에서도 PDF와 동일한 획/지우개/꼬리 제거를 사용한다."""
    def __init__(self, size: tuple[int,int]) -> None:
        import tempfile
        import pymupdf
        from dataclasses import replace
        from airboard.render import RenderConfig
        from airboard.smoothing.stroke import load_config
        self._temp = tempfile.TemporaryDirectory()
        path = Path(self._temp.name)/'blank.pdf'
        with pymupdf.open() as doc:
            doc.new_page(width=size[0],height=size[1]); doc.save(path)
        cfg = replace(RenderConfig.from_json(ROOT/'config/default.json'),max_width=size[0],max_height=size[1])
        self.renderer = PDFRenderer(str(path),cfg,smoothing_config={**load_config(),**Settings().ui['handwriting_smoothing']})
        self.previous = None

    def __del__(self):
        if hasattr(self, "renderer"):
            self.close()

    def consume(self, packet: object) -> np.ndarray:
        return self.renderer.consume(packet)

    def clear_page(self) -> None:
        self.renderer.clear_page()

    def close(self) -> None:
        self.renderer.close()
        self._temp.cleanup()


class PDFRenderer:
    """UI 패킷을 팀 PDF 렌더러의 공개 API에 연결한다."""
    def __init__(self, path: str, config: object, state_dir: Path | None = None, smoothing_config: dict | None = None) -> None:
        from airboard.render import AirBoardRenderer
        self.board = AirBoardRenderer(path, config)
        self.state_path = None if state_dir is None else Path(state_dir)/(self.board.document_key+'.json')
        self.smoother = None
        self._was_active = False
        self._erase_previous = None
        self.display_position = None
        self._ink_style = None
        self._last_tool = None
        if smoothing_config is not None:
            from airboard.smoothing.stroke import StrokeSmoother, make_filter
            from airboard.render import SmoothedInkBridge
            self.smoother = StrokeSmoother.from_config(smoothing_config)
            self.cursor_filter = make_filter(smoothing_config['method'], smoothing_config)
            self.bridge = SmoothedInkBridge(self.board)
        self._dirty = False
        self._last_save = time.monotonic()
        if self.state_path is not None and self.state_path.exists():
            try:
                self.board.load_session(self.state_path)
            except Exception:
                self.board.close()
                raise

    def finish_stroke(self) -> None:
        """꼬리 제거를 적용해 진행 획을 확정한다. 지우개도 먼저 확정한다."""
        if self.smoother is not None:
            from airboard.render import InkStyle
            if self._last_tool == 'eraser':
                self.board.end_stroke()
            self.bridge.consume(self.smoother.update(None,False,time.monotonic()),
                                self._ink_style or InkStyle())
            self.cursor_filter.reset()
        self.board.end_stroke()
        self._dirty = True

    def consume(self, packet: object) -> np.ndarray:
        """PDF 좌표를 팀 보정기에 통과시켜 미리보기와 최종 획을 분리한다."""
        self.display_position = packet.position
        if packet.tool != 'eraser' or not packet.active:
            self._erase_previous = None
        if packet.tool == 'eraser':
            self.finish_stroke()
            if packet.active and packet.position is not None:
                start = self._erase_previous or packet.position
                radius = packet.width/2
                steps = max(1,int(np.linalg.norm(np.asarray(packet.position)-start)/max(radius,1))+1)
                for p in np.linspace(start,packet.position,steps+1):
                    self.board.erase_stroke_at(tuple(p),radius=radius)
                self._erase_previous = packet.position
            self._dirty = True
            self.checkpoint(force=not packet.active)
            return self.board.render_page()
        if self.smoother is None:
            if packet.phase in ('down','up'):
                self.board.end_stroke()
            self.board.update(packet.position,packet.tool,packet.active,
                              eraser_active=packet.active and packet.tool=='eraser',
                              color=packet.color,
                              width=30 if packet.tool=='eraser' else packet.width)
        else:
            from airboard.render import InkStyle
            now=time.monotonic()
            if packet.tool != self._last_tool or packet.phase=='down':
                self.finish_stroke()
            self._last_tool=packet.tool
            if packet.tool in ('pen','highlighter'):
                self._ink_style=InkStyle(packet.tool,packet.color,packet.width)
                result=self.smoother.update(packet.position,packet.active,now)
                self.bridge.consume(result,self._ink_style)
                self.display_position=result.point if packet.active else packet.position
            else:
                if packet.position is None:
                    self.cursor_filter.reset()
                    point=None
                else:
                    point=tuple(float(v) for v in self.cursor_filter(packet.position,now))
                self.display_position=point
                self.board.update(point,packet.tool,False,
                                  eraser_active=packet.active and packet.tool=='eraser',
                                  color=packet.color,width=30)
        if packet.active or self._was_active:
            self._dirty = True
        self._was_active = packet.active
        self.checkpoint(force=not packet.active)
        return self.board.render_page()

    def checkpoint(self, force: bool = False) -> None:
        """획 종료 또는 2초 간격으로 진행 상태를 저장한다."""
        if self.state_path is not None and self._dirty and (force or time.monotonic()-self._last_save>=2):
            self.board.save_session(self.state_path)
            self._dirty = False
            self._last_save = time.monotonic()

    def change_page(self, offset: int) -> None:
        """페이지 변경과 직전 필기를 저장한다."""
        self.finish_stroke()
        self.board.change_page(offset)
        self._dirty = True
        self.checkpoint(force=True)

    def clear_page(self) -> None:
        """지운 상태도 저장하여 재실행 시 지운 획이 나타나지 않게 한다."""
        self.finish_stroke()
        self.board.clear_page()
        self._dirty = True
        self.checkpoint(force=True)

    def close(self) -> None:
        """종료 직전 진행 중 필기와 페이지를 저장하고 자원을 닫는다."""
        try:
            self.finish_stroke()
            self._dirty = True
            self.checkpoint(force=True)
        finally:
            self.board.close()


def reset_calibration(controller: UIController, renderer: object,
                      corners: list, path: object) -> None:
    """기본 영역을 저장하고 현재 획을 종료해 새 좌표계로 연결되지 않게 한다."""
    candidate = Calibration(corners, controller.calibration.size)
    candidate.save(path)
    if isinstance(renderer, PDFRenderer):
        renderer.finish_stroke()
    else:
        renderer.previous = None
    controller.calibration = candidate
    controller.active = False
    controller.blocked = True


def compose_screen(camera: np.ndarray, slide: np.ndarray, page: int,
                   tool: str, message: str = '', page_count: int | None = None) -> np.ndarray:
    """필기 영역 아래에 불투명 상태줄에 페이지와 도구 상태를 표시한다."""
    height = max(camera.shape[0], slide.shape[0])
    width = camera.shape[1] + slide.shape[1]
    screen = np.full((height + 78, width, 3), 235, np.uint8)
    screen[:camera.shape[0], :camera.shape[1]] = camera
    screen[:slide.shape[0], camera.shape[1]:] = slide
    screen[height:] = (32, 32, 32)
    page_label = f'PDF PAGE {page}/{page_count}' if page_count is not None else f'PREVIEW PAGE {page} (no PDF)'
    lines = [f'{page_label} | TOOL: {tool.upper()} | {message}',
             'S Settings  C Set area  R Reset area  X Clear  Q Quit | UP/DOWN Page']
    for i, line in enumerate(lines):
        cv2.putText(screen, line, (16, height + 28 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
    return screen


def draw_settings(slide: np.ndarray, mapping: dict, row: int) -> None:
    """설정 안내를 큰 글씨와 불투명 배경으로 분리해 축소 화면에서도 읽게 한다."""
    slide[:] = 245
    cv2.rectangle(slide, (0, 0), (slide.shape[1]-1, 153), (32,32,32), -1)
    lines = ['GESTURE SETTINGS', 'UP / DOWN: select gesture',
             'LEFT / RIGHT: change action', 'ENTER: save   ESC: cancel   R: defaults']
    for i, text in enumerate(lines):
        cv2.putText(slide, text, (20, 32+i*34), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (255,255,255), 2, cv2.LINE_AA)
    for i, (gesture, action) in enumerate(mapping.items()):
        y = 188+i*50
        if i == row:
            cv2.rectangle(slide, (10,y-27), (slide.shape[1]-11,y+12), (220,232,250), -1)
        cv2.putText(slide, f'{gesture} -> {action}', (20,y), cv2.FONT_HERSHEY_SIMPLEX,
                    .75, (20,20,160) if i==row else (30,30,30), 2, cv2.LINE_AA)


def main() -> None:
    """웹캠 기본; --mouse/--virtual/--video 및 화면 없는 회귀 실행 지원."""
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--mouse',action='store_true')
    sources.add_argument('--virtual',action='store_true')
    sources.add_argument('--video')
    parser.add_argument('--trace-gestures', action='store_true', help='Write numeric gesture trace locally without camera images')
    parser.add_argument('--debug-gestures', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--pdf', help='PDF file to display and annotate')
    parser.add_argument('--no-smoothing', action='store_true', help='Compare PDF input without smoothing or tail trimming')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--camera-index',type=int,default=0)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--frames',type=int,default=0)
    args = parser.parse_args()
    if args.headless and (not args.virtual or args.frames<=0):
        parser.error('--headless requires --virtual --frames N (N>0)')
    settings = Settings()
    size = tuple(settings.ui['slide_size'])
    calfile = ROOT/'config/calibration-user.json'
    calibration = Calibration.load(calfile) if calfile.exists() else Calibration(settings.ui['default_corners'],size)
    renderer = None if args.pdf else PreviewRenderer(calibration.size)
    if args.pdf:
        from dataclasses import replace
        from airboard.render import RenderConfig
        cfg = replace(RenderConfig.from_json(ROOT/'config/default.json'), max_width=1280, max_height=720)
        from airboard.smoothing.stroke import load_config
        smoothing_cfg = None if args.no_smoothing else {**load_config(str(ROOT/'config/default.json')),
                                                       **settings.ui.get('handwriting_smoothing',{})}
        renderer = PDFRenderer(args.pdf, cfg, state_dir=ROOT/'data/sessions',smoothing_config=smoothing_cfg)
        calibration = Calibration(calibration.corners.tolist(), renderer.board.slide_size)
    controller = UIController(calibration,settings)
    overlay = Overlay(settings)
    camera_height = calibration.size[1]
    camera_width = round(camera_height*4/3)

    def move_page(offset: int) -> str:
        """실제 페이지를 이동하고 페이지 크기에 맞춰 좌표 변환을 갱신한다."""
        nonlocal page
        previous_page = page
        if args.pdf:
            renderer.change_page(offset)
            page = renderer.board.current_page+1
            controller.calibration = Calibration(controller.calibration.corners.tolist(),renderer.board.slide_size)
            controller.active = False
            controller.blocked = True
        else:
            page = max(1,page+offset)
            renderer.previous = None
        if page == previous_page:
            return 'Page boundary reached'
        return ('NEXT' if offset>0 else 'PREVIOUS') + f': {previous_page} -> {page}'
    cap,tracker = None,None
    trace_stream = None
    mouse = {'position':None,'down':False}
    calibration_points = None
    editing,row = False,0
    page = renderer.board.current_page+1 if args.pdf else 1
    message = ''

    def mouse_event(kind: int, x: int, y: int, flags: int, param: object) -> None:
        """왼쪽 640×480 입력 패널의 마우스를 카메라 정규화 좌표로 해석."""
        mouse['position'] = (x/(camera_width-1),y/(camera_height-1)) if 0<=x<camera_width and 0<=y<camera_height else None
        if kind==cv2.EVENT_LBUTTONDOWN:
            mouse['down'] = True
            if calibration_points is not None and mouse['position'] is not None:
                calibration_points.append(mouse['position'])
                mouse['down'] = False
        elif kind==cv2.EVENT_LBUTTONUP:
            mouse['down'] = False

    try:
        if args.trace_gestures:
            trace_dir=ROOT/'data/diagnostics'
            trace_dir.mkdir(parents=True,exist_ok=True)
            trace_path=trace_dir/f'gestures-{time.time_ns()}.jsonl'
            trace_stream=trace_path.open('w',encoding='utf-8')
            print(f'Gesture trace: {trace_path}')
        if not args.mouse and not args.virtual:
            from airboard.input.camera import open_camera
            from airboard.tracking.hand_tracker import HandTracker
            from airboard.gestures.gesture import GestureRecognizer
            cap = cv2.VideoCapture(args.video) if args.video else open_camera(args.camera_index)
            if cap is None or not cap.isOpened():
                raise RuntimeError('Camera unavailable. Close other apps, check permissions or use --camera-index 1 / --mouse.')
            from .settings import read_json
            all_cfg = read_json(ROOT/'config/default.json')
            gesture_cfg = all_cfg.get('gestures',{})
            tracker = HandTracker(detection_confidence=all_cfg.get('tracking',{}).get('detection_confidence',.5))
            recognizer = GestureRecognizer(gesture_cfg)
        if not args.headless:
            cv2.namedWindow('AirBoard UI demo',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('AirBoard UI demo',1800,1000)
            if args.fullscreen:
                cv2.setWindowProperty('AirBoard UI demo',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback('AirBoard UI demo',mouse_event)
        frame_number = 0
        input_failed = False
        while True:
            now = time.monotonic()
            camera = np.full((480,640,3),35,np.uint8)
            trace_result = None
            if tracker:
                ok,frame = cap.read() if not input_failed else (False,None)
                if not ok:
                    input_failed = True
                    state = InputState(None)
                    cv2.putText(camera,'Input stopped',(20,220),cv2.FONT_HERSHEY_SIMPLEX,
                                1.,(255,255,255),2,cv2.LINE_AA)
                else:
                    frame = cv2.flip(frame,1)
                    h,w = frame.shape[:2]
                    hands = tracker.detect(frame)
                    result = recognizer.update(hands,w,h,now)
                    trace_result = result
                    state = from_gesture_result(result,(w,h))
                    camera = cv2.resize(frame,(640,480))
            elif args.virtual:
                t = frame_number/30
                state = InputState((.5+.28*math.cos(t),.5+.27*math.sin(t*1.8)),frozenset({'POINTER'} if t%4<3 else ()))
            else:
                gestures = {'POINTER'} if mouse['down'] else set()
                state = InputState(mouse['position'],frozenset(gestures))
            if calibration_points is not None and len(calibration_points)>=4:
                try:
                    candidate = Calibration(calibration_points[:4],controller.calibration.size)
                    candidate.save(calfile)
                    controller.calibration = candidate
                    message = 'Calibration saved'
                except (ValueError,OSError) as exc:
                    print(exc)
                    message = 'Calibration rejected; previous area kept. Press C to retry.'
                calibration_points = None
            pos = controller.calibration.transform(state.position)
            selected = overlay.dwell(pos if not editing and calibration_points is None else None,now)
            if selected:
                setattr(controller,*selected)
            packet = controller.update(state,now,
                                       suppressed=editing or calibration_points is not None or overlay.target is not None,
                                       navigation_suppressed=editing or calibration_points is not None or selected is not None)
            for command in packet.commands:
                message = move_page(1 if command=='page_next' else -1)
            slide = renderer.consume(packet)
            if trace_stream is not None:
                record={'t':now, 'pose':trace_result.pose if trace_result else 'NONE',
                        'hands':len(trace_result.hands) if trace_result else 0,
                        'events':trace_result.events if trace_result else [],
                        'metrics':trace_result.debug if trace_result else {},
                        'tool':packet.tool,'active':packet.active,'phase':packet.phase,
                        'raw_slide_position':packet.position,
                        'filtered_slide_position':renderer.display_position if args.pdf else packet.position,
                        'palette':overlay.target is not None,
                        'outside':state.position is not None and pos is None,
                        'blocked':controller.blocked,'navigation':controller.navigation_status}
                trace_stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                if frame_number%30==0:
                    trace_stream.flush()
            notice = controller.notification if now<controller.notice_until else ''
            if args.pdf:
                from dataclasses import replace
                packet=replace(packet,position=renderer.display_position)
            overlay.draw(slide,packet,notice)
            contour = np.rint(controller.calibration.corners*np.float32([639,479])).astype(np.int32)
            cv2.polylines(camera,[contour],True,(0,180,180),2)
            for i,p in enumerate(controller.calibration.corners):
                cv2.circle(camera,(round(float(p[0])*639),round(float(p[1])*479)),5,(0,255,255),-1)
            if state.position:
                mapped = controller.calibration.transform(state.position)
                if mapped is not None:
                    cv2.putText(camera,f'PDF X {mapped[0]/(controller.calibration.size[0]-1):.0%}  Y {mapped[1]/(controller.calibration.size[1]-1):.0%}',(15,450),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                cv2.circle(camera,(round(state.position[0]*639),round(state.position[1]*479)),6,(0,255,0),2)
            if calibration_points is not None:
                label = f'CAL {len(calibration_points)+1}/4 TL TR BR BL: SPACE or click'
                cv2.putText(camera,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                for i,p in enumerate(calibration_points):
                    cv2.putText(camera,str(i+1),(round(p[0]*639),round(p[1]*479)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
            if editing:
                draw_settings(slide, settings.mapping, row)
            frame_number += 1
            if args.frames and frame_number>=args.frames:
                print(f'Processed {frame_number} frames; last phase={packet.phase}')
                break
            if args.headless:
                continue
            screen = compose_screen(cv2.resize(camera,(camera_width,camera_height)), slide, page, packet.tool, message,
                                    renderer.board.page_count if args.pdf else None)
            cv2.imshow('AirBoard UI demo',screen)
            key = cv2.waitKeyEx(1)
            if key==ord('q') or cv2.getWindowProperty('AirBoard UI demo',cv2.WND_PROP_VISIBLE)<1:
                break
            up,down,left,right = (2490368,2621440,2424832,2555904)
            if editing:
                names = list(settings.mapping)
                if key in (up,down):
                    row = (row+(-1 if key==up else 1))%len(names)
                elif key in (left,right):
                    name = names[row]
                    settings.mapping[name] = ACTIONS[(ACTIONS.index(settings.mapping[name])+(-1 if key==left else 1))%len(ACTIONS)]
                elif key==ord('r'):
                    from .settings import read_json
                    settings.mapping = read_json(ROOT/'config/gestures.json')
                    row = 0
                    message = 'Default mapping restored; ENTER to save'
                elif key in (10,13):
                    try:
                        settings.save()
                        editing = False
                        message = ''
                    except (ValueError,OSError) as exc:
                        message = str(exc)
                elif key==27:
                    settings.mapping = backup.copy()
                    editing = False
                continue
            if key==27:
                break
            if key==ord('c'):
                calibration_points = []
            elif key==ord('r'):
                try:
                    reset_calibration(controller,renderer,settings.ui['default_corners'],calfile)
                    calibration_points = None
                    overlay.dwell(None,time.monotonic())
                    message = ''
                except (OSError,ValueError) as exc:
                    message = f'Reset failed: {exc}'
            elif key==32 and calibration_points is not None and state.position is not None:
                calibration_points.append(state.position)
            elif key in (ord('['),ord(']')):
                controller.writing_scale = round(min(1.,max(.3,controller.writing_scale+(-.1 if key==ord('[') else .1))),1)
                message = f'Writing size: {controller.writing_scale:.0%} (next stroke)'
            elif key==ord('s'):
                backup = settings.mapping.copy()
                editing,row = True,0
            elif key in (up,down):
                message = move_page(-1 if key==up else 1)
            elif key==ord('x'):
                if args.pdf:
                    renderer.clear_page()
                else:
                    renderer.clear_page()
                    renderer.previous = None
                controller.active = False
                controller.blocked = True
    finally:
        try:
            renderer.close()
        finally:
            if trace_stream is not None:
                trace_stream.close()
            if tracker:
                tracker.close()
            if cap is not None:
                cap.release()
            if not args.headless:
                cv2.destroyAllWindows()


if __name__=='__main__':
    main()
