"""좌표 변환 및 UI 데모. --pdf로 팀 PDF 렌더러를 연결한다."""
import argparse
import math
import time
import json
from pathlib import Path
import cv2
import numpy as np
from airboard.calibration import Calibration
from .settings import Settings, ROOT, allowed_actions
from .controller import InputState, UIController, from_gesture_result
from .overlay import Overlay, bgr


class PreviewRenderer:
    """흰 배경 데모 전용 레이어. PDF/저장/undo는 팀 render 담당."""
    def __init__(self, size: tuple[int,int]) -> None:
        w,h = size
        self.ink = np.full((h,w,3),255,np.uint8)
        self.previous = None

    def consume(self, packet: object) -> np.ndarray:
        """렌더 패킷을 소비하고 UI를 그릴 복사본을 반환한다."""
        if not packet.active:
            self.previous = None
        if packet.active and packet.position:
            p = tuple(round(v) for v in packet.position)
            a = self.previous or p
            if packet.tool=='eraser':
                cv2.line(self.ink,a,p,(255,255,255),30,cv2.LINE_AA)
            elif packet.tool=='highlighter':
                layer = self.ink.copy()
                cv2.line(layer,a,p,bgr(packet.color),packet.width,cv2.LINE_AA)
                cv2.addWeighted(layer,packet.opacity,self.ink,1-packet.opacity,0,dst=self.ink)
            else:
                cv2.line(self.ink,a,p,bgr(packet.color),packet.width,cv2.LINE_AA)
            self.previous = p
        return self.ink.copy()


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
            from airboard.smoothing.filters import NoFilter
            from airboard.smoothing.stroke import StrokeSmoother, make_filter
            from airboard.render import SmoothedInkBridge
            self.smoother = StrokeSmoother.from_config(smoothing_config)
            # 형광펜은 슬라이드의 글자를 정확히 덮어야 해서 보정·곡선 없이 원래 좌표를 쓴다 (끝 꼬리 제거만 같게).
            self.highlighter_smoother = StrokeSmoother(NoFilter(),smoothing_config.get('start_skip',0),
                                                       smoothing_config.get('end_trim',0))
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

    def _smoother_for(self, tool: str | None) -> object:
        """펜은 떨림 보정·곡선, 형광펜은 보정 없음."""
        return self.highlighter_smoother if tool=='highlighter' else self.smoother

    def finish_stroke(self) -> None:
        """꼬리 제거를 적용해 진행 획을 확정한다. 지우개도 먼저 확정한다."""
        if self.smoother is not None:
            from airboard.render import InkStyle
            if self._last_tool == 'eraser':
                self.board.end_stroke()
            self.bridge.consume(self._smoother_for(self._last_tool).update(None,False,time.monotonic()),
                                self._ink_style or InkStyle())
            self.cursor_filter.reset()
        self.board.end_stroke()
        self._dirty = True

    def consume(self, packet: object) -> np.ndarray:
        """PDF 좌표를 팀 보정기에 통과시켜 미리보기와 최종 획을 분리한다."""
        self.display_position = packet.position
        tail = None
        if packet.tool != 'eraser' or not packet.active:
            self._erase_previous = None
        if packet.tool == 'eraser':
            # 주먹을 쥔 채 지나간 길을 반경 간격으로 나눠 닿은 획을 통째로 지운다 (빠르게 움직여도 건너뛰지 않음).
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
                result=self._smoother_for(packet.tool).update(packet.position,packet.active,now)
                self.bridge.consume(result,self._ink_style)
                self.display_position=result.point
                if packet.active and result.stroke and result.point is not None:
                    # 곡선 보간은 마지막 한 구간을 다음 프레임에 확정하므로, 그 사이는 직선으로 잠깐 보여 늦게 따라오지 않게 한다.
                    tail=(result.stroke[-1],result.point,self._ink_style)
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
        image = self.board.render_page()
        if tail is not None and tail[2].tool=='pen':
            a,b,style = tail
            cv2.line(image,tuple(round(v) for v in a),tuple(round(v) for v in b),bgr(style.color),
                     max(1,round(style.width)),cv2.LINE_AA)
        return image

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

    def undo(self) -> bool:
        """현재 페이지의 마지막 획/지우개 작업을 취소한다."""
        self.finish_stroke()
        changed = self.board.undo()
        self._dirty = True
        self.checkpoint(force=True)
        return changed

    def redo(self) -> bool:
        """취소한 마지막 획/지우개 작업을 다시 적용한다."""
        self.finish_stroke()
        changed = self.board.redo()
        self._dirty = True
        self.checkpoint(force=True)
        return changed

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
             'S Settings  C Set area  R Reset area  X Clear  [ ] Text size  Q Quit']
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
    renderer = PreviewRenderer(calibration.size)
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
    camera_width, camera_height = (512,384) if args.pdf else (640,480)

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
                    # 매핑에 따라 인식기 판정을 켜고 끈다 (설정 화면에서 바꾸면 바로 반영)
                    #  - 검지로 쓰기: 검지 자세 유지와 ✌️ 인식
                    #  - ✌️를 멈춤 말고 다른 동작에 연결: ✌️ 인식 (멈춤은 검지로 쓸 때만 의미가 있다)
                    #  - 핀치로 쓰기: 손가락이 벌어지기 시작하면 잉크 멈춤 (지우개 등으로 쓸 때는 끈다)
                    recognizer.config['index_writing'] = settings.mapping.get('POINTER') == 'draw'
                    recognizer.config['two_fingers'] = settings.mapping.get('TWO_FINGERS','none') not in ('none','stop')
                    recognizer.config['pinch_writing'] = settings.mapping.get('PINCH') == 'draw'
                    result = recognizer.update(hands,w,h,now)
                    trace_result = result
                    state = from_gesture_result(result,(w,h))
                    camera = cv2.resize(frame,(640,480))
            elif args.virtual:
                t = frame_number/30
                state = InputState((.5+.28*math.cos(t),.5+.27*math.sin(t*1.8)),frozenset({'PINCH'} if t%4<3 else ()))
            else:
                gestures = {'PINCH'} if mouse['down'] else set()
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
                if command in ('undo','redo'):
                    if args.pdf:
                        changed = renderer.undo() if command == 'undo' else renderer.redo()
                        message = ('UNDO' if command == 'undo' else 'REDO') if changed else 'Nothing to restore'
                    else:
                        message = 'Undo/redo requires a PDF'
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
                cv2.circle(camera,(round(state.position[0]*639),round(state.position[1]*479)),6,(0,255,0),2)
            if calibration_points is not None:
                label = f'CAL {len(calibration_points)+1}/4 TL TR BR BL: SPACE or click'
                cv2.putText(camera,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                for i,p in enumerate(calibration_points):
                    cv2.putText(camera,str(i+1),(round(p[0]*639),round(p[1]*479)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
            if editing:
                visible_mapping = {name:action for name,action in settings.mapping.items()
                                   if name not in ('SWIPE_LEFT','SWIPE_RIGHT')}
                draw_settings(slide, visible_mapping, row)
            frame_number += 1
            if args.frames and frame_number>=args.frames:
                print(f'Processed {frame_number} frames; last phase={packet.phase}')
                break
            if args.headless:
                continue
            screen = compose_screen(cv2.resize(camera,(camera_width,camera_height)), slide, page, packet.tool, message,
                                    renderer.board.page_count if args.pdf else None)
            cv2.imshow('AirBoard UI demo',screen)
            key = cv2.waitKeyEx(16)
            if key==ord('q') or cv2.getWindowProperty('AirBoard UI demo',cv2.WND_PROP_VISIBLE)<1:
                break
            up,down,left,right = (2490368,2621440,2424832,2555904)
            if editing:
                names = [name for name in settings.mapping if name not in ('SWIPE_LEFT','SWIPE_RIGHT')]
                if key in (up,down):
                    row = (row+(-1 if key==up else 1))%len(names)
                    if 'draw' not in settings.mapping.values():
                        message = 'SAVE BLOCKED: choose a held gesture and assign DRAW.'
                elif key in (left,right):
                    # 이 제스처에 의미 있는 동작만 돌아가며 고른다. draw는 여러 제스처에 지정할 수 있다.
                    name = names[row]
                    choices = allowed_actions(name)
                    current = settings.mapping[name]
                    i = choices.index(current) if current in choices else 0
                    settings.set_action(name, choices[(i+(-1 if key==left else 1))%len(choices)])
                    message = (f'DRAW assigned to {name}; press ENTER to save'
                               if settings.mapping[name] == 'draw' else
                               'SAVE BLOCKED: choose a held gesture and assign DRAW.'
                               if 'draw' not in settings.mapping.values() else '')
                elif key==ord('r'):
                    from .settings import read_json
                    settings.mapping = read_json(ROOT/'config/gestures.json')
                    row = 0
                    message = 'Default mapping restored; press ENTER to save'
                elif key in (10,13):
                    try:
                        settings.save()
                        editing = False
                        message = 'Settings saved'
                    except ValueError as exc:
                        if 'draw' in str(exc).lower():
                            message = 'SAVE BLOCKED: choose a held gesture and assign DRAW.'
                        else:
                            message = f'SAVE ERROR: {exc}'
                    except OSError:
                        message = 'SAVE ERROR: could not write config/ui-user.json'
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
            elif key==ord('s'):
                backup = settings.mapping.copy()
                editing,row = True,0
            elif key in (left,right):
                message = move_page(-1 if key==left else 1)
            elif key in (ord('['),ord(']')):
                # 글씨 크기 30~100% 조절 (이번 실행에만 적용, 기본값은 config의 ui.writing_scale)
                step = -.1 if key==ord('[') else .1
                controller.writing_scale = round(min(1.,max(.3,controller.writing_scale+step)),1)
                message = f'Text size {round(controller.writing_scale*100)}%'
            elif key==ord('x'):
                if args.pdf:
                    renderer.clear_page()
                else:
                    renderer.ink[:] = 255
                    renderer.previous = None
                controller.active = False
                controller.blocked = True
    finally:
        try:
            if args.pdf:
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
