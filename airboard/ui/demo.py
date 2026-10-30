"""임준서 파트 실행 어댑터. 팀 main.py 및 render 모듈과 독립적이다."""
import argparse
import math
import time
import cv2
import numpy as np
from airboard.calibration import Calibration
from .settings import Settings, ACTIONS, ROOT
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
                cv2.line(layer,a,p,bgr(packet.color),packet.width*4,cv2.LINE_AA)
                cv2.addWeighted(layer,packet.opacity,self.ink,1-packet.opacity,0,dst=self.ink)
            else:
                cv2.line(self.ink,a,p,bgr(packet.color),packet.width,cv2.LINE_AA)
            self.previous = p
        return self.ink.copy()


def main() -> None:
    """웹캠 기본; --mouse/--virtual/--video 및 화면 없는 회귀 실행 지원."""
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--mouse',action='store_true')
    sources.add_argument('--virtual',action='store_true')
    sources.add_argument('--video')
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
    controller = UIController(calibration,settings)
    overlay,renderer = Overlay(settings),PreviewRenderer(calibration.size)
    cap,tracker = None,None
    mouse = {'position':None,'down':False}
    calibration_points = None
    editing,row = False,0
    simulated = set()
    event = set()
    page = 1
    message = ''

    def mouse_event(kind: int, x: int, y: int, flags: int, param: object) -> None:
        """왼쪽 640×480 입력 패널의 마우스를 카메라 정규화 좌표로 해석."""
        mouse['position'] = (x/639,y/479) if 0<=x<640 and 0<=y<480 else None
        if kind==cv2.EVENT_LBUTTONDOWN:
            mouse['down'] = True
            if calibration_points is not None and mouse['position'] is not None:
                calibration_points.append(mouse['position'])
                mouse['down'] = False
        elif kind==cv2.EVENT_LBUTTONUP:
            mouse['down'] = False

    try:
        if not args.mouse and not args.virtual:
            from airboard.input.camera import open_camera
            from airboard.tracking.hand_tracker import HandTracker
            from airboard.gestures.gesture import GestureRecognizer
            cap = cv2.VideoCapture(args.video) if args.video else open_camera(args.camera_index)
            if cap is None or not cap.isOpened():
                raise RuntimeError('Camera unavailable. Close other apps, check permissions or use --camera-index 1 / --mouse.')
            tracker,recognizer = HandTracker(),GestureRecognizer()
        if not args.headless:
            cv2.namedWindow('AirBoard UI demo',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('AirBoard UI demo',1200,405)
            cv2.setMouseCallback('AirBoard UI demo',mouse_event)
        frame_number = 0
        while True:
            now = time.monotonic()
            camera = np.full((480,640,3),35,np.uint8)
            if tracker:
                ok,frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame,1)
                h,w = frame.shape[:2]
                hands = tracker.detect(frame)
                result = recognizer.update(hands,w,h,now)
                state = from_gesture_result(result,(w,h))
                camera = cv2.resize(frame,(640,480))
            elif args.virtual:
                t = frame_number/30
                state = InputState((.5+.28*math.cos(t),.5+.27*math.sin(t*1.8)),frozenset({'PINCH'} if t%4<3 else ()))
            else:
                gestures = simulated | ({'PINCH'} if mouse['down'] else set()) | event
                state = InputState(mouse['position'],frozenset(gestures))
            event.clear()
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
            packet = controller.update(state,now,suppressed=editing or calibration_points is not None or overlay.target is not None)
            for command in packet.commands:
                page = max(1,page+(1 if command=='page_next' else -1))
            slide = renderer.consume(packet)
            notice = controller.notification if now<controller.notice_until else ''
            overlay.draw(slide,packet,notice)
            cv2.putText(slide,f'Preview page {page} (no PDF) | {message}',(12,slide.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,.5,(50,50,50),1)
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
                slide[:] = 245
                cv2.putText(slide,'SETTINGS: UP/DOWN row, LEFT/RIGHT action, ENTER save, ESC cancel',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
                for i,(gesture,action) in enumerate(settings.mapping.items()):
                    cv2.putText(slide,f'{gesture} -> {action}',(20,75+i*40),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,220) if i==row else (40,40,40),2)
            frame_number += 1
            if args.frames and frame_number>=args.frames:
                print(f'Processed {frame_number} frames; last phase={packet.phase}')
                break
            if args.headless:
                continue
            height = max(480,slide.shape[0])
            screen = np.full((height,640+slide.shape[1],3),235,np.uint8)
            screen[:480,:640] = camera
            screen[:slide.shape[0],640:] = slide
            cv2.imshow('AirBoard UI demo',screen)
            key = cv2.waitKeyEx(16)
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
                elif key in (10,13):
                    try:
                        settings.save()
                        editing = False
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
            elif key==32 and calibration_points is not None and state.position is not None:
                calibration_points.append(state.position)
            elif key==ord('s'):
                backup = settings.mapping.copy()
                editing,row = True,0
            elif key in (left,right):
                page = max(1,page+(-1 if key==left else 1))
            elif args.mouse and key==ord('h'):
                event.add('DOUBLE_CLAP')
            elif args.mouse and key in (ord('f'),ord('p')):
                gesture = 'FIST' if key==ord('f') else 'POINTER'
                simulated.symmetric_difference_update({gesture})
            elif key==ord('x'):
                renderer.ink[:] = 255
    finally:
        if tracker:
            tracker.close()
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__=='__main__':
    main()
