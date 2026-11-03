"""OpenCV 이미지 위에 커서·팔레트·알림을 그린다."""
import math
import time
import cv2
import numpy as np
from .controller import RenderPacket
from .settings import Settings


def bgr(color: str) -> tuple[int,int,int]:
    """#RRGGBB를 OpenCV BGR로 바꾼다."""
    return tuple(int(color[i:i+2],16) for i in (5,3,1))


def draw_sparkle(image: np.ndarray, p: tuple[int,int], now: float) -> None:
    """레이저 포인터 커서: 빛 번짐 + 밝은 심 + 회전하며 깜빡이는 별빛. 커서 주변 영역만 계산한다."""
    r = 36
    h,w = image.shape[:2]
    x0,y0,x1,y1 = max(0,p[0]-r),max(0,p[1]-r),min(w,p[0]+r+1),min(h,p[1]+r+1)
    if x0>=x1 or y0>=y1:
        return
    roi = image[y0:y1,x0:x1]
    c = (p[0]-x0,p[1]-y0)
    pulse = .5+.5*math.sin(now*8)                     # 초당 약 1.3번 숨 쉬듯 밝아졌다 어두워짐
    # 빛 번짐: 흐린 원 마스크로 색을 섞어 흰 슬라이드 위에서도 보이게 한다.
    mask = np.zeros(roi.shape[:2],np.float32)
    cv2.circle(mask,c,round(10+5*pulse),1.,-1,cv2.LINE_AA)
    a = cv2.GaussianBlur(mask,(0,0),6)[...,None]*(.55+.25*pulse)
    roi[:] = np.rint(roi*(1-a)+np.array((40,40,255),np.float32)*a).astype(np.uint8)
    spin = now*2.5
    for k in range(8):                                # 긴 십자 빛 4개 + 짧은 대각선 빛 4개
        ang = spin+k*math.pi/4
        length = (20+10*pulse) if k%2==0 else (10+5*pulse)
        tip = (round(c[0]+length*math.cos(ang)),round(c[1]+length*math.sin(ang)))
        cv2.line(roi,c,tip,(0,90,255) if k%2==0 else (0,180,255),2 if k%2==0 else 1,cv2.LINE_AA)
    cv2.circle(roi,c,6,(30,30,230),-1,cv2.LINE_AA)
    cv2.circle(roi,c,3,(255,255,255),-1,cv2.LINE_AA)


class Overlay:
    """일정 시간 머무르기 팔레트. 상태는 UI 인스턴스 안에서만 유지."""
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.target,self.since,self.done = None,0.,False
        self.cells = []
        for i,c in enumerate(settings.ui['colors']):
            self.cells.append((10+i*55,8,58+i*55,46,'color',c))
        offset = 20+len(settings.ui['colors'])*55
        for i,w in enumerate(settings.ui['widths']):
            self.cells.append((offset+i*65,8,offset+58+i*65,46,'width',w))

    def dwell(self, position: tuple | None, now: float) -> tuple | None:
        """팔레트 진입을 감지하고 시간 충족 시 한 번만 선택을 반환."""
        target = next(((k,v) for a,b,c,d,k,v in self.cells if position is not None and a<=position[0]<c and b<=position[1]<d),None)
        if target != self.target:
            self.target,self.since,self.done = target,now,False
        if target and now-self.since>=self.settings.ui['dwell_seconds'] and not self.done:
            self.done = True
            return target
        return None

    def draw(self, image: object, packet: RenderPacket, notice: str = '') -> None:
        """전달받은 이미지를 수정; 창 생성/입력 처리는 호출자가 담당."""
        for a,b,c,d,k,v in self.cells:
            cv2.rectangle(image,(a,b),(c,d),bgr(v) if k=='color' else (35,35,35),-1)
            if k=='width':
                text = str(v)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .95, 2)
                cv2.putText(image,text,(a+(c-a-tw)//2,b+(d-b+th)//2),
                            cv2.FONT_HERSHEY_SIMPLEX,.95,(255,255,255),2,cv2.LINE_AA)
                if v == packet.width:
                    cv2.rectangle(image,(a,b),(c,d),(255,255,255),2)
            if (k,v)==self.target:
                cv2.rectangle(image,(a,b),(c,d),(0,220,255),2)
        if packet.position:
            p = tuple(round(v) for v in packet.position)
            if packet.tool=='pointer':
                draw_sparkle(image,p,time.monotonic())
            else:
                cv2.circle(image,p,round(packet.width/2) if packet.tool=='eraser' else 7,(0,0,255) if packet.active else (255,100,0),2)
