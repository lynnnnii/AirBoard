"""OpenCV 이미지 위에 커서·팔레트·알림을 그린다."""
import cv2
from .controller import RenderPacket
from .settings import Settings


def bgr(color: str) -> tuple[int,int,int]:
    """#RRGGBB를 OpenCV BGR로 바꾼다."""
    return tuple(int(color[i:i+2],16) for i in (5,3,1))


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
            cv2.circle(image,p,15 if packet.tool=='eraser' else 7,(0,0,255) if packet.active else (255,100,0),2)
        cv2.putText(image,f'{packet.tool} / {packet.phase}  {notice}',(12,80),cv2.FONT_HERSHEY_SIMPLEX,.65,(40,40,180),2)
