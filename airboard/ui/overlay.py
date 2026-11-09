"""Draw cursors, palettes, and notifications on OpenCV images."""
import math
import time
import cv2
import numpy as np
from .controller import RenderPacket
from .settings import Settings


def bgr(color: str) -> tuple[int,int,int]:
    """Convert #RRGGBB to OpenCV BGR."""
    return tuple(int(color[i:i+2],16) for i in (5,3,1))


def draw_sparkle(image: np.ndarray, p: tuple[int,int], now: float) -> None:
    """Laser cursor with glow, bright center, and rotating twinkles; render only near the cursor."""
    r = 36
    h,w = image.shape[:2]
    x0,y0,x1,y1 = max(0,p[0]-r),max(0,p[1]-r),min(w,p[0]+r+1),min(h,p[1]+r+1)
    if x0>=x1 or y0>=y1:
        return
    roi = image[y0:y1,x0:x1]
    c = (p[0]-x0,p[1]-y0)
    pulse = .5+.5*math.sin(now*8)                     # Pulse in brightness approximately 1.3 times per second
    # Glow: blend a blurred circular mask so the pointer remains visible on white slides.
    mask = np.zeros(roi.shape[:2],np.float32)
    cv2.circle(mask,c,round(10+5*pulse),1.,-1,cv2.LINE_AA)
    a = cv2.GaussianBlur(mask,(0,0),6)[...,None]*(.55+.25*pulse)
    roi[:] = np.rint(roi*(1-a)+np.array((40,40,255),np.float32)*a).astype(np.uint8)
    spin = now*2.5
    for k in range(8):                                # Four long cardinal rays and four short diagonal rays
        ang = spin+k*math.pi/4
        length = (20+10*pulse) if k%2==0 else (10+5*pulse)
        tip = (round(c[0]+length*math.cos(ang)),round(c[1]+length*math.sin(ang)))
        cv2.line(roi,c,tip,(0,90,255) if k%2==0 else (0,180,255),2 if k%2==0 else 1,cv2.LINE_AA)
    cv2.circle(roi,c,6,(30,30,230),-1,cv2.LINE_AA)
    cv2.circle(roi,c,3,(255,255,255),-1,cv2.LINE_AA)


class Overlay:
    """Dwell-based palette; keep state within the UI instance."""
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
        """Detect palette entry and return a single selection after the dwell threshold."""
        target = next(((k,v) for a,b,c,d,k,v in self.cells if position is not None and a<=position[0]<c and b<=position[1]<d),None)
        if target != self.target:
            self.target,self.since,self.done = target,now,False
        if target and now-self.since>=self.settings.ui['dwell_seconds'] and not self.done:
            self.done = True
            return target
        return None

    def draw(self, image: object, packet: RenderPacket, notice: str = '') -> None:
        """Modify the provided image; the caller handles window creation and input."""
        for a,b,c,d,k,v in self.cells:
            if k == 'width':
                # Draw only the width-selector cell at 4x resolution and downsample for smooth edges.
                scale = 4
                cell_w,cell_h = c-a+1,d-b+1
                tile = np.full((cell_h*scale,cell_w*scale,3),(35,35,35),np.uint8)
                widths = self.settings.ui['widths']
                size_index = widths.index(v)
                sample_width = 2 + 2 * size_index
                y = (cell_h//2)*scale
                curve = np.array([
                    (8*scale,y+5*scale), (14*scale,y+5*scale), (17*scale,y-4*scale),
                    (21*scale,y-6*scale), (25*scale,y+scale), (29*scale,y+5*scale),
                    (34*scale,y+3*scale), (40*scale,y-4*scale), ((cell_w-8)*scale,y-8*scale)
                ],dtype=np.int32)
                thickness = sample_width*scale
                cv2.polylines(tile,[curve],False,(255,255,255),thickness,cv2.LINE_AA)
                radius = max(1,thickness//2)
                cv2.circle(tile,tuple(curve[0]),radius,(255,255,255),-1,cv2.LINE_AA)
                cv2.circle(tile,tuple(curve[-1]),radius,(255,255,255),-1,cv2.LINE_AA)
                if v == packet.width:
                    cv2.rectangle(tile,(0,0),(cell_w*scale-1,cell_h*scale-1),(255,255,255),2*scale,cv2.LINE_AA)
                if (k,v)==self.target:
                    cv2.rectangle(tile,(0,0),(cell_w*scale-1,cell_h*scale-1),(0,220,255),2*scale,cv2.LINE_AA)
                image[b:d+1,a:c+1] = cv2.resize(tile,(cell_w,cell_h),interpolation=cv2.INTER_AREA)
                continue
            cv2.rectangle(image,(a,b),(c,d),bgr(v) if k=='color' else (35,35,35),-1)
            if (k,v)==self.target:
                cv2.rectangle(image,(a,b),(c,d),(0,220,255),2)
        if packet.position:
            p = tuple(round(v) for v in packet.position)
            if packet.tool=='pointer':
                draw_sparkle(image,p,time.monotonic())
            else:
                cv2.circle(image,p,round(packet.width/2) if packet.tool=='eraser' else 7,(0,0,255) if packet.active else (255,100,0),2)
