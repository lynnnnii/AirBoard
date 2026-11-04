"""인식 결과를 받아 렌더러 독립 패킷을 생성한다. 웹캠/창 생성 없음."""
import math
from dataclasses import dataclass
from .settings import Settings
from airboard.calibration import Calibration


@dataclass(frozen=True)
class InputState:
    """정규화 카메라 좌표와 프레임별 제스처 이름. 이벤트는 발생 프레임만 전달."""
    position: tuple[float,float] | None
    gestures: frozenset[str] = frozenset()
    navigation_allowed: bool = True


@dataclass(frozen=True)
class RenderPacket:
    """up은 좌표가 없어도 기존 획 종료. tool은 현재 선택 도구."""
    position: tuple[float,float] | None
    tool: str
    active: bool
    phase: str
    color: str
    width: int
    opacity: float
    commands: tuple[str,...]


def from_gesture_result(result: object, frame_size: tuple[int,int]) -> InputState:
    """김현보 GestureResult의 공개 필드만 사용; 픽셀→정규화 어댑터."""
    w,h = frame_size
    if w<2 or h<2:
        raise ValueError('프레임 크기가 너무 작습니다.')
    gestures = set(result.events)
    if result.pen_down:
        gestures.add('PINCH')
    if result.pose in ('FIST','POINTER','TWO_FINGERS'):
        gestures.add(result.pose)
    p = result.pointer_point if result.pose in ('FIST','POINTER','TWO_FINGERS') else result.pen_point
    # 손 소실 프레임에서도 DOUBLE_CLAP 같은 이미 판정된 이벤트는 보존한다.
    return InputState(None if p is None else (float(p[0])/(w-1),float(p[1])/(h-1)),frozenset(gestures),
                      len(result.hands)==1 and result.pose not in ('PINCH','FIST','POINTER','TWO_FINGERS') and not result.pen_down)


class UIController:
    """도구/획 경계와 우선순위만 관리; 인식·떨림 보정·필기 저장은 외부 담당."""

    def __init__(self, calibration: Calibration, settings: Settings) -> None:
        self.calibration,self.settings = calibration,settings
        self.tool,self.base_tool = 'pen','pen'
        self.color,self.width = settings.ui['colors'][0],settings.ui['widths'][min(1,len(settings.ui['widths'])-1)]
        self.active,self.blocked = False,False
        self.temporary = None
        self.candidate,self.count = None,0
        self.notification,self.notice_until = '',0.
        self._toggle_held = False
        self._page_blocked_until = float('-inf')
        self._pending_page = None
        self.navigation_status = 'ready'
        self.writing_scale = float(settings.ui.get('writing_scale',1.))
        self._anchor = None          # 글씨 축소 기준점 (단어의 첫 획 시작점, 슬라이드 픽셀)
        self._last_ink = float('-inf')
        self._draw_seen = float('-inf')
        self._draw_by_pinch = False
        self._was_holding = False
        self._last_pos = None        # 마지막으로 쓴 슬라이드 좌표 (흔들림 유예 중 고정 위치)
        self._clap_page_until = float('-inf')   # 박수 페이지 이동 재입력 대기 (두 손 프레임은 일반 대기를 계속 늘리므로 따로 둔다)

    def update(self, state: InputState, now: float, suppressed: bool = False, navigation_suppressed: bool | None = None) -> RenderPacket:
        """매 프레임 호출. 팔레트·설정·캘리브레이션 중에는 suppressed=True."""
        mapped = {self.settings.mapping.get(g,'none') for g in state.gestures}
        nav_suppressed = suppressed if navigation_suppressed is None else navigation_suppressed
        actions = (mapped if not suppressed else
                   mapped & {'toggle_highlighter'} if not nav_suppressed else set())
        # 'stop'(기본: 검지·중지 두 손가락)은 쓰는 중이라도 그 프레임에 바로 펜을 든다.
        # 아래 흔들림 유예로 이어 쓰지 않도록 마지막 쓰기 시각도 지운다.
        if 'stop' in actions:
            actions = actions - {'draw'}
            self._draw_seen = float('-inf')
        # 검지처럼 핀치가 아닌 제스처로 쓰는 중이면, 손 모양 판정이 잠깐(draw_grace_seconds) 흔들려도 획을 이어 간다.
        # 그동안은 잉크를 늘리지 않고 마지막 위치에 멈춰 둔다. 핀치의 흔들림은 인식기가 따로 처리한다.
        if 'draw' in actions:
            self._draw_seen = now
            self._draw_by_pinch = self.settings.mapping.get('PINCH') == 'draw' and 'PINCH' in state.gestures
        holding = ('draw' not in actions and self.active and self.tool in ('pen','highlighter')
                   and not self._draw_by_pinch and state.position is not None and not suppressed
                   and not mapped & {'eraser','pointer'}
                   and now-self._draw_seen <= self.settings.ui.get('draw_grace_seconds',.2))
        if holding:
            actions = actions | {'draw'}
        previous_tool = self.tool
        wanted = 'eraser' if 'eraser' in actions else 'pointer' if 'pointer' in actions else None
        if wanted == self.candidate:
            self.count += 1
        else:
            self.candidate,self.count = wanted,1
        if wanted is None:
            if self.temporary:
                self.blocked = True
            self.temporary = None
        elif self.count >= (self.settings.ui.get('eraser_hold_frames',1) if wanted=='eraser' else self.settings.ui['tool_hold_frames']):
            self.temporary = wanted
        # 지우개·포인터를 놓은 뒤의 쓰기 잠금은 핀치로 쓸 때만 유지한다(핀치를 잡은 채 주먹을 풀면 바로 써지지 않게).
        # 검지처럼 자세 자체가 쓰기인 제스처는 계속 그 자세를 하고 있으므로, 잠그면 손을 다 펴기 전까지 못 쓴다.
        pinch_draw = self.settings.mapping.get('PINCH') == 'draw' and 'PINCH' in state.gestures
        if 'draw' not in actions or not pinch_draw:
            self.blocked = False
        # 일회성 이벤트는 입력 계약에 따라 발생 프레임에만 실행한다.
        toggle = 'toggle_highlighter' in actions
        if toggle and not self._toggle_held and not wanted and 'draw' not in actions:
            self.base_tool = 'pen' if self.base_tool=='highlighter' else 'highlighter'
        self._toggle_held = toggle
        self.tool = self.temporary or self.base_tool
        if self.tool != previous_tool:
            self.notification,self.notice_until = self.tool,now+self.settings.ui['notification_seconds']
        pos = None if suppressed else self.calibration.transform(state.position)
        active = pos is not None and wanted == self.temporary and not self.blocked and (self.tool=='eraser' or ('draw' in actions and self.tool in ('pen','highlighter')))
        if self.active and self.tool != previous_tool:
            active = False
        # 글씨 축소(펜만): 손 이동을 기준점에서 writing_scale배로 줄여 슬라이드에 작게 쓴다.
        # 획 사이 쉬는 시간이 writing_anchor_seconds 안이면 같은 기준점을 써서 한 단어의 획 배치도 함께 줄어든다.
        # 그동안은 커서도 줄어든 위치에 보여 다음 획이 어디서 시작될지 알 수 있고, 그 뒤에는
        # writing_return_seconds 동안 실제 손 위치로 미끄러지듯 돌아와 커서가 튀지 않는다.
        # 형광펜은 슬라이드의 글자를 그대로 덮어야 하므로 줄이지 않는다.
        anchor_s = self.settings.ui.get('writing_anchor_seconds',1.)
        if self.tool == 'pen' and pos is not None:
            if active and not self.active and now-self._last_ink > anchor_s:
                self._anchor = None          # 쉬었다가 새로 쓰기 시작: 지금 손 위치가 새 기준점
            if active and self._anchor is None:
                self._anchor = pos
            if self._anchor is not None:
                scaled = tuple(a+(p-a)*self.writing_scale for a,p in zip(self._anchor,pos))
                idle = 0. if active else now-self._last_ink-anchor_s
                k = min(1.,max(0.,idle/self.settings.ui.get('writing_return_seconds',.4)))
                pos = tuple(s+(p-s)*k for s,p in zip(scaled,pos))
                if k >= 1:
                    self._anchor = None
            if active:
                self._last_ink = now
        elif self.tool != 'pen':
            self._anchor = None
        if holding:
            pos = self._last_pos
        elif (active and self.active and self._was_holding and pos is not None and self._last_pos is not None
              and math.dist(pos,self._last_pos) > self.settings.ui.get('draw_resume_break',40)):
            active = False       # 흔들리는 동안 손이 멀리 움직였다면 이어 긋지 않고 다음 프레임에 새 획으로 시작
        self._was_holding = holding
        if active:
            self._last_pos = pos
        phase ='move' if active and self.active else 'down' if active else 'up' if self.active else 'hover'
        # 손 제스처로 페이지를 넘기는 기능은 오작동 방지를 위해 비활성화한다.
        commands = ()
        self.navigation_status = 'disabled'
        self.active = bool(active)
        # 형광펜은 팔레트 색·굵기와 무관하게 고정 색(노랑)과 고정 굵기를 쓴다.
        if self.tool == 'highlighter':
            return RenderPacket(pos,self.tool,bool(active),phase,self.settings.ui.get('highlighter_color','#FFFF00'),
                                self.settings.ui.get('highlighter_width',16),self.settings.ui['highlighter_opacity'],commands)
        # 지우개 width는 지우개 원의 지름이다 (닿은 획을 통째로 지우는 판정 범위).
        width = 2*self.settings.ui.get('eraser_radius',28) if self.tool=='eraser' else self.width
        return RenderPacket(pos,self.tool,bool(active),phase,self.color,width,1.,commands)
