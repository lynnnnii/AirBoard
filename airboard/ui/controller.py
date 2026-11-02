"""인식 결과를 받아 렌더러 독립 패킷을 생성한다. 웹캠/창 생성 없음."""
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
    if result.pose in ('FIST','POINTER'):
        gestures.add(result.pose)
    p = result.pointer_point if result.pose in ('FIST','POINTER') else result.pen_point
    # 손 소실 프레임에서도 DOUBLE_CLAP 같은 이미 판정된 이벤트는 보존한다.
    return InputState(None if p is None else (float(p[0])/(w-1),float(p[1])/(h-1)),frozenset(gestures),
                      len(result.hands)==1 and result.pose not in ('PINCH','FIST','POINTER') and not result.pen_down)


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

    def update(self, state: InputState, now: float, suppressed: bool = False, navigation_suppressed: bool | None = None) -> RenderPacket:
        """매 프레임 호출. 팔레트·설정·캘리브레이션 중에는 suppressed=True."""
        mapped = {self.settings.mapping.get(g,'none') for g in state.gestures}
        nav_suppressed = suppressed if navigation_suppressed is None else navigation_suppressed
        actions = (mapped if not suppressed else
                   mapped & {'toggle_highlighter'} if not nav_suppressed else set())
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
        elif self.count >= (self.settings.ui.get('eraser_hold_frames', 1) if wanted == 'eraser' else self.settings.ui['tool_hold_frames']):
            self.temporary = wanted
        if 'draw' not in actions:
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
        phase = 'move' if active and self.active else 'down' if active else 'up' if self.active else 'hover'
        quiet = self.settings.ui.get('navigation_quiet_seconds', .15)
        unsafe = (nav_suppressed or active or self.active or wanted or toggle
                  or not state.navigation_allowed or 'draw' in mapped)
        page_actions = mapped & {'page_next','page_prev'}
        commands = ()
        if unsafe:
            self._page_blocked_until = now + quiet
            self._pending_page = None
            self.navigation_status = ('settings/palette' if nav_suppressed else
                                      'writing/tool' if active or self.active or wanted or toggle or 'draw' in mapped
                                      else 'need one open hand')
        else:
            if len(page_actions)>1:
                self._pending_page = None
            elif len(page_actions)==1:
                # 모드 해제 직후 단발 이벤트가 발생하면 짧게 보관한다.
                if self._page_blocked_until-now <= quiet:
                    self._pending_page = (next(iter(page_actions)), now + .35)
            if self._pending_page is not None and now>self._pending_page[1]:
                self._pending_page = None
            if self._pending_page is not None and now>=self._page_blocked_until:
                commands = (self._pending_page[0],)
                self._pending_page = None
                self._page_blocked_until = now + self.settings.ui.get('navigation_cooldown_seconds', .8)
            self.navigation_status = ('page moved' if commands else 'waiting' if self._pending_page
                                      else 'cooldown' if now<self._page_blocked_until else 'ready')
        self.active = bool(active)
        color = self.settings.ui.get('highlighter_color', '#FFFF00') if self.tool == 'highlighter' else self.color
        width = self.settings.ui.get('highlighter_width', 24) if self.tool == 'highlighter' else self.width
        return RenderPacket(pos,self.tool,bool(active),phase,color,width,
                            self.settings.ui['highlighter_opacity'] if self.tool=='highlighter' else 1.,commands)
