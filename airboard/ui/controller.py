"""인식 결과를 받아 렌더러 독립 패킷을 생성한다. 웹캠/창 생성 없음."""
from dataclasses import dataclass
from .settings import Settings
from airboard.calibration import Calibration


@dataclass(frozen=True)
class InputState:
    """정규화 카메라 좌표와 프레임별 제스처 이름. 이벤트는 발생 프레임만 전달."""
    position: tuple[float,float] | None
    gestures: frozenset[str] = frozenset()


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
    return InputState(None if p is None else (float(p[0])/(w-1),float(p[1])/(h-1)),frozenset(gestures))


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

    def update(self, state: InputState, now: float, suppressed: bool = False) -> RenderPacket:
        """매 프레임 호출. 팔레트·설정·캘리브레이션 중에는 suppressed=True."""
        actions = set() if suppressed else {self.settings.mapping.get(g,'none') for g in state.gestures}
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
        elif self.count >= self.settings.ui['tool_hold_frames']:
            self.temporary = wanted
        if 'draw' not in actions:
            self.blocked = False
        # 일회성 이벤트는 입력 계약에 따라 발생 프레임에만 실행한다.
        toggle = 'toggle_highlighter' in actions
        if toggle and not self._toggle_held and not wanted and not self.active:
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
        commands = tuple(a for a in ('page_next','page_prev') if a in actions) if not active and not self.active and not wanted else ()
        self.active = bool(active)
        return RenderPacket(pos,self.tool,bool(active),phase,self.color,self.width,
                            self.settings.ui['highlighter_opacity'] if self.tool=='highlighter' else 1.,commands)
