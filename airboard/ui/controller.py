"""Create renderer-independent packets from recognition results; does not manage webcams or windows."""
import math
from dataclasses import dataclass
from .settings import Settings
from airboard.calibration import Calibration


@dataclass(frozen=True)
class InputState:
    """Normalized camera coordinates and per-frame gesture names; events are emitted only on their trigger frame."""
    position: tuple[float,float] | None
    gestures: frozenset[str] = frozenset()
    navigation_allowed: bool = True


@dataclass(frozen=True)
class RenderPacket:
    """An up event ends the current stroke even without coordinates; tool is the currently selected tool."""
    position: tuple[float,float] | None
    tool: str
    active: bool
    phase: str
    color: str
    width: int
    opacity: float
    commands: tuple[str,...]


def from_gesture_result(result: object, frame_size: tuple[int,int]) -> InputState:
    """Adapt pixel to normalized coordinates using only the public GestureResult fields."""
    w,h = frame_size
    if w<2 or h<2:
        raise ValueError('Frame dimensions are too small.')
    gestures = set(result.events)
    if result.pen_down:
        gestures.add('PINCH')
    if result.pose in ('FIST','POINTER','TWO_FINGERS'):
        gestures.add(result.pose)
    p = result.pointer_point if result.pose in ('FIST','POINTER','TWO_FINGERS') else result.pen_point
    # Preserve completed events such as DOUBLE_CLAP even on frames where the hand is lost.
    return InputState(None if p is None else (float(p[0])/(w-1),float(p[1])/(h-1)),frozenset(gestures),
                      len(result.hands)==1 and result.pose not in ('PINCH','FIST','POINTER','TWO_FINGERS') and not result.pen_down)


class UIController:
    """Manage tool/stroke boundaries and priorities only; recognition, smoothing, and persistence are handled elsewhere."""

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
        self._anchor = None          # Anchor for writing-scale reduction (first stroke of a word, in slide pixels)
        self._last_ink = float('-inf')
        self._draw_seen = float('-inf')
        self._draw_by_pinch = False
        self._was_holding = False
        self._last_pos = None        # Last drawn position in slide coordinates (held during gesture grace period)
        self._clap_page_until = float('-inf')   # Clap/page-navigation cooldown (separate from the standard two-hand cooldown)

    def update(self, state: InputState, now: float, suppressed: bool = False, navigation_suppressed: bool | None = None) -> RenderPacket:
        """Call on each frame; pass suppressed=True during palette, settings, or calibration operations."""
        mapped = {self.settings.mapping.get(g,'none') for g in state.gestures}
        nav_suppressed = suppressed if navigation_suppressed is None else navigation_suppressed
        actions = (mapped if not suppressed else
                   mapped & {'toggle_highlighter'} if not nav_suppressed else set())
        history_action = next(iter(mapped & {'undo','redo'}), None)
        if history_action:
            actions = set()
            self._draw_seen = float('-inf')
        # The stop gesture (default: index and middle fingers) lifts the pen immediately, even during drawing.
        # Clear the last drawing time so the grace period cannot reconnect the stroke.
        if 'stop' in actions:
            actions = actions - {'draw'}
            self._draw_seen = float('-inf')
        # For non-pinch drawing gestures such as the index finger, allow brief classification instability (draw_grace_seconds).
        # During this grace period, hold the previous point without adding ink. The recognizer handles pinch jitter separately.
        if 'draw' in actions:
            self._draw_seen = now
            self._draw_by_pinch = self.settings.mapping.get('PINCH') == 'draw' and 'PINCH' in state.gestures
        holding = ('draw' not in actions and self.active and self.tool in ('pen','highlighter')
                   and not history_action
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
        # Keep the post-eraser/pointer drawing lock only for pinch drawing, preventing a held pinch from immediately drawing again.
        # For pose-based drawing (e.g., index pointing), such a lock would incorrectly require fully opening the hand.
        pinch_draw = self.settings.mapping.get('PINCH') == 'draw' and 'PINCH' in state.gestures
        if 'draw' not in actions or not pinch_draw:
            self.blocked = False
        # Trigger one-shot events only in their originating frame.
        toggle = 'toggle_highlighter' in actions
        if toggle and not self._toggle_held and not wanted and 'draw' not in actions:
            self.base_tool = 'pen' if self.base_tool=='highlighter' else 'highlighter'
        self._toggle_held = toggle
        self.tool = self.temporary or self.base_tool
        if self.tool != previous_tool:
            self.notification,self.notice_until = self.tool,now+self.settings.ui['notification_seconds']
        pos = None if suppressed else self.calibration.transform(state.position)
        active = not history_action and pos is not None and wanted == self.temporary and not self.blocked and (self.tool=='eraser' or ('draw' in actions and self.tool in ('pen','highlighter')))
        if self.active and self.tool != previous_tool:
            active = False
        # Pen-only writing scale: reduce hand movement relative to the anchor by writing_scale.
        # Reuse the anchor between strokes within writing_anchor_seconds to preserve scaled spacing within a word.
        # Display the cursor at the scaled position so users can see where the next stroke will start, then
        # smoothly return to the actual hand position over writing_return_seconds to avoid a cursor jump.
        # Do not scale the highlighter because it must align with the underlying slide text.
        anchor_s = self.settings.ui.get('writing_anchor_seconds',1.)
        if self.tool == 'pen' and pos is not None:
            if active and not self.active and now-self._last_ink > anchor_s:
                self._anchor = None          # Starting after a pause: use the current hand position as the new anchor
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
            active = False       # If the hand moved too far during the grace period, start a new stroke on the next frame
        self._was_holding = holding
        if active:
            self._last_pos = pos
        phase ='move' if active and self.active else 'down' if active else 'up' if self.active else 'hover'
        # Disable gesture-based page navigation to prevent accidental page changes.
        commands = ()
        if history_action:
            commands = (history_action,)
            self.navigation_status = 'history changed'
        else:
            self.navigation_status = 'disabled'
        self.active = bool(active)
        # Use a fixed yellow color and width for the highlighter, independent of palette settings.
        if self.tool == 'highlighter':
            return RenderPacket(pos,self.tool,bool(active),phase,self.settings.ui.get('highlighter_color','#FFFF00'),
                                self.settings.ui.get('highlighter_width',16),self.settings.ui['highlighter_opacity'],commands)
        # Eraser width is its circular hit-test diameter; touched strokes are erased in full.
        width = 2*self.settings.ui.get('eraser_radius',28) if self.tool=='eraser' else self.width
        return RenderPacket(pos,self.tool,bool(active),phase,self.color,width,1.,commands)
