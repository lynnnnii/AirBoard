"""단독 데모의 도구/획/페이지 전환 제어. 제스처 판정이나 보정 알고리즘은 구현하지 않는다."""
from __future__ import annotations
from typing import Any
from .engine import AirBoardRenderer, Point, Tool
from .bridge import InkStyle, SmoothedInkBridge


class RenderSession:
    """팀 보정 모듈의 공개 update API를 렌더러와 연결한다."""

    def __init__(self, board: AirBoardRenderer, smoother: Any, eraser_filter: Any) -> None:
        """필기 보정기와 지우개 좌표 필터를 외부에서 주입한다."""
        self.board = board
        self.smoother = smoother
        self.eraser_filter = eraser_filter
        self.bridge = SmoothedInkBridge(board)
        self.selected: Tool = 'pen'
        self.effective: Tool = 'pen'
        self.style = InkStyle('pen')
        self.cursor: Point | None = None
        self.lift_required = False

    def flush(self, t: float) -> None:
        """페이지/도구/저장 전 현재 획을 꼬리 제거 후 확정한다."""
        if self.effective == 'eraser':
            self.board.end_stroke()
        result = self.smoother.update(None, False, t)
        self.bridge.consume(result, self.style)
        self.board.end_stroke()
        self.eraser_filter.reset()
        self.cursor = None

    def select_tool(self, tool: Tool, t: float) -> None:
        """현재 획 확정 후 새 도구를 선택한다. 다시 펜을 들어야 시작한다."""
        self.flush(t)
        self.selected = tool
        self.effective = tool
        self.style = InkStyle(tool)
        self.lift_required = True

    def change_page(self, offset: int, t: float) -> bool:
        """이전 페이지 획을 확정하고 이동한다. 손이 눌린 채 새 페이지에 그리지 않는다."""
        self.flush(t)
        changed = self.board.change_page(offset)
        self.lift_required = True
        return changed

    def feed(self, point: Point | None, pen_down: bool, t: float, *,
             temporary_tool: Tool | None = None, eraser_active: bool = False) -> None:
        """슬라이드 좌표 입력. 주먹/포인터는 temporary_tool로 기존 펜 선택을 보존한다."""
        tool = temporary_tool or self.selected
        if tool != self.effective:
            self.flush(t)
            self.effective = tool
            self.style = InkStyle(tool)
        w, h = self.board.slide_size
        if point is not None and not (0 <= point[0] < w and 0 <= point[1] < h):
            point = None
        # 추적 소실만으로는 재필기를 허용하지 않고 실제 pen-up을 확인한다.
        if point is not None and not pen_down and not eraser_active:
            self.lift_required = False
        blocked = self.lift_required
        if tool == 'eraser':
            self.cursor = None if point is None else tuple(float(v) for v in self.eraser_filter(point, t))
            if point is None:
                self.eraser_filter.reset()
            self.board.update(self.cursor, 'eraser', False, eraser_active=eraser_active and not blocked)
        elif tool == 'pointer':
            self.board.end_stroke()
            self.cursor = point
        else:
            result = self.smoother.update(point, pen_down and not blocked, t)
            self.bridge.consume(result, self.style)
            self.cursor = result.point
