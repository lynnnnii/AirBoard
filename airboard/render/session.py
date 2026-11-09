"""Control tool, stroke, and page transitions in the standalone demo; does not implement gesture recognition or smoothing."""
from __future__ import annotations
from typing import Any
from .engine import AirBoardRenderer, Point, Tool
from .bridge import InkStyle, SmoothedInkBridge


class RenderSession:
    """Connect the team's smoothing module public update API to the renderer."""

    def __init__(self, board: AirBoardRenderer, smoother: Any, eraser_filter: Any) -> None:
        """Inject the handwriting smoother and eraser coordinate filter."""
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
        """Finalize the current stroke with tail trimming before page/tool changes or saving."""
        if self.effective == 'eraser':
            self.board.end_stroke()
        result = self.smoother.update(None, False, t)
        self.bridge.consume(result, self.style)
        self.board.end_stroke()
        self.eraser_filter.reset()
        self.cursor = None

    def select_tool(self, tool: Tool, t: float) -> None:
        """Finalize the current stroke before selecting a tool; a fresh pen-down is required."""
        self.flush(t)
        self.selected = tool
        self.effective = tool
        self.style = InkStyle(tool)
        self.lift_required = True

    def change_page(self, offset: int, t: float) -> bool:
        """Finalize strokes before changing pages; do not draw on the next page while the pen is still down."""
        self.flush(t)
        changed = self.board.change_page(offset)
        self.lift_required = True
        return changed

    def feed(self, point: Point | None, pen_down: bool, t: float, *,
             temporary_tool: Tool | None = None, eraser_active: bool = False) -> None:
        """Accept slide coordinates; fist/pointer tools are temporary and preserve the selected pen."""
        tool = temporary_tool or self.selected
        if tool != self.effective:
            self.flush(t)
            self.effective = tool
            self.style = InkStyle(tool)
        w, h = self.board.slide_size
        if point is not None and not (0 <= point[0] < w and 0 <= point[1] < h):
            point = None
        # Tracking loss alone does not unlock drawing; require an actual pen-up.
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
