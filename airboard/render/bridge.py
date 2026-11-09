"""Connect the team's StrokeSmoother output to the renderer without accessing smoothing internals."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .engine import AirBoardRenderer, Point, Tool


class SmoothFrame(Protocol):
    """Consume the team's SmoothResult interface without modifying teammates' modules."""
    point: Point | None
    stroke: list[Point] | None
    finished: list[Point] | None


@dataclass(frozen=True)
class InkStyle:
    """Tool, color, and width to lock when a stroke begins."""
    tool: Tool = 'pen'
    color: str | None = None
    width: float | None = None


class SmoothedInkBridge:
    """Preview in-progress strokes and persist only completed strokes."""

    def __init__(self, renderer: AirBoardRenderer) -> None:
        """Attach the renderer and initialize stroke style tracking."""
        self.renderer = renderer
        self._style: InkStyle | None = None

    def consume(self, result: SmoothFrame, style: InkStyle) -> None:
        """Call once per frame, including the frame in which the stroke finishes.

        Before changing tools or pages, forward smoother.update(None, False, t).
        Remember the style even when initial skipped frames produce stroke=[].
        """
        if result.finished is not None:
            old = self._style or style
            self.renderer.commit_stroke(list(result.finished), old.tool,
                                        color=old.color, width=old.width)
            self._style = None
        if result.stroke is not None:
            if self._style is None:
                if style.tool not in ('pen', 'highlighter'):
                    raise ValueError('Only pass pen/highlighter strokes to SmoothedInkBridge.')
                self._style = style
            old = self._style
            self.renderer.set_preview(list(result.stroke), old.tool,
                                      color=old.color, width=old.width)
        else:
            self.renderer.discard_preview()
            self._style = None
