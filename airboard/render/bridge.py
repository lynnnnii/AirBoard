"""팀 StrokeSmoother의 결과와 렌더러를 연결한다. smoothing 내부 상태는 참조하지 않는다."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .engine import AirBoardRenderer, Point, Tool


class SmoothFrame(Protocol):
    """팀 SmoothResult를 구조적으로 받으므로 다른 담당자 파일을 수정하지 않는다."""
    point: Point | None
    stroke: list[Point] | None
    finished: list[Point] | None


@dataclass(frozen=True)
class InkStyle:
    """획 시작 시 고정할 도구/색/굵기."""
    tool: Tool = 'pen'
    color: str | None = None
    width: float | None = None


class SmoothedInkBridge:
    """진행 획은 미리보기, finished만 영구 필기로 저장한다."""

    def __init__(self, renderer: AirBoardRenderer) -> None:
        """렌더러를 연결하고 획 스타일 기록을 초기화한다."""
        self.renderer = renderer
        self._style: InkStyle | None = None

    def consume(self, result: SmoothFrame, style: InkStyle) -> None:
        """매 프레임 한 번 호출한다. 완료 프레임도 반드시 전달한다.

        도구/페이지 변경 전에 smoother.update(None, False, t) 결과를 먼저 전달한다.
        시작 skip으로 stroke=[]인 프레임부터 스타일을 기억한다.
        """
        if result.finished is not None:
            old = self._style or style
            self.renderer.commit_stroke(list(result.finished), old.tool,
                                        color=old.color, width=old.width)
            self._style = None
        if result.stroke is not None:
            if self._style is None:
                if style.tool not in ('pen', 'highlighter'):
                    raise ValueError('펜/형광펜 획만 SmoothedInkBridge에 전달하세요.')
                self._style = style
            old = self._style
            self.renderer.set_preview(list(result.stroke), old.tool,
                                      color=old.color, width=old.width)
        else:
            self.renderer.discard_preview()
            self._style = None
