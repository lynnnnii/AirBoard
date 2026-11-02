"""
떨림 보정 + 획 꼬리 제거 (담당: Lee Subin)

필기 좌표와 pen_down을 매 프레임 넣으면 보정된 커서 좌표와 획을 돌려준다.
좌표 단위는 상관없다 (카메라 픽셀이든 슬라이드 픽셀이든 2D 점이면 됨).

  smoother = StrokeSmoother.from_config(load_config())   # config/default.json의 "smoothing"
  s = smoother.update(point, pen_down, t)
  - s.point    : 보정된 커서 좌표 (손이 없으면 None) -> 커서 그리기
  - s.stroke   : 지금 쓰고 있는 획의 점 목록 (안 쓰는 중이면 None) -> 실시간 렌더링
  - s.finished : 이번 프레임에 끝난 획 (꼬리 제거 완료, 없으면 None) -> 필기 레이어에 확정

꼬리 제거
  - 시작 꼬리: 펜을 내린 직후 start_skip 프레임을 버린다.
  - 끝 꼬리: 핀치를 떼는 동안 손끝이 먼저 움직이고, 비율이 pinch_off를 넘고 유지 프레임이 지나야
    pen_down이 False가 된다. 그래서 끝 꼬리는 상태가 바뀌기 '전에' 이미 획에 들어가 있으므로,
    펜을 든 순간 마지막 end_trim개 점을 거슬러 지운다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

from airboard.smoothing.filters import EMAFilter, NoFilter, OneEuroFilter, PointFilter

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "default.json")

DEFAULTS: dict[str, Any] = {
    "method": "one_euro",        # "none" | "ema" | "one_euro"
    "ema_alpha": 0.3,
    "one_euro_min_cutoff": 1.0,
    "one_euro_beta": 0.01,
    "one_euro_d_cutoff": 1.0,
    "start_skip": 2,
    "end_trim": 3,
}

Point = tuple[float, float]


def load_config(path: str = DEFAULT_CONFIG) -> dict[str, Any]:
    """config/default.json의 "smoothing" 항목을 기본값 위에 덮어써서 돌려준다."""
    with open(path, encoding="utf-8") as f:
        section = json.load(f).get("smoothing", {})
    return {**DEFAULTS, **section}


def make_filter(method: str, cfg: dict[str, Any] | None = None) -> PointFilter:
    """보정 방식 이름으로 필터를 만든다."""
    c = {**DEFAULTS, **(cfg or {})}
    if method == "none":
        return NoFilter()
    if method == "ema":
        return EMAFilter(c["ema_alpha"])
    if method == "one_euro":
        return OneEuroFilter(c["one_euro_min_cutoff"], c["one_euro_beta"], c["one_euro_d_cutoff"])
    raise ValueError(f"알 수 없는 보정 방식: {method}")


@dataclass
class SmoothResult:
    """한 프레임의 보정 결과."""

    point: Point | None = None
    stroke: list[Point] | None = None
    finished: list[Point] | None = None


class StrokeSmoother:
    """필터로 좌표를 보정하고, pen_down 변화에 맞춰 획을 나누고 꼬리를 없앤다."""

    def __init__(self, point_filter: PointFilter | None = None,
                 start_skip: int = DEFAULTS["start_skip"], end_trim: int = DEFAULTS["end_trim"]):
        self.filter = OneEuroFilter() if point_filter is None else point_filter
        self.start_skip = start_skip
        self.end_trim = end_trim
        self.stroke: list[Point] | None = None
        self.skip_left = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any], method: str | None = None) -> StrokeSmoother:
        """설정값으로 만든다. method를 주면 설정의 method 대신 쓴다 (비교용)."""
        c = {**DEFAULTS, **cfg}
        return cls(make_filter(method or c["method"], c), c["start_skip"], c["end_trim"])

    def _finish(self) -> list[Point] | None:
        stroke, self.stroke = self.stroke, None
        if self.end_trim > 0 and len(stroke) > self.end_trim + 1:
            stroke = stroke[:-self.end_trim]
        return stroke or None

    def update(self, point: Sequence[float] | None, pen_down: bool, t: float) -> SmoothResult:
        """point: 필기 좌표 (손이 없으면 None), pen_down: 쓰는 중인지, t: 초 단위 시각."""
        if point is None:
            # 손을 놓치면 획을 끝내고 필터를 초기화해 다시 나타날 때 선이 튀지 않게 한다
            self.filter.reset()
            return SmoothResult(finished=self._finish() if self.stroke is not None else None)

        # 새 획은 이전 획/커서의 필터 잔상 없이 즉시 첫 좌표부터 시작한다.
        if pen_down and self.stroke is None:
            self.filter.reset()
        p = tuple(float(v) for v in self.filter(point, t))
        finished = None
        if pen_down and self.stroke is None:
            self.stroke = []
            self.skip_left = self.start_skip
        elif not pen_down and self.stroke is not None:
            finished = self._finish()

        if self.stroke is not None:
            if self.skip_left > 0:
                self.skip_left -= 1
            else:
                self.stroke.append(p)
        return SmoothResult(point=p, stroke=self.stroke, finished=finished)
