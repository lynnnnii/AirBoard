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

곡선 보간 (spline=True)
  웹캠이 초당 10여 장이라 보정된 점 사이를 직선으로 이으면 둥근 글씨가 각져 보인다.
  점 사이를 Catmull-Rom 곡선으로 촘촘히 채운다. 곡선의 한 구간은 다음 점이 있어야 정해지므로
  진행 중인 획은 마지막 한 점 뒤까지만 그려지고(한 프레임 늦음), 펜을 들면 끝까지 채운다.
  이미 내보낸 점은 바뀌지 않으므로 렌더러는 새 점만 이어 그리면 된다.
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
    "spline": False,             # True면 점 사이를 곡선으로 채운다
}

Point = tuple[float, float]
SPLINE_STEP = 3.0                # 곡선 위 점 간격(좌표 단위, 보통 슬라이드 px)
SPLINE_MAX_STEPS = 12


def _catmull_rom(p0: Point, p1: Point, p2: Point, p3: Point) -> list[Point]:
    """p1→p2 구간을 곡선으로 채운 점들 (p1 제외, p2 포함). p0, p3은 곡선 방향을 정하는 이웃 점."""
    length = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** .5
    n = min(SPLINE_MAX_STEPS, max(1, int(length / SPLINE_STEP + .999)))
    out = []
    for i in range(1, n + 1):
        t = i / n
        t2, t3 = t * t, t * t * t
        out.append(tuple(.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - a - 3 * c + d) * t3)
                         for a, b, c, d in zip(p0, p1, p2, p3)))
    return out


def densify(points: list[Point]) -> list[Point]:
    """점 목록 전체를 곡선으로 채운다. 양 끝은 끝점을 이웃으로 써서 곡선이 튀어 나가지 않게 한다."""
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i > 0 else points[i]
        p3 = points[i + 2] if i + 2 < len(points) else points[i + 1]
        out += _catmull_rom(p0, points[i], points[i + 1], p3)
    return out


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
                 start_skip: int = DEFAULTS["start_skip"], end_trim: int = DEFAULTS["end_trim"],
                 spline: bool = DEFAULTS["spline"]):
        self.filter = OneEuroFilter() if point_filter is None else point_filter
        self.start_skip = start_skip
        self.end_trim = end_trim
        self.spline = spline
        self.stroke: list[Point] | None = None
        self.skip_left = 0
        self._ctrl: list[Point] = []      # 곡선 보간 전 보정 점 (spline일 때만 사용)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], method: str | None = None) -> StrokeSmoother:
        """설정값으로 만든다. method를 주면 설정의 method 대신 쓴다 (비교용)."""
        c = {**DEFAULTS, **cfg}
        return cls(make_filter(method or c["method"], c), c["start_skip"], c["end_trim"], bool(c["spline"]))

    def _finish(self) -> list[Point] | None:
        stroke, self.stroke = self.stroke, None
        if self.spline:
            # 꼬리는 보정 점 기준으로 지우고, 남은 점 전체를 다시 곡선으로 채운다.
            stroke, self._ctrl = self._ctrl, []
        if self.end_trim > 0 and len(stroke) > self.end_trim + 1:
            stroke = stroke[:-self.end_trim]
        if self.spline:
            stroke = densify(stroke)
        return stroke or None

    def _add(self, p: Point) -> None:
        """획에 보정 점을 더한다. spline이면 이제 모양이 정해진 직전 구간을 곡선으로 채워 내보낸다."""
        if not self.spline:
            self.stroke.append(p)
            return
        c = self._ctrl
        c.append(p)
        if len(c) == 1:
            self.stroke.append(p)
        elif len(c) >= 3:
            self.stroke += _catmull_rom(c[-4] if len(c) >= 4 else c[-3], c[-3], c[-2], c[-1])

    def update(self, point: Sequence[float] | None, pen_down: bool, t: float) -> SmoothResult:
        """point: 필기 좌표 (손이 없으면 None), pen_down: 쓰는 중인지, t: 초 단위 시각."""
        if point is None:
            # 손을 놓치면 획을 끝내고 필터를 초기화해 다시 나타날 때 선이 튀지 않게 한다
            self.filter.reset()
            return SmoothResult(finished=self._finish() if self.stroke is not None else None)

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
                self._add(p)
        return SmoothResult(point=p, stroke=self.stroke, finished=finished)
