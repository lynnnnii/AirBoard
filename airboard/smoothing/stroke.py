"""
Jitter smoothing and stroke-tail trimming (contributor: Lee Subin)

Provide coordinates and pen_down each frame to obtain smoothed cursor positions and strokes.
Any 2D coordinate units work, including camera or slide pixels.

  smoother = StrokeSmoother.from_config(load_config())   # The "smoothing" section of config/default.json
  s = smoother.update(point, pen_down, t)
  - s.point    : Smoothed cursor position (None when the hand is missing) for cursor rendering
  - s.stroke   : Current stroke points (None if not drawing) for live rendering
  - s.finished : Completed stroke with tail trimmed (or None) for committing to the ink layer

Tail trimming
  - Start tail: discard the first start_skip frames after pen-down.
  - End tail: fingertip motion precedes pen-up while releasing a pinch; the ratio must exceed pinch_off
    for enough frames before pen_down becomes False. Thus tail points enter the stroke before state changes,
    so remove the last end_trim points when lifting the pen.

Curve interpolation (spline=True)
  At low webcam frame rates, straight interpolation makes curved handwriting angular.
  Fill gaps with Catmull-Rom splines. Each curve segment needs a subsequent point,
  so an active stroke is rendered one point behind; complete the remaining segment at pen-up.
  Previously emitted points remain fixed, so the renderer only needs to append new points.
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
    "spline": False,             # If True, interpolate curves between points
}

Point = tuple[float, float]
SPLINE_STEP = 3.0                # Distance between spline sample points (typically slide pixels)
SPLINE_MAX_STEPS = 12


def _catmull_rom(p0: Point, p1: Point, p2: Point, p3: Point) -> list[Point]:
    """Points along p1 to p2 (excluding p1, including p2); p0 and p3 set tangent directions."""
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
    """Interpolate the full point sequence; duplicate endpoints to avoid spline overshoot."""
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i > 0 else points[i]
        p3 = points[i + 2] if i + 2 < len(points) else points[i + 1]
        out += _catmull_rom(p0, points[i], points[i + 1], p3)
    return out


def load_config(path: str = DEFAULT_CONFIG) -> dict[str, Any]:
    """The "smoothing" section of config/default.json 항목을 기본값 위에 덮어써서 돌려준다."""
    with open(path, encoding="utf-8") as f:
        section = json.load(f).get("smoothing", {})
    return {**DEFAULTS, **section}


def make_filter(method: str, cfg: dict[str, Any] | None = None) -> PointFilter:
    """Construct a filter from its method name."""
    c = {**DEFAULTS, **(cfg or {})}
    if method == "none":
        return NoFilter()
    if method == "ema":
        return EMAFilter(c["ema_alpha"])
    if method == "one_euro":
        return OneEuroFilter(c["one_euro_min_cutoff"], c["one_euro_beta"], c["one_euro_d_cutoff"])
    raise ValueError(f"Unknown smoothing method: {method}")


@dataclass
class SmoothResult:
    """Smoothing result for one frame."""

    point: Point | None = None
    stroke: list[Point] | None = None
    finished: list[Point] | None = None


class StrokeSmoother:
    """Filter coordinates, split strokes on pen_down transitions, and remove tails."""

    def __init__(self, point_filter: PointFilter | None = None,
                 start_skip: int = DEFAULTS["start_skip"], end_trim: int = DEFAULTS["end_trim"],
                 spline: bool = DEFAULTS["spline"]):
        self.filter = OneEuroFilter() if point_filter is None else point_filter
        self.start_skip = start_skip
        self.end_trim = end_trim
        self.spline = spline
        self.stroke: list[Point] | None = None
        self.skip_left = 0
        self._ctrl: list[Point] = []      # Filtered control points before spline interpolation (only used for splines)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], method: str | None = None) -> StrokeSmoother:
        """Build from config; an explicit method overrides the configured method for comparisons."""
        c = {**DEFAULTS, **cfg}
        return cls(make_filter(method or c["method"], c), c["start_skip"], c["end_trim"], bool(c["spline"]))

    def _finish(self) -> list[Point] | None:
        stroke, self.stroke = self.stroke, None
        if self.spline:
            # Trim filtered control points, then reinterpolate the remaining stroke.
            stroke, self._ctrl = self._ctrl, []
        if self.end_trim > 0 and len(stroke) > self.end_trim + 1:
            stroke = stroke[:-self.end_trim]
        if self.spline:
            stroke = densify(stroke)
        return stroke or None

    def _add(self, p: Point) -> None:
        """Append a filtered point; with splines, emit the newly determined previous curve segment."""
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
        """point: input position (None if hand missing); pen_down: writing state; t: time in seconds."""
        if point is None:
            # Finish the stroke and reset filters on tracking loss to avoid jumps when the hand reappears
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
