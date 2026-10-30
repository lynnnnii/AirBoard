"""
좌표 떨림 보정 필터 (담당: Lee Subin)

모두 2D 좌표 (x, y)와 시각 t(초)를 받아 보정된 좌표를 돌려준다.
  - NoFilter      : 보정 없음 (비교·--no-smoothing 용)
  - EMAFilter     : 지수이동평균. 단순하지만 손이 빠르면 늦게 따라온다.
  - OneEuroFilter : 손이 느리면 강하게, 빠르면 약하게 보정해 떨림과 지연을 같이 줄인다.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

import numpy as np


class PointFilter(Protocol):
    """필터 공통 형식."""

    def reset(self) -> None: ...

    def __call__(self, point: Sequence[float], t: float) -> np.ndarray: ...


class NoFilter:
    """보정 없이 그대로 돌려준다."""

    def reset(self) -> None:
        pass

    def __call__(self, point: Sequence[float], t: float) -> np.ndarray:
        return np.asarray(point, dtype=float)


class EMAFilter:
    """새 값에 alpha, 이전 값에 (1 - alpha) 가중치. alpha가 작을수록 부드럽지만 늦게 따라온다."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.prev: np.ndarray | None = None

    def reset(self) -> None:
        self.prev = None

    def __call__(self, point: Sequence[float], t: float) -> np.ndarray:
        p = np.asarray(point, dtype=float)
        self.prev = p if self.prev is None else self.alpha * p + (1 - self.alpha) * self.prev
        return self.prev.copy()


def _smoothing_factor(cutoff: float, dt: float) -> float:
    """차단 주파수(Hz)와 프레임 간격(초)으로 EMA 계수를 구한다."""
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro 필터 (Casiez et al., CHI 2012).

    min_cutoff(Hz): 낮출수록 멈춰 있을 때 떨림이 줄어든다.
    beta: 높일수록 빠르게 움직일 때 늦게 따라오는 현상이 줄어든다. 속도(px/s)에 곱해지므로 좌표 단위에 따라 다시 맞춰야 한다.
    d_cutoff(Hz): 속도 추정값을 다듬는 정도 (보통 1.0 고정).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x_prev: np.ndarray | None = None
        self.raw_prev: np.ndarray | None = None
        self.dx_prev = np.zeros(2)
        self.t_prev = 0.0

    def __call__(self, point: Sequence[float], t: float) -> np.ndarray:
        x = np.asarray(point, dtype=float)
        if self.x_prev is None:
            self.x_prev, self.t_prev = x.copy(), t
            self.raw_prev = x.copy()
            return x.copy()

        dt = max(t - self.t_prev, 1e-6)
        a_d = _smoothing_factor(self.d_cutoff, dt)
        dx_hat = a_d * (x - self.raw_prev) / dt + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = _smoothing_factor(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.raw_prev = x.copy()
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat.copy()
