"""
Coordinate jitter smoothing filters (contributor: Lee Subin)

Each filter accepts 2D coordinates (x, y) and time t (seconds), then returns smoothed coordinates.
  - NoFilter      : No smoothing (for comparisons and --no-smoothing)
  - EMAFilter     : Exponential moving average: simple, but lags behind fast hand movement.
  - OneEuroFilter : Applies stronger smoothing when slow and weaker smoothing when fast, balancing jitter and lag.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

import numpy as np


class PointFilter(Protocol):
    """Common filter interface."""

    def reset(self) -> None: ...

    def __call__(self, point: Sequence[float], t: float) -> np.ndarray: ...


class NoFilter:
    """Return coordinates unchanged."""

    def reset(self) -> None:
        pass

    def __call__(self, point: Sequence[float], t: float) -> np.ndarray:
        return np.asarray(point, dtype=float)


class EMAFilter:
    """Weight the new value by alpha and the previous value by 1-alpha; lower alpha is smoother but increases lag."""

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
    """Calculate the EMA coefficient from cutoff frequency (Hz) and frame interval (seconds)."""
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012).

    min_cutoff(Hz): Lower values reduce jitter when stationary.
    beta: Higher values reduce lag during fast motion; recalibrate for coordinate units since beta scales velocity (px/s).
    d_cutoff(Hz): Smoothing of the velocity estimate (usually fixed at 1.0).
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
