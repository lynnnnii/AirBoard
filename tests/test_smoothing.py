"""Unit tests for jitter smoothing and tail trimming. Run: python -m unittest tests.test_smoothing"""

import json
import os
import tempfile
import unittest

import numpy as np

from airboard.smoothing.filters import EMAFilter, NoFilter, OneEuroFilter
from airboard.smoothing.stroke import DEFAULTS, StrokeSmoother, densify, load_config, make_filter

FPS = 30


def noisy_path(seed=0, sigma=3.0):
    """Noisy path: stationary 0–2 s; slow circle (0.5 rev/s) 2–4 s; fast circle (2 rev/s) 4–6 s."""
    t = np.arange(0, 6, 1 / FPS)
    angle = np.where(t < 2, 0.0, np.where(t < 4, (t - 2) * np.pi, 2 * np.pi + (t - 4) * 4 * np.pi))
    moving = t >= 2
    truth = np.full((len(t), 2), 300.0)
    truth[moving] += 100 * np.stack([np.cos(angle[moving]) - 1, np.sin(angle[moving])], axis=1)
    noisy = truth + np.random.default_rng(seed).normal(0, sigma, truth.shape)
    return t, truth, noisy


def rms_error(f, t, truth, noisy, mask):
    out = np.array([f(p, ti) for p, ti in zip(noisy, t)])
    err = np.linalg.norm(out - truth, axis=1)[mask]
    return float(np.sqrt((err ** 2).mean()))


class FilterTest(unittest.TestCase):
    def setUp(self):
        self.t, self.truth, self.noisy = noisy_path()
        self.still = (self.t >= 0.5) & (self.t < 2)
        self.fast = self.t >= 4.2

    def test_first_point_passes_through(self):
        for f in (NoFilter(), EMAFilter(), OneEuroFilter()):
            np.testing.assert_allclose(f((10, 20), 0.0), (10, 20))

    def test_filters_reduce_jitter_when_still(self):
        raw = rms_error(NoFilter(), self.t, self.truth, self.noisy, self.still)
        for f in (EMAFilter(), OneEuroFilter()):
            self.assertLess(rms_error(f, self.t, self.truth, self.noisy, self.still), raw * 0.6)

    def test_one_euro_lags_less_than_ema_when_fast(self):
        ema = rms_error(EMAFilter(), self.t, self.truth, self.noisy, self.fast)
        one_euro = rms_error(OneEuroFilter(), self.t, self.truth, self.noisy, self.fast)
        self.assertLess(one_euro, ema * 0.5)

    def test_reset_forgets_previous_position(self):
        for f in (EMAFilter(), OneEuroFilter()):
            f((0, 0), 0.0)
            f.reset()
            np.testing.assert_allclose(f((500, 500), 0.1), (500, 500))


class StrokeSmootherTest(unittest.TestCase):
    def run_stroke(self, n_down, start_skip=2, end_trim=3):
        """Draw for n_down frames before lifting the pen; x = frame index * 10."""
        s = StrokeSmoother(NoFilter(), start_skip, end_trim)
        finished = None
        for i in range(n_down + 2):
            r = s.update((i * 10, 0), i < n_down, i / FPS)
            finished = r.finished or finished
        return finished

    def test_trims_start_and_end_tails(self):
        xs = [p[0] for p in self.run_stroke(10)]
        self.assertEqual(xs, [20, 30, 40, 50, 60])   # Trim first two points (0, 10) and last three (70, 80, 90)

    def test_no_trim_keeps_all_points(self):
        self.assertEqual(len(self.run_stroke(10, start_skip=0, end_trim=0)), 10)

    def test_short_stroke_is_not_trimmed_away(self):
        # Do not trim the end if the stroke contains at most end_trim + 1 points (preserve short dots/underlines)
        self.assertEqual(len(self.run_stroke(5)), 3)

    def test_finished_reported_once(self):
        s = StrokeSmoother(NoFilter(), 0, 0)
        reports = [s.update((i, 0), i < 5, i / FPS).finished for i in range(8)]
        self.assertEqual(sum(r is not None for r in reports), 1)

    def test_stroke_is_live_while_writing(self):
        s = StrokeSmoother(NoFilter(), 0, 0)
        s.update((0, 0), True, 0.0)
        r = s.update((10, 0), True, 1 / FPS)
        self.assertEqual(r.stroke, [(0.0, 0.0), (10.0, 0.0)])
        self.assertIsNone(r.finished)

    def test_hand_lost_finishes_stroke_and_resets_filter(self):
        s = StrokeSmoother(EMAFilter(0.3), 0, 0)
        for i in range(5):
            s.update((i * 10, 0), True, i / FPS)
        r = s.update(None, False, 5 / FPS)
        self.assertEqual(len(r.finished), 5)
        self.assertIsNone(r.point)
        self.assertIsNone(s.stroke)
        # Reset the filter so a reacquired hand jumps directly to the new position rather than drifting toward the old one
        self.assertEqual(s.update((900, 900), False, 6 / FPS).point, (900.0, 900.0))

    def test_no_stroke_while_pen_up(self):
        s = StrokeSmoother(NoFilter())
        for i in range(5):
            r = s.update((i, 0), False, i / FPS)
            self.assertIsNone(r.stroke)
            self.assertIsNone(r.finished)


class SplineTest(unittest.TestCase):
    """Spline interpolation passes through control points, fills gaps, and never modifies already-emitted points."""

    def corner(self):
        return [(0, 0), (30, 0), (60, 0), (60, 30), (60, 60)]

    def test_densify_passes_through_points_and_fills_gaps(self):
        pts = self.corner()
        out = densify(pts)
        for p in pts:
            self.assertIn(p, [tuple(round(v, 6) for v in q) for q in out])
        gaps = np.linalg.norm(np.diff(np.array(out), axis=0), axis=1)
        self.assertLess(gaps.max(), 4)      # 3 px intervals measured along straight segments; curves can be slightly longer

    def test_live_stroke_only_grows_and_finish_matches_densify(self):
        s = StrokeSmoother(NoFilter(), 0, 0, spline=True)
        previous = []
        for i, p in enumerate(self.corner()):
            stroke = list(s.update(p, True, i / FPS).stroke)
            self.assertEqual(stroke[:len(previous)], previous)   # Renderer only needs to append new points
            previous = stroke
        finished = s.update((60, 60), False, 1.).finished
        np.testing.assert_allclose(finished, densify(self.corner()))

    def test_tail_trim_counts_filtered_points(self):
        s = StrokeSmoother(NoFilter(), 0, 2, spline=True)
        for i, p in enumerate(self.corner()):
            s.update(p, True, i / FPS)
        finished = s.update((60, 60), False, 1.).finished
        np.testing.assert_allclose(finished[-1], (60, 0))       # Ends after trimming the last two smoothed control points


class ConfigTest(unittest.TestCase):
    def test_repo_default_config_loads(self):
        cfg = load_config()
        self.assertEqual(set(cfg), set(DEFAULTS))

    def test_missing_keys_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"smoothing": {"method": "ema", "ema_alpha": 0.5}}, f)
            cfg = load_config(path)
        self.assertEqual(cfg["end_trim"], DEFAULTS["end_trim"])
        s = StrokeSmoother.from_config(cfg)
        self.assertIsInstance(s.filter, EMAFilter)
        self.assertEqual(s.filter.alpha, 0.5)

    def test_method_override_and_unknown_method(self):
        self.assertIsInstance(StrokeSmoother.from_config({}, method="none").filter, NoFilter)
        with self.assertRaises(ValueError):
            make_filter("kalman")


if __name__ == "__main__":
    unittest.main()
