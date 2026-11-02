"""요청된 빠른 지우개·형광펜·핀치 해제·꼬리 제거 설정 회귀 검사."""
import json
import tempfile
import unittest
from pathlib import Path

from airboard.calibration import Calibration
from airboard.gestures.gesture import PinchDetector
from airboard.smoothing.stroke import StrokeSmoother
from airboard.ui.controller import InputState, UIController
from airboard.ui.overlay import Overlay
from airboard.ui.settings import ROOT, Settings


class InputResponsivenessTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.controller = UIController(
            Calibration([[0, 0], [1, 0], [1, 1], [0, 1]]),
            self.settings,
        )

    def frame(self, *gestures, position=(.5, .5), t=1.):
        return self.controller.update(InputState(position, frozenset(gestures)), t)

    def test_no_green_in_palette_and_thin_pen(self):
        colors = self.settings.ui['colors']
        self.assertEqual(colors, ['#2563eb', '#dc2626', '#111827'])
        self.assertEqual(self.settings.ui['widths'], [1, 2, 3])
        self.assertEqual([v for *_, kind, v in Overlay(self.settings).cells if kind == 'color'], colors)
        self.assertIn(self.frame('PINCH').width, (1, 2, 3))

    def test_highlighter_is_immediately_ready_and_only_yellow(self):
        self.frame('DOUBLE_CLAP', position=None)
        packet = self.frame('PINCH')
        self.assertTrue(packet.active)
        self.assertEqual(packet.tool, 'highlighter')
        self.assertEqual(packet.color, '#FFFF00')
        self.assertEqual(packet.width, 24)
        self.frame()
        self.controller.color = '#dc2626'
        packet = self.frame('PINCH')
        self.assertEqual((packet.color, packet.width), ('#FFFF00', 24))
        self.assertTrue(packet.active)

    def test_eraser_reengages_on_first_fist_frame(self):
        first = self.frame('FIST')
        self.assertEqual(first.tool, 'eraser')
        self.assertTrue(first.active)
        self.frame(position=None)
        again = self.frame('FIST')
        self.assertEqual(again.tool, 'eraser')
        self.assertTrue(again.active)

    def test_pinch_small_release_is_one_frame(self):
        cfg = json.loads((ROOT / 'config/default.json').read_text(encoding='utf-8'))['gestures']
        detector = PinchDetector(cfg)
        for _ in range(cfg['pinch_on_frames']):
            detector.update(.1)
        self.assertTrue(detector.down)
        self.assertFalse(detector.update(.31))

    def test_tail_trim_enabled_and_filter_resets_between_strokes(self):
        cfg = self.settings.ui['handwriting_smoothing']
        self.assertEqual(cfg['end_trim'], 3)
        smoother = StrokeSmoother.from_config(cfg)
        smoother.update((10, 10), True, 1.)
        smoother.update((20, 20), False, 1.1)
        fresh = smoother.update((200, 100), True, 2.)
        self.assertIsNotNone(fresh.point)
        self.assertAlmostEqual(fresh.point[0], 200., delta=1.)

    def test_previous_branch_personal_ui_cannot_restore_old_palette(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config').mkdir()
            for name in ('default.json', 'gestures.json'):
                (root / 'config' / name).write_bytes((ROOT / 'config' / name).read_bytes())
            user = {'mapping': {}, 'ui': {
                'colors': ['#2563eb', '#dc2626', '#16a34a', '#111827'],
                'widths': [2, 4, 8],
                'handwriting_smoothing': {'start_skip': 0, 'end_trim': 0,
                                          'method': 'one_euro',
                                          'one_euro_min_cutoff': 1.2,
                                          'one_euro_beta': .008},
            }}
            (root / 'config/ui-user.json').write_text(json.dumps(user), encoding='utf-8')
            restored = Settings(root)
            self.assertEqual(restored.ui['colors'], ['#2563eb', '#dc2626', '#111827'])
            self.assertEqual(restored.ui['widths'], [1, 2, 3])
            self.assertEqual(restored.ui['handwriting_smoothing']['end_trim'], 3)


if __name__ == '__main__':
    unittest.main()
