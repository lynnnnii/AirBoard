"""팀 피드백 반영 회귀 테스트: 획 지우개, 펜 떼는 중 잉크 정지, 박수 뒤 재필기, 형광펜 고정 색·굵기."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pymupdf

from airboard.calibration import Calibration
from airboard.gestures.gesture import GestureRecognizer
from airboard.render import AirBoardRenderer, RenderConfig
from airboard.ui.controller import InputState, UIController
from airboard.ui.settings import ROOT, Settings


def gesture_config():
    return json.loads((ROOT / 'config/default.json').read_text(encoding='utf-8'))['gestures']


def hand(x, y=50):
    """손바닥 길이 30픽셀인 합성 랜드마크."""
    a = np.zeros((21, 3), np.float32)
    a[:, 0] = x
    a[:, 1] = y
    a[0, 1] = y + 30
    return a


class StrokeEraserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / 'source.pdf'
        with pymupdf.open() as doc:
            doc.new_page(width=320, height=180)
            doc.save(path)
        self.board = AirBoardRenderer(path, RenderConfig(320, 180, 1, '#FF0000', 4, '#FFFF00', 20, .3, 20))
        self.bg = self.board.render_page()
        self.board.commit_stroke([(20, 40), (250, 40)])
        self.board.commit_stroke([(20, 100), (250, 100)], 'highlighter')

    def tearDown(self):
        self.board.close()
        self.temp.cleanup()

    def test_touching_one_point_removes_whole_stroke_only(self):
        self.board.update((60, 45), 'eraser', False, eraser_active=True)
        image = self.board.render_page()
        np.testing.assert_array_equal(image[40, 240], self.bg[40, 240])   # 닿지 않은 끝까지 사라짐
        self.assertFalse(np.array_equal(image[100, 240], self.bg[100, 240]))  # 다른 획은 남음
        self.assertEqual(self.board.operation_count(), 3)

    def test_undo_restores_erased_stroke_and_miss_does_nothing(self):
        before = self.board.render_page()
        self.assertEqual(self.board.erase_at((160, 160)), 0)
        self.assertEqual(self.board.operation_count(), 2)
        self.assertEqual(self.board.erase_at((60, 100)), 1)
        self.assertTrue(self.board.undo())
        np.testing.assert_array_equal(self.board.render_page(), before)

    def test_erased_strokes_survive_session_reload(self):
        self.board.erase_at((60, 40))
        expected = self.board.render_page()
        state = Path(self.temp.name) / 'state.json'
        self.board.save_session(state)
        other = AirBoardRenderer(self.board.pdf_path, self.board.config)
        try:
            other.load_session(state)
            np.testing.assert_array_equal(other.render_page(), expected)
            other.undo()
            self.assertFalse(np.array_equal(other.render_page()[40, 240], self.bg[40, 240]))
        finally:
            other.close()


class PenLiftTests(unittest.TestCase):
    def frame(self, r, x, ratio, t):
        with patch('airboard.gestures.gesture.classify_pose', return_value='NONE'), \
                patch('airboard.gestures.gesture.pinch_ratio', return_value=ratio):
            return r.update([hand(x)], 640, 480, t)

    def start(self):
        r = GestureRecognizer(gesture_config())
        for i in range(3):
            result = self.frame(r, 100, .1, i * .05)
        self.assertTrue(result.pen_down)
        return r, result.pen_point

    def test_opening_fingers_stops_ink_and_moving_away_starts_new_stroke(self):
        r, start = self.start()
        for i, x in enumerate((150, 200)):          # 손가락을 벌리기 시작(0.30)한 채 이동
            held = self.frame(r, x, .30, .2 + i * .05)
            self.assertTrue(held.pen_down)
            self.assertEqual(held.pen_point, start)
        split = self.frame(r, 250, .1, .3)           # 다른 곳에서 다시 붙임
        self.assertFalse(split.pen_down)             # 이어 긋지 않도록 한 프레임 끊는다
        resumed = self.frame(r, 255, .1, .35)
        self.assertTrue(resumed.pen_down)
        self.assertEqual(resumed.pen_point[0], 255)

    def test_single_frame_wobble_keeps_writing(self):
        r, start = self.start()
        self.assertEqual(self.frame(r, 150, .30, .2).pen_point, start)
        self.assertTrue(self.frame(r, 160, .1, .25).pen_down)


class ClapRearmTests(unittest.TestCase):
    def run_clap_then_pinch(self, config):
        r = GestureRecognizer(config)
        with patch('airboard.gestures.gesture.classify_pose', return_value='NONE'), \
                patch('airboard.gestures.gesture.pinch_ratio', return_value=.1):
            events = []
            for t, gap in ((0, 4), (.2, 1.4), (.8, 3.5), (1.5, 1.4)):
                events += r.update([hand(100), hand(100 + 30 * gap)], 640, 480, t).events
            self.assertIn('DOUBLE_CLAP', events)
            return [r.update([hand(100)], 640, 480, 1.7 + i * .05).pen_down for i in range(4)]

    def test_pinched_hand_writes_right_after_clap(self):
        self.assertTrue(self.run_clap_then_pinch(gesture_config())[-1])

    def test_without_rearm_old_behaviour_needs_finger_open(self):
        self.assertFalse(any(self.run_clap_then_pinch({**gesture_config(), 'clap_rearm_pinch': False})))


class ToolStyleTests(unittest.TestCase):
    def test_highlighter_is_yellow_and_thick_pen_uses_palette(self):
        settings = Settings()
        c = UIController(Calibration([[0, 0], [1, 0], [1, 1], [0, 1]]), settings)
        pen = c.update(InputState((.5, .5)), 0)
        self.assertIn(pen.color, settings.ui['colors'])
        self.assertLessEqual(pen.width, 3)
        c.update(InputState((.5, .5), frozenset({'DOUBLE_CLAP'})), 1)
        hl = c.update(InputState((.5, .5)), 1.1)
        self.assertEqual(hl.tool, 'highlighter')
        self.assertEqual(hl.color, settings.ui['highlighter_color'])
        self.assertGreater(hl.width, pen.width * 5)


if __name__ == '__main__':
    unittest.main()
