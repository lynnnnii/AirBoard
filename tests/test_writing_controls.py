"""요청된 획 삭제·분리·도구 스타일을 실제 픽셀과 저장 복원으로 검증한다."""
import json
import os
os.environ.setdefault('AIRBOARD_USER_SETTINGS', os.path.join(os.path.dirname(__file__), '_no_user_settings.json'))
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pymupdf

from airboard.calibration import Calibration
from airboard.gestures.gesture import SwipeDetector
from airboard.render import AirBoardRenderer, RenderConfig
from airboard.smoothing.filters import EMAFilter
from airboard.smoothing.stroke import StrokeSmoother
from airboard.ui.controller import InputState, RenderPacket, UIController
from airboard.ui.demo import PreviewRenderer
from airboard.ui.settings import ROOT, Settings


class WritingControlTests(unittest.TestCase):
    def controller(self) -> UIController:
        return UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]), Settings())

    def test_highlighter_first_pinch_after_pointer_and_pen_style_restore(self) -> None:
        c = self.controller()
        def frame(t: float, *gestures: str) -> RenderPacket:
            return c.update(InputState((.5,.5), frozenset(gestures)), t)
        self.assertEqual(frame(0).width, 2)
        c.color, c.width = '#dc2626', 3
        self.assertEqual(frame(.1, 'DOUBLE_CLAP').tool, 'highlighter')
        for i in range(c.settings.ui['tool_hold_frames']):
            frame(.2 + i*.03, 'POINTER')
        writing = frame(.4, 'PINCH')
        self.assertTrue(writing.active)
        self.assertEqual((writing.color, writing.width), ('#FFFF00', 16))
        frame(.5)
        frame(.6, 'DOUBLE_CLAP')
        writing = frame(.7, 'PINCH')
        self.assertTrue(writing.active)
        self.assertEqual((writing.color, writing.width), ('#dc2626', 3))

    def test_legacy_preferences_migrate_without_losing_custom_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'config').mkdir()
            for name in ('default.json', 'gestures.json'):
                (root/'config'/name).write_bytes((ROOT/'config'/name).read_bytes())
            old = {'mapping': {'PINCH':'draw', 'DOUBLE_CLAP':'none',
                               'SWIPE_LEFT':'page_prev', 'SWIPE_RIGHT':'page_next'},
                   'ui': {'colors':['#16a34a'], 'widths':[2,4,8],
                          'handwriting_smoothing':{'end_trim':0, 'one_euro_beta':.02}}}
            (root/'config/ui-user.json').write_text(json.dumps(old))
            settings = Settings(root)
            self.assertEqual(settings.mapping['DOUBLE_CLAP'], 'none')
            self.assertEqual(settings.mapping['SWIPE_DOWN'], 'page_next')
            self.assertNotIn('SWIPE_RIGHT', settings.mapping)
            self.assertEqual(settings.ui['colors'], ['#2563eb','#dc2626','#000000'])
            self.assertEqual(settings.ui['widths'], [1,2,3])
            self.assertEqual(settings.ui['handwriting_smoothing']['end_trim'], 2)
            self.assertEqual(settings.ui['handwriting_smoothing']['one_euro_beta'], .02)
            settings.save()
            self.assertEqual(Settings(root).ui, settings.ui)

    def test_swipe_uses_screen_height_and_rejects_horizontal_or_diagonal(self) -> None:
        for points in ([(100,100),(200,100),(300,100)],
                       [(100,100),(200,150),(300,200)]):
            d = SwipeDetector({'swipe_dist':.14})
            for i, point in enumerate(points):
                self.assertIsNone(d.update(point, True, 1000, 500, i*.1))
        for ys, expected in (([100,140,180], 'SWIPE_DOWN'), ([180,140,100], 'SWIPE_UP')):
            d = SwipeDetector({'swipe_dist':.14})
            events = [d.update((100,y), True, 1000, 500, i*.1) for i,y in enumerate(ys)]
            self.assertEqual(events, [None, None, expected])

    def test_new_stroke_does_not_inherit_hover_filter_position(self) -> None:
        s = StrokeSmoother(EMAFilter(.1), 0, 0)
        s.update((20,20), True, 0)
        s.update((30,20), False, .03)
        s.update((40,20), False, .06)
        result = s.update((250,130), True, .09)
        self.assertEqual(result.stroke, [(250.,130.)])

    def test_preview_and_pdf_pipeline_separate_strokes_and_trim_tail(self) -> None:
        for tool, width, color in (('pen',2,'#000000'), ('highlighter',16,'#FFFF00')):
            with self.subTest(tool=tool):
                options = {**Settings().ui['handwriting_smoothing'], 'method':'none'}
                r = PreviewRenderer((320,180), options)
                try:
                    def frame(point, active, phase):
                        return r.consume(RenderPacket(point,tool,active,phase,color,width,.3,()))
                    for i,x in enumerate((20,30,40,50,60,100,140)):
                        frame((x,60), True, 'down' if i==0 else 'move')
                    # 해제 프레임의 이동과 직전 꼬리 둘 다 영구 획에 남지 않는다.
                    frame((220,60), False, 'up')
                    image = frame((250,130), True, 'down')
                    frame(None, False, 'up')
                    self.assertEqual(r.board.operation_count(), 2)
                    np.testing.assert_array_equal(image[60,100], [255,255,255])
                    np.testing.assert_array_equal(image[95,150], [255,255,255])
                    self.assertFalse(np.array_equal(image[60,30], [255,255,255]))
                    self.assertFalse(np.array_equal(image[130,250], [255,255,255]))
                    frame((30,60), False, 'hover')
                    r.consume(RenderPacket((30,60),'eraser',True,'down','#000000',40,1.,()))
                    r.consume(RenderPacket(None,'eraser',False,'up','#000000',40,1.,()))
                    np.testing.assert_array_equal(r.board.render_page()[60,30], [255,255,255])
                    self.assertFalse(np.array_equal(r.board.render_page()[130,250], [255,255,255]))
                finally:
                    r.close()
