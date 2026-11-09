"""Regression checks for the current pinch-writing and smoothing profile."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from airboard.gestures.gesture import GestureRecognizer
from airboard.smoothing.stroke import StrokeSmoother
from airboard.smoothing.filters import OneEuroFilter
from airboard.ui.settings import Settings, ROOT
from airboard.ui.controller import UIController, InputState
from airboard.calibration import Calibration
from tests.test_index_writing import hand


class ReferenceWritingTests(unittest.TestCase):
    def controller(self):
        return UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())

    def test_pinch_draws(self):
        p=self.controller().update(InputState((.5,.5),frozenset({'PINCH'})),0.)
        self.assertEqual(p.phase,'down')

    def test_empty_pose_never_starts_ink(self):
        p=self.controller().update(InputState((.5,.5),frozenset()),0.)
        self.assertFalse(p.active)

    def test_stop_lifts_pen_immediately(self):
        c=self.controller();c.update(InputState((.5,.5),frozenset({'PINCH'})),0)
        p=c.update(InputState((.6,.6),frozenset({'PINCH','TWO_FINGERS'})),.1)
        self.assertEqual(p.phase,'up')

    def test_pointer_does_not_draw(self):
        p=self.controller().update(InputState((.5,.5),frozenset({'POINTER'})),0)
        self.assertFalse(p.active)

    def test_undo_command(self):
        c=self.controller();p=c.update(InputState((.5,.5),frozenset({'THUMB_LEFT'})),0)
        self.assertEqual(p.commands,('undo',))

    def test_short_stroke_end_and_next_stroke_preserved(self):
        s=StrokeSmoother.from_config(Settings().ui['handwriting_smoothing'])
        for i in range(6):s.update((100+i*2,100),True,i/30)
        self.assertTrue(s.update((400,400),False,.3).finished)
        new=s.update((300,300),True,.4)
        self.assertIsNotNone(new.stroke)

    def test_landmarks_reach_adapter_as_integer_pixels(self):
        # 현재 인식기는 검지 좌표를 정수 픽셀로 넘긴다. 소수점 전달은 실제 필기감이 나빠져 되돌렸다.
        r=GestureRecognizer();lm=hand();lm[8,0]+=.375
        result=r.update([lm],640,480,0)
        self.assertEqual(result.pointer_point[0],85)

    def test_mapping_profile_preserves_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir()
            for name in ('default.json','gestures.json'):
                (root/'config'/name).write_bytes((ROOT/'config'/name).read_bytes())
            settings=Settings(root)
            settings.set_action('POINTER','draw');settings.save()
            loaded=Settings(root)
            self.assertEqual(loaded.mapping['POINTER'],'draw')
            self.assertEqual(loaded.ui['default_corners'],settings.ui['default_corners'])

    def test_one_euro_stabilizes_noise(self):
        f=OneEuroFilter(1.5,.01)
        raw=np.array([(50+i*3,80+(-4 if i%2==0 else 4)) for i in range(100)],float)
        after=np.array([f(p,i/30) for i,p in enumerate(raw)])
        self.assertLess(np.std(after[20:,1]),np.std(raw[20:,1]))
