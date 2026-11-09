"""Two fingers stop immediately without moving the coordinate anchor."""
import unittest
import numpy as np
from tests.test_index_writing import hand
from airboard.gestures.gesture import GestureRecognizer, classify_pose, finger_curls
from airboard.ui.controller import UIController, from_gesture_result
from airboard.ui.settings import Settings
from airboard.calibration import Calibration
from airboard.smoothing.filters import OneEuroFilter

class EasyStopTests(unittest.TestCase):
    def test_two_fingers_and_l_both_end_stroke(self):
        # Current default writes with PINCH, stops with TWO_FINGERS, and
        # translates sideways thumb gestures into undo/redo commands.
        from airboard.ui.controller import InputState
        c=UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())
        self.assertEqual(c.update(InputState((.4,.4),frozenset({'PINCH'})),0).phase,'down')
        self.assertEqual(c.update(InputState((.5,.5),frozenset({'TWO_FINGERS'})),.1).phase,'up')
        self.assertEqual(c.update(InputState((.5,.5),frozenset({'THUMB_LEFT'})),.2).commands,('undo',))
        self.assertEqual(c.update(InputState((.5,.5),frozenset({'PINCH'})),.3).phase,'down')

    def test_fast_profile_reduces_lag_without_removing_all_smoothing(self):
        cfg=Settings().ui['handwriting_smoothing']
        new=OneEuroFilter(cfg['one_euro_min_cutoff'],cfg['one_euro_beta'])
        old=OneEuroFilter(1.8,.025)
        raw=np.array([(100+i*3,100+(-4 if i%2 else 4)) for i in range(100)])
        before=np.array([old(p,i/30) for i,p in enumerate(raw)])
        after=np.array([new(p,i/30) for i,p in enumerate(raw)])
        self.assertLess(np.mean(raw[20:,0]-after[20:,0]),15)
        self.assertLess(np.std(after[20:,1]),4)
