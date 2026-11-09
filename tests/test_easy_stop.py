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
        for separation in [0,20,50]:
            r=GestureRecognizer();c=UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())
            stop=hand();stop[12,:2]=[85+separation,70]
            self.assertEqual(classify_pose(stop,finger_curls(stop)),'TWO_FINGERS')
            poses=[hand(),stop,stop,hand(),hand(True),hand()]
            phases=[]
            for i,lm in enumerate(poses):
                result=r.update([lm],640,480,i/30)
                packet=c.update(from_gesture_result(result,(640,480)),i/30)
                phases.append(packet.phase)
                self.assertEqual(packet.commands,())
            self.assertEqual(phases,['down','up','hover','down','up','down'])

    def test_fast_profile_reduces_lag_without_removing_all_smoothing(self):
        cfg=Settings().ui['handwriting_smoothing']
        new=OneEuroFilter(cfg['one_euro_min_cutoff'],cfg['one_euro_beta'])
        old=OneEuroFilter(1.8,.025)
        raw=np.array([(100+i*3,100+(-4 if i%2 else 4)) for i in range(100)])
        before=np.array([old(p,i/30) for i,p in enumerate(raw)])
        after=np.array([new(p,i/30) for i,p in enumerate(raw)])
        self.assertLess(np.mean(raw[20:,0]-after[20:,0]),15)
        self.assertLess(np.std(after[20:,1]),4)
