"""Verify jitter suppression on straight lines and stroke separation on pen release."""
import os
os.environ.setdefault('AIRBOARD_USER_SETTINGS', os.path.join(os.path.dirname(__file__), '_no_user_settings.json'))  # Test with team defaults rather than personal settings
import json
import unittest
from unittest.mock import patch
import numpy as np
from airboard.smoothing.filters import OneEuroFilter
from airboard.gestures.gesture import GestureRecognizer,PinchDetector
from airboard.ui.settings import ROOT,Settings


class HandwritingQualityTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/'config/default.json').read_text(encoding='utf-8'))

    def test_moving_straight_line_reduces_sideways_noise_with_bounded_lag(self):
        """Reduce vertical jitter during horizontal motion while limiting directional lag."""
        cfg=Settings().ui['handwriting_smoothing']
        f=OneEuroFilter(cfg['one_euro_min_cutoff'],cfg['one_euro_beta'])
        raw=np.array([(50+i*3,80+(4 if i%2 else -4)) for i in range(100)],float)
        filtered=np.array([f(p,i/30) for i,p in enumerate(raw)])
        raw_rms=np.sqrt(np.mean((raw[20:,1]-80)**2))
        result_rms=np.sqrt(np.mean((filtered[20:,1]-80)**2))
        self.assertLess(result_rms,raw_rms*.65)
        self.assertLess(np.mean(raw[20:,0]-filtered[20:,0]),15)

    def test_pending_release_freezes_endpoint_without_breaking_contact(self):
        """Do not ink movement during release confirmation; continue if the release was a false positive."""
        r=GestureRecognizer(self.config()['gestures'])
        hand=np.zeros((21,3),np.float32)
        hand[:,0]=100;hand[:,1]=50;hand[0,1]=80
        def frame(x,ratio,t):
            h=hand.copy();h[:,0]=x
            with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=ratio):
                return r.update([h],640,480,t)
        for i in range(3): result=frame(100,.1,i*.05)
        start=result.pen_point
        pending=frame(200,.45,.2)
        self.assertEqual(pending.pen_point,start)  # Position remains frozen during release debounce.
        for i in range(3): resumed=frame(120,.1,.25+i*.05)
        self.assertTrue(resumed.pen_down)
        self.assertTrue(frame(300,.45,.5).pen_down)  # First release candidate is debounced.
        self.assertFalse(frame(300,.45,.55).pen_down)

    def test_opening_fingers_stops_ink_before_pen_up(self):
        """Stop adding ink when fingers start separating (above pinch_hold), even before pen-up."""
        r=GestureRecognizer(self.config()['gestures'])
        hand=np.zeros((21,3),np.float32)
        hand[:,0]=100;hand[:,1]=50;hand[0,1]=80
        def frame(x,ratio,t):
            h=hand.copy();h[:,0]=x
            with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=ratio):
                return r.update([h],640,480,t)
        for i in range(3): result=frame(100,.1,i*.05)
        start=result.pen_point
        for i,x in enumerate((140,180)):       # 0.32: Still drawing because the ratio is below pinch_off (0.38)
            lifting=frame(x,.32,.2+i*.05)
            self.assertTrue(lifting.pen_down)
            self.assertEqual(lifting.pen_point,start)
        self.assertNotEqual(frame(200,.1,.3).pen_point,start)   # Continue drawing if fingertips touch again

    def test_clear_release_ends_stroke_on_first_frame(self):
        """A clearly opened hand ends the stroke immediately without waiting two frames."""
        d=PinchDetector(self.config()['gestures'])
        for _ in range(3):d.update(.1)
        self.assertTrue(d.down)
        self.assertFalse(d.update(.7))
        for _ in range(3): d.update(.1)
        self.assertTrue(d.down)
