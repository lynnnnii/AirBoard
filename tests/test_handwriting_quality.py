"""직선 떨림과 펜 해제 구간의 획 분리를 검증한다."""
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
        """수평 이동 중 수직 떨림을 줄이면서 진행 방향 지연을 제한한다."""
        cfg=Settings().ui['handwriting_smoothing']
        f=OneEuroFilter(cfg['one_euro_min_cutoff'],cfg['one_euro_beta'])
        raw=np.array([(50+i*3,80+(4 if i%2 else -4)) for i in range(100)],float)
        filtered=np.array([f(p,i/30) for i,p in enumerate(raw)])
        raw_rms=np.sqrt(np.mean((raw[20:,1]-80)**2))
        result_rms=np.sqrt(np.mean((filtered[20:,1]-80)**2))
        self.assertLess(result_rms,raw_rms*.65)
        self.assertLess(np.mean(raw[20:,0]-filtered[20:,0]),15)

    def test_small_pinch_release_stops_without_drawing_tail(self):
        """손가락을 조금 떼는 첫 프레임에 펜을 들고 다음 핀치는 새 획이다."""
        r=GestureRecognizer(self.config()['gestures'])
        hand=np.zeros((21,3),np.float32)
        hand[:,0]=100;hand[:,1]=50;hand[0,1]=80
        def frame(x,ratio,t):
            h=hand.copy();h[:,0]=x
            with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=ratio):
                return r.update([h],640,480,t)
        for i in range(self.config()['gestures']['pinch_on_frames']):
            result=frame(100,.1,i*.05)
        self.assertTrue(result.pen_down)
        self.assertFalse(frame(200,.31,.2).pen_down)
        self.assertFalse(frame(120,.1,.25).pen_down)
        self.assertTrue(frame(130,.1,.3).pen_down)

    def test_clear_release_ends_stroke_on_first_frame(self):
        """확실하게 벌린 손은 두 프레임 대기 없이 획을 끝낸다."""
        d=PinchDetector(self.config()['gestures'])
        for _ in range(3):d.update(.1)
        self.assertTrue(d.down)
        self.assertFalse(d.update(.7))
        for _ in range(3): d.update(.1)
        self.assertTrue(d.down)
