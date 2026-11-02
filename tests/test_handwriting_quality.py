"""직선 떨림과 펜 해제 구간의 획 분리를 검증한다."""
import os
os.environ.setdefault('AIRBOARD_USER_SETTINGS', os.path.join(os.path.dirname(__file__), '_no_user_settings.json'))  # 개인 설정 없이 팀 기본값으로 검사
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

    def test_pending_release_freezes_endpoint_without_breaking_contact(self):
        """해제 확인 중 이동은 획에 넣지 않고 오판이면 이어 쓴다."""
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
        self.assertTrue(pending.pen_down)
        self.assertEqual(pending.pen_point,start)
        resumed=frame(120,.1,.25)
        self.assertTrue(resumed.pen_down)
        self.assertNotEqual(resumed.pen_point,start)
        pending=frame(250,.45,.3)
        self.assertEqual(pending.pen_point,resumed.pen_point)
        self.assertFalse(frame(300,.45,.35).pen_down)

    def test_opening_fingers_stops_ink_before_pen_up(self):
        """손가락이 벌어지기 시작하면(pinch_hold 초과) 펜이 들리기 전이라도 잉크를 늘리지 않는다."""
        r=GestureRecognizer(self.config()['gestures'])
        hand=np.zeros((21,3),np.float32)
        hand[:,0]=100;hand[:,1]=50;hand[0,1]=80
        def frame(x,ratio,t):
            h=hand.copy();h[:,0]=x
            with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=ratio):
                return r.update([h],640,480,t)
        for i in range(3): result=frame(100,.1,i*.05)
        start=result.pen_point
        for i,x in enumerate((140,180)):       # 0.32: pinch_off(0.38)보다 작아 아직 쓰는 중
            lifting=frame(x,.32,.2+i*.05)
            self.assertTrue(lifting.pen_down)
            self.assertEqual(lifting.pen_point,start)
        self.assertNotEqual(frame(200,.1,.3).pen_point,start)   # 다시 붙이면 계속 쓴다

    def test_clear_release_ends_stroke_on_first_frame(self):
        """확실하게 벌린 손은 두 프레임 대기 없이 획을 끝낸다."""
        d=PinchDetector(self.config()['gestures'])
        for _ in range(3):d.update(.1)
        self.assertTrue(d.down)
        self.assertFalse(d.update(.7))
        for _ in range(3): d.update(.1)
        self.assertTrue(d.down)
