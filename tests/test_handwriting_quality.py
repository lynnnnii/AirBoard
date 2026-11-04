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

    def test_release_breaks_contact_before_releasing_hand_moves(self):
        """손가락을 떼는 첫 프레임부터 필기를 멈추고 새 핀치를 별도 획으로 시작한다."""
        r=GestureRecognizer(self.config()['gestures'])
        hand=np.zeros((21,3),np.float32)
        hand[:,0]=100;hand[:,1]=50;hand[0,1]=80
        def frame(x,ratio,t):
            h=hand.copy();h[:,0]=x
            with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=ratio):
                return r.update([h],640,480,t)
        for i in range(3): result=frame(100,.1,i*.05)
        self.assertTrue(result.pen_down)
        self.assertFalse(frame(200,.30,.2).pen_down)
        self.assertFalse(frame(120,.1,.25).pen_down)
        self.assertFalse(frame(120,.1,.30).pen_down)
        self.assertTrue(frame(120,.1,.35).pen_down)
        self.assertFalse(frame(300,.45,.4).pen_down)

    def test_opening_fingers_ends_stroke_without_waiting_for_full_release(self):
        """손가락이 벌어지기 시작하면(pinch_hold 도달) 획을 종료한다."""
        r=GestureRecognizer(self.config()['gestures'])
        hand=np.zeros((21,3),np.float32)
        hand[:,0]=100;hand[:,1]=50;hand[0,1]=80
        def frame(x,ratio,t):
            h=hand.copy();h[:,0]=x
            with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=ratio):
                return r.update([h],640,480,t)
        for i in range(3): result=frame(100,.1,i*.05)
        self.assertTrue(result.pen_down)
        for i,x in enumerate((140,180)):
            self.assertFalse(frame(x,.32,.2+i*.05).pen_down)
        for i in range(2):
            self.assertFalse(frame(200,.1,.3+i*.05).pen_down)
        self.assertTrue(frame(200,.1,.4).pen_down)

    def test_clear_release_ends_stroke_on_first_frame(self):
        """확실하게 벌린 손은 두 프레임 대기 없이 획을 끝낸다."""
        d=PinchDetector(self.config()['gestures'])
        for _ in range(3):d.update(.1)
        self.assertTrue(d.down)
        self.assertFalse(d.update(.7))
        for _ in range(3): d.update(.1)
        self.assertTrue(d.down)
