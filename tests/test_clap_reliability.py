"""박수 가림 허용 및 페이지 오작동 방지를 검증한다."""
import os
os.environ.setdefault('AIRBOARD_USER_SETTINGS', os.path.join(os.path.dirname(__file__), '_no_user_settings.json'))  # 개인 설정 없이 팀 기본값으로 검사
import json
import unittest
from unittest.mock import patch
import numpy as np
from airboard.gestures.gesture import ClapDetector,GestureRecognizer
from airboard.ui.settings import ROOT,Settings
from airboard.ui.controller import UIController,from_gesture_result
from airboard.calibration import Calibration


def hand(x):
    """손바닥 길이 30픽셀인 합성 랜드마크."""
    a=np.zeros((21,3),np.float32)
    a[:,0]=x; a[:,1]=50
    a[0,1]=80; a[9,1]=50
    return a


def pair(distance):
    return [hand(100),hand(100+30*distance)]


class ClapTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'config/default.json').read_text(encoding='utf-8'))['gestures']

    def test_two_deliberate_claps_and_cooldown(self):
        d=ClapDetector(self.config)
        events=[]
        for t,gap in ((0,4),(.2,1.4),(.8,3.5),(1.5,1.4),(1.6,3.5),(1.7,1.4)):
            if d.update(pair(gap),t):events.append(t)
        self.assertEqual(events,[1.5])

    def test_two_occluded_contacts_are_supported(self):
        d=ClapDetector(self.config)
        events=[]
        sequence=[(0,pair(4)),(.15,pair(2.6)),(.2,[hand(100)]),
                  (.45,pair(3.5)),(.6,pair(2.6)),(.65,[])]
        for t,hands in sequence:
            if d.update(hands,t):events.append(t)
        self.assertEqual(events,[.65])

    def test_held_contact_or_random_loss_is_not_double_clap(self):
        d=ClapDetector(self.config)
        self.assertFalse(d.update(pair(4),0))
        self.assertFalse(d.update(pair(1.4),.1))
        for t in (.2,.3,.6,1.,1.5):
            self.assertFalse(d.update([hand(100)],t))
        self.assertEqual(len(d.claps),1)
        d=ClapDetector(self.config)
        for i in range(10):
            self.assertFalse(d.update([hand(100)] if i%2 else [],i*.1))
        self.assertEqual(len(d.claps),0)

    def test_two_hand_occlusion_cannot_become_swipe(self):
        r=GestureRecognizer(self.config)
        curls={name:2. for name in ('index','middle','ring','pinky')}
        with patch('airboard.gestures.gesture.classify_pose',return_value='OPEN'),patch('airboard.gestures.gesture.finger_curls',return_value=curls),patch.object(r.pinch,'update',return_value=False):
            r.update(pair(4),640,480,0)
            for t,x in ((.1,100),(.2,240),(.3,400),(.5,100)):
                self.assertFalse(r.update([hand(x)],640,480,t).events)
            found=[]
            for t,x in ((1.,100),(1.1,180),(1.2,260)):
                found+=r.update([hand(x)],640,480,t).events
            self.assertEqual(found,[])

    def test_clap_event_toggles_ui_without_page_command(self):
        r=GestureRecognizer(self.config)
        c=UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())
        c.settings.mapping['PINCH']='draw'  # Explicit legacy preset under test
        commands=[]
        with patch('airboard.gestures.gesture.classify_pose',return_value='FIST'):
            for t,hands in [(0,pair(4)),(.15,pair(2.6)),(.2,[hand(100)]),
                            (.45,pair(3.5)),(.6,pair(2.6)),(.65,[hand(100)])]:
                result=r.update(hands,640,480,t)
                packet=c.update(from_gesture_result(result,(640,480)),t)
                commands+=packet.commands
        self.assertEqual(c.base_tool,'highlighter')
        self.assertEqual(commands,[])


    def test_failed_first_clap_does_not_lock_writing(self):
        """박수 1회 대기 중에도 한 손으로 돌아와 새 핀치를 하면 쓴다."""
        r=GestureRecognizer(self.config)
        with patch('airboard.gestures.gesture.classify_pose',return_value='OPEN'),patch('airboard.gestures.gesture.pinch_ratio',return_value=.5):
            r.update(pair(4),640,480,0)
            r.update(pair(1.4),640,480,.2)
            r.update([hand(100)],640,480,.4)
        with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=.1):
            result=None
            for t in (.45,.5,.55):
                result=r.update([hand(100)],640,480,t)
        self.assertTrue(result.pen_down)
        self.assertFalse(result.debug['clap_ink_blocked'])
        self.assertTrue(result.debug['clap_navigation_blocked'])

    def test_highlighter_writes_before_navigation_cooldown_finishes(self):
        """형광펜 전환 뒤 페이지 잠금과 별개로 새 필기를 허용한다."""
        r=GestureRecognizer(self.config)
        c=UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())
        c.settings.mapping['PINCH']='draw'  # Explicit legacy preset under test
        with patch('airboard.gestures.gesture.classify_pose',return_value='OPEN'),patch('airboard.gestures.gesture.pinch_ratio',return_value=.5):
            for t,gap in ((0,4),(.2,1.4),(.4,3.5),(.6,1.4)):
                result=r.update(pair(gap),640,480,t)
                c.update(from_gesture_result(result,(640,480)),t)
            result=r.update([hand(100)],640,480,.75)
            c.update(from_gesture_result(result,(640,480)),.75)
        with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=.1):
            for t in (.8,.85,.9):
                result=r.update([hand(100)],640,480,t)
                packet=c.update(from_gesture_result(result,(640,480)),t)
        self.assertEqual(packet.tool,'highlighter')
        self.assertTrue(packet.active)
        self.assertEqual(packet.commands,())
        self.assertTrue(result.debug['clap_navigation_blocked'])


    def test_contact_after_short_reacquisition_is_accepted(self):
        """가까워지던 두 손이 잠깐 사라져도 다시 검출된 접촉을 이어 센다."""
        d=ClapDetector(self.config)
        self.assertFalse(d.update(pair(4),0))
        self.assertFalse(d.update(pair(3.3),.1))
        self.assertFalse(d.update([hand(100)],.2))
        self.assertEqual(len(d.claps),1)
        self.assertFalse(d.update(pair(2.4),.7))
        self.assertTrue(d.update([hand(100)],.8))

    def test_fist_noise_during_pinch_does_not_require_repinch(self):
        """로그에서 관찰한 경계 주먹 자세가 지속 핀치를 끊지 않게 한다."""
        r=GestureRecognizer(self.config)
        tips={'index':.78,'middle':.5,'ring':.4,'pinky':.35}
        with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=.1):
            for t in (0,.05,.1):r.update([hand(100)],640,480,t)
        with patch('airboard.gestures.gesture.classify_pose',return_value='FIST'),patch('airboard.gestures.gesture.tip_to_palm',return_value=tips),patch('airboard.gestures.gesture.pinch_ratio',return_value=.1):
            for t in (.15,.2,.25,.3,.35):
                self.assertTrue(r.update([hand(100)],640,480,t).pen_down)
        tips['index']=.6
        with patch('airboard.gestures.gesture.classify_pose',return_value='FIST'),patch('airboard.gestures.gesture.tip_to_palm',return_value=tips),patch('airboard.gestures.gesture.pinch_ratio',return_value=.1):
            for t in (.4,.45,.5,.55,.6):
                result=r.update([hand(100)],640,480,t)
                self.assertTrue(result.pen_down)
                self.assertEqual(result.pose,'PINCH')
        with patch('airboard.gestures.gesture.classify_pose',return_value='OPEN'),patch('airboard.gestures.gesture.pinch_ratio',return_value=.5):
            r.update([hand(100)],640,480,.65)
            self.assertFalse(r.update([hand(100)],640,480,.68).pen_down)
        with patch('airboard.gestures.gesture.classify_pose',return_value='FIST'),patch('airboard.gestures.gesture.tip_to_palm',return_value=tips),patch('airboard.gestures.gesture.pinch_ratio',return_value=.1):
            for t in (.7,.75,.8,.85):
                result=r.update([hand(100)],640,480,t)
                self.assertFalse(result.pen_down)
            self.assertEqual(result.pose,'FIST')


    def test_release_during_clap_guard_allows_first_pinch_after_guard(self):
        """보호 시간 중 벌린 손을 기억해 형광펜 전환 후 재핀치를 요구하지 않는다."""
        r=GestureRecognizer(self.config)
        c=UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())
        c.settings.mapping['PINCH']='draw'  # Explicit legacy preset under test
        with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=.7):
            for t,gap in ((0,4),(.2,1.4),(.4,3.5),(.6,1.4)):
                result=r.update(pair(gap),640,480,t)
                c.update(from_gesture_result(result,(640,480)),t)
            result=r.update([hand(100)],640,480,.64)
            c.update(from_gesture_result(result,(640,480)),.64)
            self.assertTrue(result.debug['pinch_armed'])
        with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=.1):
            for t in (.68,.73,.78,.83):
                result=r.update([hand(100)],640,480,t)
                packet=c.update(from_gesture_result(result,(640,480)),t)
        self.assertTrue(packet.active)
        self.assertEqual(packet.tool,'highlighter')
        self.assertEqual(packet.commands,())

    def test_guard_without_observed_release_still_requires_open(self):
        """양손 가림 후 손을 벌린 증거가 없으면 임의로 필기를 시작하지 않는다."""
        r=GestureRecognizer(self.config)
        with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.pinch_ratio',return_value=.1):
            r.update(pair(4),640,480,0)
            for t in (.03,.08,.15,.2,.25,.3):
                self.assertFalse(r.update([hand(100)],640,480,t).pen_down)


    def test_completed_clap_ends_previous_ink_and_switches_once(self):
        """직전 프레임이 필기여도 완료된 박수를 버리지 않고 획 종료 후 전환한다."""
        from airboard.ui.controller import InputState
        c=UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())
        c.settings.mapping['PINCH']='draw'  # Explicit legacy preset under test
        c.update(InputState((.5,.5),frozenset({'PINCH'})),0)
        packet=c.update(InputState((.5,.5),frozenset({'DOUBLE_CLAP'})),.1)
        self.assertEqual(packet.tool,'highlighter')
        self.assertFalse(packet.active)
        self.assertEqual(packet.phase,'up')
        self.assertEqual(packet.commands,())
        self.assertEqual(c.update(InputState((.5,.5)),.2).tool,'highlighter')

    def test_completed_clap_passes_passive_palette_but_not_settings(self):
        """팔레트를 스치는 커서는 박수 전환을 막지 않고 설정 편집은 차단한다."""
        from airboard.ui.controller import InputState
        state=InputState((.2,.03),frozenset({'DOUBLE_CLAP'}))
        c=UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())
        c.settings.mapping['PINCH']='draw'  # Explicit legacy preset under test
        packet=c.update(state,0,suppressed=True,navigation_suppressed=False)
        self.assertEqual(packet.tool,'highlighter')
        self.assertFalse(packet.active)
        c=UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())
        c.settings.mapping['PINCH']='draw'  # Explicit legacy preset under test
        self.assertEqual(c.update(state,0,suppressed=True,navigation_suppressed=True).tool,'pen')
