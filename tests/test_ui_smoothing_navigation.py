"""PDF 보정 연결과 도구/페이지 제스처 충돌 방지를 검사한다."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pymupdf
from airboard.render import RenderConfig
from airboard.ui.demo import PDFRenderer
from airboard.ui.controller import RenderPacket,InputState,UIController,from_gesture_result
from airboard.ui.settings import Settings
from airboard.calibration import Calibration
from airboard.gestures.gesture import SwipeDetector, PinchDetector


class SmoothingConnectionTests(unittest.TestCase):
    def test_tail_trim_is_saved_after_mid_stroke_checkpoint(self):
        """진행 획 저장 후 펜을 떼면 꼬리 제거 결과로 저장이 갱신된다."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/'source.pdf'
            with pymupdf.open() as doc:
                doc.new_page(width=320,height=180); doc.save(path)
            cfg=RenderConfig(320,180,1,'#FF0000',4,'#FFFF00',20,.3,30)
            options={'method':'none','start_skip':2,'end_trim':3}
            r=PDFRenderer(str(path),cfg,root/'state',options)
            bg=r.board.render_page()
            for i in range(10):
                r.consume(RenderPacket((20+i*10,70),'pen',True,'down' if i==0 else 'move','#FF0000',4,1.,()))
            self.assertFalse(np.array_equal(r.board.render_page()[70,100],bg[70,100]))
            # 긴 획 중 저장이 이미 수행된 상황을 만든다.
            r.checkpoint(force=True)
            r.consume(RenderPacket(None,'pen',False,'up','#FF0000',4,1.,()))
            np.testing.assert_array_equal(r.board.render_page()[70,100],bg[70,100])
            self.assertEqual(r.board.operation_count(),1)
            saved=PDFRenderer(str(path),cfg,root/'state',options)
            np.testing.assert_array_equal(saved.board.render_page(),r.board.render_page())
            saved.close(); r.close()

    def test_pdf_cursor_reduces_jitter(self):
        """실제 PDF 어댑터에 연결된 EMA 커서의 정지 떨림이 줄어든다."""
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.pdf'
            with pymupdf.open() as doc:
                doc.new_page(width=320,height=180); doc.save(path)
            cfg=RenderConfig(320,180,1,'#FF0000',4,'#FFFF00',20,.3,30)
            r=PDFRenderer(str(path),cfg,smoothing_config={'method':'ema','ema_alpha':.2,'start_skip':0,'end_trim':0})
            values=[]
            try:
                for i in range(40):
                    x=100+(-5 if i%2 else 5)
                    r.consume(RenderPacket((x,80),'pen',False,'hover','#FF0000',4,1.,()))
                    values.append(r.display_position[0])
                self.assertLess(np.std(values[10:]),2)
            finally:
                r.close()

    def test_smoothing_flush_preserves_highlighter_and_eraser(self):
        """보정 모드의 페이지 이동·종료 시 형광펜과 지우개 작업을 유실하지 않는다."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/'source.pdf'
            with pymupdf.open() as doc:
                doc.new_page(width=320,height=180)
                doc.new_page(width=320,height=180)
                doc.save(path)
            cfg=RenderConfig(320,180,1,'#FF0000',4,'#FFFF00',20,.3,30)
            r=PDFRenderer(str(path),cfg,root/'state',{'method':'none','start_skip':0,'end_trim':0})
            bg=r.board.render_page()
            for i,x in enumerate((30,100)):
                r.consume(RenderPacket((x,60),'highlighter',True,'down' if i==0 else 'move','#FFFF00',4,.3,()))
            r.change_page(1); r.change_page(-1)
            self.assertFalse(np.array_equal(r.board.render_page()[60,60],bg[60,60]))
            r.consume(RenderPacket((60,60),'eraser',True,'down','#FFFF00',4,1.,()))
            expected=r.board.render_page()
            r.close()
            r=PDFRenderer(str(path),cfg,root/'state',{'method':'none','start_skip':0,'end_trim':0})
            try:
                np.testing.assert_array_equal(r.board.render_page(),expected)
                np.testing.assert_array_equal(r.board.render_page()[60,60],bg[60,60])
                self.assertEqual(r.board.operation_count(),2)
            finally:
                r.close()


class NavigationTests(unittest.TestCase):
    def controller(self):
        return UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())

    def test_tool_transition_and_clap_block_page_event(self):
        """필기·포인터·박수 직후 페이지 이벤트를 무시하고 안정 후 허용한다."""
        c=self.controller()
        frame=lambda t,*g:c.update(InputState((.5,.5),frozenset(g)),t)
        frame(0,'PINCH'); frame(.1)
        self.assertEqual(frame(.2,'SWIPE_RIGHT').commands,())
        self.assertEqual(frame(.7,'DOUBLE_CLAP','SWIPE_RIGHT').commands,())
        self.assertEqual(frame(1.3,'SWIPE_RIGHT').commands,('page_next',))
        self.assertEqual(frame(1.4,'SWIPE_LEFT').commands,())
        self.assertEqual(frame(2.2,'SWIPE_LEFT','SWIPE_RIGHT').commands,())

    def test_two_hands_and_pointer_cannot_navigate(self):
        """인식기에서 페이지 이벤트가 섞여 와도 양손·포인터에서는 막는다."""
        c=self.controller()
        result=SimpleNamespace(hands=[object(),object()],pose='OPEN',pen_down=False,
                               pen_point=(50,50),pointer_point=(50,50),events=['SWIPE_RIGHT'])
        self.assertEqual(c.update(from_gesture_result(result,(100,100)),0).commands,())
        result.hands=[object()]; result.pose='POINTER'
        self.assertEqual(c.update(from_gesture_result(result,(100,100)),1).commands,())
        result.pose='OPEN'
        self.assertEqual(c.update(from_gesture_result(result,(100,100)),2).commands,('page_next',))

    def test_swipe_does_not_reuse_tool_movement(self):
        """이전 펼친 손과 도구 동작의 이동을 이어 스와이프로 판정하지 않는다."""
        detector=SwipeDetector()
        detector.update((10,50),True,100,100,0)
        detector.update((20,50),True,100,100,.1)
        self.assertIsNone(detector.update(None,False,100,100,.2))
        self.assertIsNone(detector.update((70,50),True,100,100,.3))
        self.assertIsNone(detector.update((80,50),True,100,100,.4))
        self.assertEqual(detector.update((95,50),True,100,100,.5),'SWIPE_RIGHT')
        self.assertIsNone(detector.update((50,50),True,100,100,.6))


    def test_relaxed_open_hand_event_is_not_discarded(self):
        """네 손가락 모두 OPEN이 아니어도 인식기가 승인한 스와이프를 허용한다."""
        c=self.controller()
        result=SimpleNamespace(hands=[object()],pose='NONE',pen_down=False,
                               pen_point=(50,50),pointer_point=(50,50),events=['SWIPE_RIGHT'])
        state=from_gesture_result(result,(100,100))
        self.assertTrue(state.navigation_allowed)
        self.assertEqual(c.update(state,0).commands,('page_next',))

    def test_short_open_hand_detection_gap_is_tolerated(self):
        """짧은 손가락 추정 흔들림은 무시하지만 그 프레임의 좌표는 쓰지 않는다."""
        d=SwipeDetector()
        d.update((10,50),True,100,100,0)
        d.update((18,50),True,100,100,.04)
        self.assertIsNone(d.update((80,50),False,100,100,.07))
        self.assertEqual(d.update((35,50),True,100,100,.1),'SWIPE_RIGHT')

    def test_swipe_after_short_quiet_window_is_delivered_once(self):
        """직전 도구 차단이 끝나기 전 발생한 단발 이벤트를 한 번 전달한다."""
        c=self.controller()
        c.update(InputState((.5,.5),frozenset({'PINCH'})),0)
        c.update(InputState((.5,.5)),.1)
        p=c.update(InputState((.5,.5),frozenset({'SWIPE_RIGHT'})),.2)
        self.assertEqual(p.commands,())
        self.assertEqual(c.update(InputState((.5,.5)),.3).commands,('page_next',))
        self.assertEqual(c.update(InputState((.5,.5)),.4).commands,())

    def test_passive_palette_does_not_swallow_swipe(self):
        """팔레트를 지나가는 것과 설정 모드 차단을 구별한다."""
        c=self.controller()
        state=InputState((.2,.03),frozenset({'SWIPE_RIGHT'}))
        self.assertEqual(c.update(state,0,suppressed=True,navigation_suppressed=False).commands,('page_next',))
        c=self.controller()
        self.assertEqual(c.update(state,0,suppressed=True,navigation_suppressed=True).commands,())

    def test_pending_page_is_cancelled_by_tool_or_hand_loss(self):
        """보관된 이벤트가 있더라도 필기 또는 손 소실이 끼면 취소한다."""
        for blocked in (InputState((.5,.5),frozenset({'PINCH'})),InputState(None,navigation_allowed=False)):
            c=self.controller()
            c.update(InputState((.5,.5),frozenset({'PINCH'})),0)
            c.update(InputState((.5,.5)),.1)
            c.update(InputState((.5,.5),frozenset({'SWIPE_RIGHT'})),.2)
            c.update(blocked,.22)
            self.assertEqual(c.update(InputState((.5,.5)),.8).commands,())

    def test_relaxed_finger_geometry_generates_page_event(self):
        """MCP 거리비가 작아도 PIP보다 펼친 손가락이 충분하면 인식한다."""
        from airboard.gestures.gesture import GestureRecognizer
        r=GestureRecognizer()
        hand=np.zeros((21,3),np.float32)
        hand[0]=(0,100,0); hand[9]=(0,65,0); hand[4]=(-60,0,0)
        for tip in (8,12,16,20):
            hand[tip]=(0,0,0); hand[tip-2]=(0,45,0)
        curls={name:1. for name in ('index','middle','ring','pinky')}
        with patch('airboard.gestures.gesture.classify_pose',return_value='NONE'),patch('airboard.gestures.gesture.finger_curls',return_value=curls):
            result=None
            for t,x in ((0,0),(.1,10),(.2,25)):
                sample=hand.copy(); sample[:,0]+=x
                result=r.update([sample],100,100,t)
        self.assertGreaterEqual(result.debug['swipe_open_count'],3)
        self.assertIn('SWIPE_RIGHT',result.events)

    def test_faster_swipe_profile_uses_shorter_distance(self):
        """UI 프로필은 짧은 이동을 인식하며 박수 차단 설정은 바꾸지 않는다."""
        import json
        from airboard.ui.settings import ROOT
        cfg=json.loads((ROOT/'config/default.json').read_text(encoding='utf-8'))['gestures']
        d=SwipeDetector(cfg)
        self.assertIsNone(d.update((10,50),True,100,100,0))
        self.assertIsNone(d.update((18,50),True,100,100,.05))
        self.assertEqual(d.update((26,50),True,100,100,.1),'SWIPE_RIGHT')
        self.assertIsNone(d.update((70,50),True,100,100,.2))
        self.assertIsNone(d.update((70,50),True,100,100,.6))
        self.assertIsNone(d.update((62,50),True,100,100,.65))
        self.assertEqual(d.update((54,50),True,100,100,.7),'SWIPE_LEFT')


class PinchReliabilityTests(unittest.TestCase):
    def detector(self):
        return PinchDetector({'pinch_on':.20,'pinch_off':.32,'pinch_on_frames':3,'pinch_off_frames':1})

    def test_brief_false_contact_is_ignored_and_release_is_fast(self):
        """순간 오인식은 무시하고 지속 접촉만 쓰며 손을 떼면 즉시 멈춘다."""
        d=self.detector()
        for ratio in (.1,.1,.4,.1,.4):
            self.assertFalse(d.update(ratio))
        self.assertFalse(d.update(.1)); self.assertFalse(d.update(.1))
        self.assertTrue(d.update(.1))
        self.assertTrue(d.update(.25))
        self.assertFalse(d.update(.4))

    def test_loss_or_fist_requires_open_before_new_pinch(self):
        """손 소실/주먹 뒤에는 손가락을 벌린 뒤 새로 접촉해야 쓴다."""
        for blocked in (False,True):
            with self.subTest(blocked=blocked):
                d=self.detector()
                d.update(None if not blocked else .1,blocked=blocked)
                for _ in range(5):
                    self.assertFalse(d.update(.1))
                self.assertFalse(d.update(.5))
                self.assertFalse(d.update(.1)); self.assertFalse(d.update(.1))
                self.assertTrue(d.update(.1))

    def test_rotating_pinch_distance_spikes_do_not_split_stroke(self):
        """로그의 일시적 해제 비율 상승 뒤 복귀를 지속 필기로 처리한다."""
        import json
        from airboard.ui.settings import ROOT
        cfg=json.loads((ROOT/'config/default.json').read_text(encoding='utf-8'))['gestures']
        d=PinchDetector(cfg)
        for _ in range(cfg['pinch_on_frames']):d.update(.1)
        self.assertTrue(d.down)
        for ratio in (.12,.15,.35,.13,.46,.16,.53,.24):
            self.assertTrue(d.update(ratio))
        self.assertTrue(d.update(.5))
        self.assertFalse(d.update(.5))


class HandwritingEndpointTests(unittest.TestCase):
    def test_ui_handwriting_profile_does_not_cut_stroke_ends(self):
        """UI 필기 설정은 짧은 획의 시작과 끝을 자동 삭제하지 않는다."""
        options=dict(Settings().ui['handwriting_smoothing'],method='none')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.pdf'
            with pymupdf.open() as doc:
                doc.new_page(width=320,height=180); doc.save(path)
            r=PDFRenderer(str(path),RenderConfig(320,180,1,'#FF0000',4,'#FFFF00',20,.3,30),smoothing_config=options)
            bg=r.board.render_page()
            try:
                for i,x in enumerate((30,60,90,120)):
                    r.consume(RenderPacket((x,60),'pen',True,'down' if i==0 else 'move','#FF0000',4,1.,()))
                r.consume(RenderPacket(None,'pen',False,'up','#FF0000',4,1.,()))
                image=r.board.render_page()
                self.assertFalse(np.array_equal(image[60,30],bg[60,30]))
                self.assertFalse(np.array_equal(image[60,120],bg[60,120]))
            finally:
                r.close()
