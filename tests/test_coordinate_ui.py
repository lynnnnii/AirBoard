"""카메라/창 없이 좌표·UI 상태 및 팀 인식 어댑터를 검증한다."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from airboard.calibration import Calibration
from airboard.ui.settings import Settings, ROOT
from airboard.ui.controller import UIController, InputState, from_gesture_result
from airboard.ui.overlay import Overlay
from airboard.ui.demo import PreviewRenderer, compose_screen, draw_settings, reset_calibration


class CoordinateTests(unittest.TestCase):
    """독립적인 기하 기준으로 호모그래피를 확인한다."""

    def test_projective_corners_and_center(self) -> None:
        c = Calibration([[0,0],[.5,0],[1,1],[0,1]],(201,151))
        for p,wanted in zip(c.corners,[(0,0),(200,0),(200,150),(0,150)]):
            np.testing.assert_allclose(c.transform(p),wanted,atol=1e-4)
        np.testing.assert_allclose(c.transform((1/3,1/3)),(100,75),atol=1e-4)

    def test_bad_geometry(self) -> None:
        for points in ([[0,0]]*4,[[0,0],[1,1],[1,0],[0,1]],
                       [[0,0],[1,0],[.2,.2],[0,1]],[[0,0],[1,0],[1,float('nan')],[0,1]]):
            with self.subTest(points=points),self.assertRaises(ValueError):
                Calibration(points)

    def test_outside_and_loss(self) -> None:
        c = Calibration([[.1,.1],[.9,.1],[.9,.9],[.1,.9]])
        self.assertIsNone(c.transform((0,0)))
        self.assertIsNone(c.transform(None))
        self.assertIsNone(c.transform((float('inf'),.5)))

    def test_persistence(self) -> None:
        c = Calibration([[0,0],[1,0],[1,1],[0,1]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cal.json'
            c.save(path)
            np.testing.assert_allclose(Calibration.load(path).matrix,c.matrix)


class StateTests(unittest.TestCase):
    """획 경계·도구 우선순위·일회성 이벤트 검증."""

    def setUp(self) -> None:
        self.settings = Settings()
        self.c = UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),self.settings)

    def frame(self, *gestures: str, position: tuple | None = (.5,.5), suppressed: bool = False, now: float = 1.) -> object:
        """동일 시각 기준의 테스트 프레임."""
        return self.c.update(InputState(position,frozenset(gestures)),now,suppressed)

    def test_strokes_and_tracking_loss(self) -> None:
        self.assertEqual(self.frame().phase,'hover')
        self.assertEqual(self.frame('POINTER').phase,'down')
        self.assertEqual(self.frame('POINTER').phase,'move')
        self.assertEqual(self.frame(position=None).phase,'up')
        self.assertEqual(self.frame('POINTER').phase,'down')
        self.assertEqual(self.frame('POINTER',position=(2,2)).phase,'up')
        self.assertEqual(self.frame('POINTER').phase,'down')

    def test_fist_priority_restore_and_repinch(self) -> None:
        self.frame('POINTER')
        for _ in range(self.settings.ui['tool_hold_frames']):
            p = self.frame('FIST','POINTER')
        self.assertEqual(p.tool,'eraser')
        self.assertTrue(self.frame('FIST','POINTER').active)
        self.assertFalse(self.frame('POINTER').active)
        self.assertEqual(self.c.tool,'pen')
        self.assertFalse(self.frame('POINTER').active)
        self.frame()
        self.assertTrue(self.frame('POINTER').active)

    def test_event_no_hand_and_page_guard(self) -> None:
        self.assertEqual(self.frame('DOUBLE_CLAP',position=None).tool,'highlighter')
        self.assertEqual(self.frame().tool,'highlighter')
        self.frame('POINTER')
        self.assertEqual(self.frame('POINTER','SWIPE_DOWN').commands,())
        self.frame()
        self.assertEqual(self.frame('SWIPE_DOWN',now=2.).commands,('page_next',))
        self.assertEqual(self.frame().commands,())

    def test_pointer_and_suppression(self) -> None:
        for _ in range(self.settings.ui['tool_hold_frames']):
            p = self.frame('POINTER','POINTER')
        self.assertEqual(p.tool,'pen')
        self.assertTrue(p.active)
        self.frame()
        self.frame('POINTER')
        self.assertEqual(self.frame('POINTER',suppressed=True).phase,'up')

    def test_custom_mapping(self) -> None:
        self.settings.mapping = {'WAVE':'toggle_highlighter','TOUCH':'draw'}
        self.assertEqual(self.frame('DOUBLE_CLAP').tool,'pen')
        self.assertEqual(self.frame('WAVE').tool,'highlighter')
        self.assertTrue(self.frame('TOUCH').active)

    def test_held_toggle_requires_new_gesture(self) -> None:
        """형광펜 토글은 유지 중 한 번만 적용되고 새 동작으로 펜에 복귀한다."""
        self.settings.mapping['POINTER'] = 'toggle_highlighter'
        self.assertEqual(self.frame('POINTER').tool, 'highlighter')
        self.assertEqual(self.frame('POINTER').tool, 'highlighter')
        self.frame()
        self.assertEqual(self.frame('POINTER').tool, 'pen')

    def test_settings_panel_preserves_mapping(self) -> None:
        """설정 화면은 매핑을 바꾸지 않고 별도 배경에 안내를 표시한다."""
        image = np.zeros((540,960,3),np.uint8)
        before = self.settings.mapping.copy()
        draw_settings(image,self.settings.mapping,0)
        self.assertEqual(self.settings.mapping,before)
        np.testing.assert_array_equal(image[0,0],[32,32,32])
        self.assertTrue(np.any(image[:150] == 255))

    def test_reset_calibration_persists_and_requires_lift(self) -> None:
        """리셋 후 기본 꼭짓점·현재 크기를 저장하고 누른 채 필기 재개를 막는다."""
        self.c.calibration = Calibration([[0,0],[1,0],[1,1],[0,1]],(1280,720))
        self.frame('POINTER')
        renderer = PreviewRenderer((1280,720))
        renderer.previous = (10,10)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cal.json'
            reset_calibration(self.c,renderer,self.settings.ui['default_corners'],path)
            saved = Calibration.load(path)
            np.testing.assert_allclose(saved.corners,self.settings.ui['default_corners'])
            self.assertEqual(saved.size,(1280,720))
        self.assertIsNone(renderer.previous)
        self.assertFalse(self.frame('POINTER').active)
        self.frame()
        self.assertTrue(self.frame('POINTER').active)

    def test_footer_is_outside_ink(self) -> None:
        """필기와 카메라를 보존하고 별도의 불투명 상태줄을 덧붙인다."""
        camera = np.full((480,640,3), 80, np.uint8)
        slide = np.full((540,960,3), 255, np.uint8)
        screen = compose_screen(camera, slide, 2, 'pen', 'NEXT: 1 -> 2')
        self.assertEqual(screen.shape, (618,1600,3))
        np.testing.assert_array_equal(screen[:480,:640], camera)
        np.testing.assert_array_equal(screen[:540,640:], slide)
        np.testing.assert_array_equal(screen[540,0], [32,32,32])
        self.assertTrue(np.any(screen[540:] == 255))

    def test_team_result_adapter(self) -> None:
        result = SimpleNamespace(hands=[object()],pose='POINTER',pen_down=True,pen_point=(320,240),pointer_point=(400,200),events=['DOUBLE_CLAP'])
        state = from_gesture_result(result,(641,481))
        self.assertEqual(state.position,(.625,200/480))
        self.assertEqual(state.gestures,frozenset({'POINTER','DOUBLE_CLAP'}))
        result.pose = 'FIST'
        self.assertEqual(from_gesture_result(result,(641,481)).position,(.625,200/480))

    def test_palette_dwell_single_fire_and_reset(self) -> None:
        overlay = Overlay(self.settings)
        self.assertIsNone(overlay.dwell((20,20),0))
        self.assertEqual(overlay.dwell((20,20),1),('color',self.settings.ui['colors'][0]))
        self.assertIsNone(overlay.dwell((20,20),2))
        overlay.dwell(None,3)
        self.assertIsNone(overlay.dwell((20,20),4))
        self.assertIsNotNone(overlay.dwell((20,20),5))

    def test_render_and_overlay(self) -> None:
        renderer,overlay = PreviewRenderer((960,540)),Overlay(self.settings)
        renderer.consume(self.frame('POINTER'))
        image = renderer.consume(self.frame('POINTER'))
        self.assertTrue(np.any(image!=255))
        overlay.draw(image,self.frame('POINTER'),'pen')
        self.assertEqual(image.shape,(540,960,3))


class SettingsTests(unittest.TestCase):
    """설정 JSON 검증과 사용자 설정 우선 로드."""

    def test_empty_user_mapping_falls_back_and_cannot_be_saved(self) -> None:
        """빈 개인 매핑은 기본값으로 복구하고 빈 매핑 저장은 거부한다."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'config').mkdir()
            for name in ('default.json','gestures.json'):
                (root/'config'/name).write_bytes((ROOT/'config'/name).read_bytes())
            (root/'config/ui-user.json').write_text(json.dumps({'mapping':{},'ui':{}}),encoding='utf-8')
            settings = Settings(root)
            expected = json.loads((root/'config/gestures.json').read_text(encoding='utf-8'))
            self.assertEqual(settings.mapping,expected)
            settings.mapping = {}
            with self.assertRaises(ValueError):
                settings.save()
            (root/'config/gestures.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'gestures.json'):
                Settings(root)

    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'config').mkdir()
            for name in ('default.json','gestures.json'):
                (root/'config'/name).write_text((ROOT/'config'/name).read_text(encoding='utf-8'),encoding='utf-8')
            s = Settings(root)
            s.mapping['DOUBLE_CLAP'] = 'none'
            s.save()
            self.assertEqual(Settings(root).mapping['DOUBLE_CLAP'],'none')
            s.mapping['FIST'] = 'INVALID'
            with self.assertRaises(ValueError):
                s.save()


if __name__=='__main__':
    unittest.main()
