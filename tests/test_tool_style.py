"""형광펜 고정 색, 펜 팔레트, 포인터 커서 표시를 검사한다."""
import os
os.environ.setdefault('AIRBOARD_USER_SETTINGS', os.path.join(os.path.dirname(__file__), '_no_user_settings.json'))  # 개인 설정 없이 팀 기본값으로 검사
import unittest

import numpy as np

from airboard.calibration import Calibration
from airboard.ui.controller import InputState, UIController
from airboard.ui.overlay import draw_sparkle
from airboard.ui.settings import Settings


def gesture_cfg():
    """config/default.json의 gestures 설정."""
    import json
    from airboard.ui.settings import ROOT
    return json.loads((ROOT / 'config/default.json').read_text(encoding='utf-8'))['gestures']


def feed(recognizer, curls, tips, ratios, pose='NONE', xs=None):
    """손가락 수치와 엄지-검지 비율을 고정한 합성 손을 프레임마다 넣는다."""
    from unittest.mock import patch
    out = []
    for i, ratio in enumerate(ratios):
        hand = np.zeros((21, 3), np.float32)
        hand[:, 0] = xs[i] if xs else 100
        hand[:, 1] = 50
        hand[0, 1] = 80
        with patch('airboard.gestures.gesture.classify_pose', return_value=pose),                 patch('airboard.gestures.gesture.pinch_ratio', return_value=ratio),                 patch('airboard.gestures.gesture.finger_curls', return_value=curls),                 patch('airboard.gestures.gesture.tip_to_palm', return_value=tips):
            out.append(recognizer.update([hand], 640, 480, len(out) * .05 + getattr(recognizer, '_t', 0)))
    recognizer._t = getattr(recognizer, '_t', 0) + len(ratios) * .05 + .05
    return out


class ToolStyleTests(unittest.TestCase):
    def test_highlighter_is_always_yellow(self):
        settings = Settings()
        c = UIController(Calibration([[0, 0], [1, 0], [1, 1], [0, 1]]), settings)
        c.color = settings.ui['colors'][1]          # 펜 색을 빨강으로 골라도
        c.update(InputState((.5, .5), frozenset({'DOUBLE_CLAP'})), 0)
        packet = c.update(InputState((.5, .5)), .1)
        self.assertEqual(packet.tool, 'highlighter')
        self.assertEqual(packet.color, '#FFFF00')

    def test_pen_palette_has_no_green(self):
        self.assertNotIn('#16a34a', [x.lower() for x in Settings().ui['colors']])

    def test_writing_is_scaled_around_word_start(self):
        settings = Settings()
        settings.ui['writing_scale'] = .5
        c = UIController(Calibration([[0, 0], [1, 0], [1, 1], [0, 1]], (101, 101)), settings)
        pinch = frozenset({'PINCH'})
        np.testing.assert_allclose(c.update(InputState((.2, .2), pinch), 0).position, (20, 20), atol=1e-3)
        np.testing.assert_allclose(c.update(InputState((.4, .2), pinch), .1).position, (30, 20), atol=1e-3)   # 20px 이동 → 10px
        c.update(InputState((.4, .2)), .2)                                              # 펜 들기
        hover = c.update(InputState((.6, .2)), .4)
        np.testing.assert_allclose(hover.position, (40, 20), atol=1e-3)       # 같은 단어 안: 다음 획 위치도 줄어든 좌표로 보인다
        second = c.update(InputState((.6, .2), pinch), .5)
        np.testing.assert_allclose(second.position, (40, 20), atol=1e-3)      # 다음 획도 같은 기준점에서 줄어든다
        c.update(InputState((.6, .2)), .6)
        np.testing.assert_allclose(c.update(InputState((.8, .2)), 2.).position, (80, 20), atol=1e-3)   # 쉬면 실제 위치로 돌아온다

    def test_cursor_glides_back_instead_of_jumping(self):
        settings = Settings()
        settings.ui.update(writing_scale=.5, writing_anchor_seconds=1., writing_return_seconds=.4)
        c = UIController(Calibration([[0, 0], [1, 0], [1, 1], [0, 1]], (101, 101)), settings)
        c.update(InputState((.2, .2), frozenset({'PINCH'})), 0)
        c.update(InputState((.6, .2), frozenset({'PINCH'})), .1)   # 표시 40, 실제 60
        c.update(InputState((.6, .2)), .2)
        xs = [c.update(InputState((.6, .2)), t).position[0] for t in (1.0, 1.1, 1.3, 1.5)]
        np.testing.assert_allclose(xs, (40, 40, 50, 60), atol=1e-3)  # 마지막 필기 1초 뒤부터 0.4초 동안 40 → 60

    def test_highlighter_keeps_width_and_real_position(self):
        settings = Settings()
        settings.ui['writing_scale'] = .5
        c = UIController(Calibration([[0, 0], [1, 0], [1, 1], [0, 1]], (101, 101)), settings)
        c.update(InputState((.5, .5), frozenset({'DOUBLE_CLAP'})), 0)
        c.update(InputState((.2, .2), frozenset({'PINCH'})), .1)
        hl = c.update(InputState((.6, .2), frozenset({'PINCH'})), .2)
        self.assertEqual(hl.tool, 'highlighter')
        np.testing.assert_allclose(hl.position, (60, 20), atol=1e-3)    # 형광펜은 줄이지 않는다
        self.assertEqual(hl.width, settings.ui['highlighter_width'])
        self.assertLess(c.width, hl.width)

    def test_sparkle_stays_local_and_visible_on_white(self):
        image = np.full((200, 200, 3), 255, np.uint8)
        draw_sparkle(image, (100, 100), 0.)
        changed = (image != 255).any(2)
        self.assertTrue(changed[100, 105])          # 가운데 흰 점 바로 옆의 빨간 테
        self.assertFalse(changed[:50].any() or changed[150:].any())
        draw_sparkle(image, (0, 199), 0.)           # 화면 가장자리에서도 오류 없이 그린다


class HighlighterSmoothingTests(unittest.TestCase):
    def test_highlighter_ink_is_unfiltered_and_pen_shows_live_tail(self):
        import tempfile
        from pathlib import Path
        import pymupdf
        from airboard.render import RenderConfig
        from airboard.ui.controller import RenderPacket
        from airboard.ui.demo import PDFRenderer
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.pdf'
            with pymupdf.open() as doc:
                doc.new_page(width=320, height=180); doc.save(path)
            options = {**Settings().ui['handwriting_smoothing'], 'one_euro_min_cutoff': .05, 'one_euro_beta': 0.}
            r = PDFRenderer(str(path), RenderConfig(320, 180, 1, '#FF0000', 4, '#FFFF00', 16, .3, 30), smoothing_config=options)
            try:
                bg = r.board.render_page()
                hl = lambda x, ph: RenderPacket((x, 60), 'highlighter', True, ph, '#FFFF00', 16, .3, ())
                r.consume(hl(20, 'down')); r.consume(hl(200, 'move'))
                self.assertEqual(r.display_position, (200., 60.))         # 강한 보정이어도 형광펜은 손 위치 그대로
                pen = lambda x, ph: RenderPacket((x, 120), 'pen', True, ph, '#0000FF', 2, 1., ())
                for i, x in enumerate((20, 60, 100)):
                    image = r.consume(pen(x, 'down' if i == 0 else 'move'))
                self.assertFalse(np.array_equal(image[120, int(r.display_position[0]) - 1], bg[120, 1]))  # 커서까지 선이 이어져 보인다
            finally:
                r.close()


class CustomMappingTests(unittest.TestCase):
    """설정 화면에서 바꾼 제스처 매핑이 실제로 동작하는지 검사한다."""

    def controller(self, **mapping):
        settings = Settings()
        settings.mapping.update(mapping)
        return UIController(Calibration([[0, 0], [1, 0], [1, 1], [0, 1]], (100, 100)), settings)

    def test_clap_mapped_to_page_turns_page_with_two_hands(self):
        c = self.controller(DOUBLE_CLAP='page_next')
        c.update(InputState((.5, .5), navigation_allowed=False), 0)
        p = c.update(InputState((.5, .5), frozenset({'DOUBLE_CLAP'}), navigation_allowed=False), .1)
        self.assertEqual(p.commands, ('page_next',))
        self.assertEqual(c.update(InputState(None, frozenset({'DOUBLE_CLAP'}), navigation_allowed=False), .2).commands, ())  # 재입력 대기

    def test_clap_page_is_ignored_while_writing(self):
        c = self.controller(DOUBLE_CLAP='page_next')
        c.update(InputState((.5, .5), frozenset({'PINCH'})), 0)
        self.assertEqual(c.update(InputState((.5, .5), frozenset({'PINCH', 'DOUBLE_CLAP'})), .1).commands, ())

    def test_index_drawing_survives_brief_pose_flicker(self):
        c = self.controller(POINTER='draw', PINCH='none')
        c.writing_scale = 1.
        point = frozenset({'POINTER'})
        c.update(InputState((.2, .5), point), 0)
        c.update(InputState((.3, .5), point), .1)
        flicker = c.update(InputState((.35, .5)), .2)          # 한 프레임 자세 판정이 빠짐
        self.assertTrue(flicker.active)
        np.testing.assert_allclose(flicker.position, (29.7, 49.5), atol=.01)   # 그동안 잉크는 직전 위치(.3, .5)에 멈춤
        resumed = c.update(InputState((.4, .5), point), .3)
        self.assertEqual(resumed.phase, 'move')                # 같은 획으로 이어 쓴다
        c.update(InputState((.4, .5)), .4)
        self.assertFalse(c.update(InputState((.4, .5)), .7).active)   # 유예(0.2초)가 지나면 획 종료

    def test_far_move_during_flicker_starts_new_stroke(self):
        c = self.controller(POINTER='draw', PINCH='none')
        c.writing_scale = 1.
        c.settings.ui['draw_resume_break'] = 40
        point = frozenset({'POINTER'})
        c.update(InputState((.2, .5), point), 0)
        c.update(InputState((.2, .5)), .1)
        jumped = c.update(InputState((.8, .5), point), .2)
        self.assertEqual(jumped.phase, 'up')                   # 멀리서(60px > 40px) 다시 잡히면 선으로 잇지 않는다
        self.assertEqual(c.update(InputState((.8, .5), point), .3).phase, 'down')

    def test_recognizer_keeps_index_pose_with_bent_finger(self):
        from unittest.mock import patch
        from airboard.gestures.gesture import GestureRecognizer
        import json
        from airboard.ui.settings import ROOT
        cfg = json.loads((ROOT / 'config/default.json').read_text(encoding='utf-8'))['gestures']
        hand = np.zeros((21, 3), np.float32)
        hand[:, 1] = 50
        hand[0, 1] = 80
        bent = {'index': 1.4, 'middle': 1., 'ring': 1., 'pinky': 1.}   # 검지가 살짝 굽음 (처음 기준 1.55 미만)
        folded = {'index': 1.2, 'middle': .7, 'ring': .7, 'pinky': .7}
        for keep in (True, False):
            r = GestureRecognizer({**cfg, 'index_writing': keep})   # 핀치로 쓰는 설정에서는 유지하지 않는다
            with patch('airboard.gestures.gesture.pinch_ratio', return_value=.25), \
                    patch('airboard.gestures.gesture.tip_to_palm', return_value=folded), \
                    patch('airboard.gestures.gesture.finger_curls', return_value=bent):
                with patch('airboard.gestures.gesture.classify_pose', return_value='POINTER'):
                    r.update([hand], 640, 480, 0)
                with patch('airboard.gestures.gesture.classify_pose', return_value='NONE'):
                    pose = r.update([hand], 640, 480, .1).pose
            self.assertEqual(pose, 'POINTER' if keep else 'NONE')

    def test_two_fingers_stop_writing_immediately(self):
        c = self.controller(POINTER='draw', PINCH='none')
        c.writing_scale = 1.
        point, two = frozenset({'POINTER'}), frozenset({'TWO_FINGERS'})
        c.update(InputState((.2, .5), point), 0)
        c.update(InputState((.3, .5), point), .1)
        stopped = c.update(InputState((.35, .5), two), .15)
        self.assertEqual(stopped.phase, 'up')                 # 흔들림 유예 없이 그 프레임에 펜을 든다
        moving = c.update(InputState((.6, .5), two), .2)
        self.assertFalse(moving.active)                       # 두 손가락으로 옮기는 동안 안 써진다
        self.assertIsNotNone(moving.position)                 # 커서는 따라온다
        self.assertEqual(c.update(InputState((.6, .5), point), .25).phase, 'down')   # 중지를 접으면 새 획

    def test_recognizer_two_fingers_only_when_enabled_and_never_blocks_pinch(self):
        from airboard.gestures.gesture import GestureRecognizer
        curls = {'index': 1.8, 'middle': 1.7, 'ring': 1., 'pinky': 1.}
        tips = {'index': 1.5, 'middle': 1.4, 'ring': .6, 'pinky': .6}
        for flags, expect_two in (({'index_writing': True}, True), ({'two_fingers': True}, True), ({}, False)):
            r = GestureRecognizer({**gesture_cfg(), **flags})
            poses = [x.pose for x in feed(r, curls, tips, [.6] * 3)]          # 엄지는 떨어져 있음
            self.assertEqual('TWO_FINGERS' in poses, expect_two, flags)
            after = feed(r, curls, tips, [.1] * 4)                           # 같은 손 모양에서 엄지를 검지에 붙이면
            self.assertTrue(after[-1].pen_down, flags)                       # 막히지 않고 핀치가 된다

    def test_index_writing_switches_to_pinch_without_reopening(self):
        from airboard.gestures.gesture import GestureRecognizer
        r = GestureRecognizer({**gesture_cfg(), 'index_writing': True, 'pinch_writing': False})
        bent = {'index': 1.4, 'middle': 1., 'ring': 1., 'pinky': 1.}
        folded = {'index': 1.2, 'middle': .7, 'ring': .7, 'pinky': .7}
        feed(r, bent, folded, [.25], pose='POINTER')
        kept = feed(r, bent, folded, [.25, .25])
        self.assertEqual([x.pose for x in kept], ['POINTER', 'POINTER'])      # 검지가 살짝 굽어도 유지
        pinch = feed(r, bent, folded, [.12] * 3)                             # 엄지를 붙이면
        self.assertEqual(pinch[-1].pose, 'PINCH')                            # 손을 벌렸다 다시 붙이지 않아도 핀치(지우개)로

    def test_pinch_lift_freeze_only_when_pinch_draws(self):
        from airboard.gestures.gesture import GestureRecognizer
        open_ = {'index': 1., 'middle': 1., 'ring': 1., 'pinky': 1.}
        tips = {'index': .9, 'middle': .6, 'ring': .6, 'pinky': .6}
        for pinch_writing, frozen in ((True, True), (False, False)):
            r = GestureRecognizer({**gesture_cfg(), 'pinch_writing': pinch_writing})
            out = feed(r, open_, tips, [.1, .1, .1, .3, .3], xs=[100, 100, 100, 150, 200])
            xs = [o.pen_point[0] for o in out if o.pen_down]
            if frozen:                                                       # 핀치로 쓸 때: 살짝 벌리면 잉크 멈춤 (원래대로)
                self.assertEqual(xs, [100, 100, 100])
            else:                                                            # 핀치가 지우개일 때: 위치가 손을 따라간다
                self.assertEqual(xs, [100, 150, 200])

    def test_index_writes_right_after_eraser_or_pointer(self):
        c = self.controller(POINTER='draw', PINCH='eraser', FIST='pointer')
        for tool_gesture in ('PINCH', 'FIST'):
            for i in range(3):
                c.update(InputState((.5, .5), frozenset({tool_gesture})), i * .1)
            c.update(InputState((.5, .5), frozenset({'POINTER'})), .4)
            self.assertTrue(c.update(InputState((.5, .5), frozenset({'POINTER'})), .5).active, tool_gesture)
            c.update(InputState((.5, .5)), 2.)

    def test_pinch_writing_still_waits_for_release_after_fist(self):
        c = self.controller()                                  # 기본 매핑: PINCH=draw, FIST=eraser
        for i in range(3):
            c.update(InputState((.5, .5), frozenset({'FIST'})), i * .1)
        self.assertFalse(c.update(InputState((.5, .5), frozenset({'PINCH'})), .4).active)
        c.update(InputState((.5, .5)), .5)
        self.assertTrue(c.update(InputState((.5, .5), frozenset({'PINCH'})), .6).active)

    def test_pinch_release_is_not_delayed_by_grace(self):
        c = self.controller()
        c.update(InputState((.5, .5), frozenset({'PINCH'})), 0)
        self.assertEqual(c.update(InputState((.5, .5)), .1).phase, 'up')


class SettingsSaveTests(unittest.TestCase):
    def test_save_keeps_only_changed_ui_values(self):
        import json
        import shutil
        import tempfile
        from pathlib import Path
        from airboard.ui.settings import ROOT
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config').mkdir()
            for name in ('default.json', 'gestures.json'):
                shutil.copy(ROOT / 'config' / name, root / 'config' / name)
            s = Settings(root)
            s.mapping['DOUBLE_CLAP'] = 'page_next'
            s.ui['dwell_seconds'] = 1.
            s.save()
            saved = json.loads((root / 'config/ui-user.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['ui'], {'dwell_seconds': 1.})
            self.assertEqual(saved['mapping']['DOUBLE_CLAP'], 'page_next')
            default = json.loads((root / 'config/default.json').read_text(encoding='utf-8'))
            default['ui']['writing_scale'] = .8                # 나중에 팀 기본값을 바꾸면
            (root / 'config/default.json').write_text(json.dumps(default), encoding='utf-8')
            self.assertEqual(Settings(root).ui['writing_scale'], .8)   # 개인 파일이 덮어쓰지 않는다


class MappingRuleTests(unittest.TestCase):
    """설정 화면에서 고를 수 있는 조합 규칙."""

    def test_one_shot_and_held_gestures_offer_only_meaningful_actions(self):
        from airboard.ui.settings import allowed_actions
        for g in ('DOUBLE_CLAP', 'SWIPE_LEFT', 'SWIPE_RIGHT'):
            self.assertEqual(set(allowed_actions(g)), {'toggle_highlighter', 'page_next', 'page_prev', 'undo', 'redo', 'none'})
        for g in ('PINCH', 'FIST', 'POINTER', 'TWO_FINGERS'):
            self.assertNotIn('page_next', allowed_actions(g))
            self.assertIn('draw', allowed_actions(g))

    def test_multiple_draw_gestures_and_no_draw_can_be_saved(self):
        import shutil, tempfile
        from pathlib import Path
        from airboard.ui.settings import ROOT
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config').mkdir()
            for name in ('default.json', 'gestures.json'):
                shutil.copy(ROOT / 'config' / name, root / 'config' / name)
            s = Settings(root)
            s.set_action('POINTER', 'draw')
            self.assertEqual(s.mapping['POINTER'], 'draw')
            self.assertEqual(s.mapping['PINCH'], 'draw')        # 기존 draw 설정도 유지
            with self.assertRaises(ValueError):
                s.set_action('DOUBLE_CLAP', 'eraser')           # 순간형에는 지우개를 연결할 수 없다
            s.save()
            self.assertEqual(Settings(root).mapping['POINTER'], 'draw')
            self.assertEqual(Settings(root).mapping['PINCH'], 'draw')
            s.set_action('POINTER', 'none')
            s.set_action('PINCH', 'none')
            s.save()                                        # draw가 없어도 저장 가능
            self.assertNotIn('draw', Settings(root).mapping.values())

    def test_old_personal_file_with_invalid_combos_is_repaired(self):
        import json, shutil, tempfile
        from pathlib import Path
        from airboard.ui.settings import ROOT
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config').mkdir()
            for name in ('default.json', 'gestures.json'):
                shutil.copy(ROOT / 'config' / name, root / 'config' / name)
            (root / 'config/ui-user.json').write_text(json.dumps({'mapping': {
                'FIST': 'page_next', 'DOUBLE_CLAP': 'eraser', 'POINTER': 'draw'}, 'ui': {}}), encoding='utf-8')
            s = Settings(root)                                  # 오류 없이 열린다
            self.assertEqual(s.mapping['FIST'], 'eraser')       # 기본값으로
            self.assertEqual(s.mapping['DOUBLE_CLAP'], 'toggle_highlighter')
            self.assertEqual(list(s.mapping.values()).count('draw'), 2)
            s.save()
            self.assertEqual(list(Settings(root).mapping.values()).count('draw'), 2)


if __name__ == '__main__':
    unittest.main()
