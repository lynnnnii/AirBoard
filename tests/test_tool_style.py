"""Test fixed highlighter color, pen palette, and pointer cursor display."""
import os
os.environ.setdefault('AIRBOARD_USER_SETTINGS', os.path.join(os.path.dirname(__file__), '_no_user_settings.json'))  # Test with team defaults instead of personal settings
import unittest

import numpy as np

from airboard.calibration import Calibration
from airboard.ui.controller import InputState, UIController
from airboard.ui.overlay import draw_sparkle
from airboard.ui.settings import Settings


def gesture_cfg():
    """Gesture settings from config/default.json."""
    import json
    from airboard.ui.settings import ROOT
    return json.loads((ROOT / 'config/default.json').read_text(encoding='utf-8'))['gestures']


def feed(recognizer, curls, tips, ratios, pose='NONE', xs=None):
    """Feed synthetic hands with fixed finger metrics and thumb-index ratio each frame."""
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
        c.color = settings.ui['colors'][1]          # Even if the pen color is set to red
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
        np.testing.assert_allclose(c.update(InputState((.4, .2), pinch), .1).position, (30, 20), atol=1e-3)   # 20 px motion becomes 10 px
        c.update(InputState((.4, .2)), .2)                                              # Lift the pen
        hover = c.update(InputState((.6, .2)), .4)
        np.testing.assert_allclose(hover.position, (40, 20), atol=1e-3)       # Within the same word, the next stroke is previewed at the scaled position
        second = c.update(InputState((.6, .2), pinch), .5)
        np.testing.assert_allclose(second.position, (40, 20), atol=1e-3)      # The next stroke is also scaled around the same anchor
        c.update(InputState((.6, .2)), .6)
        np.testing.assert_allclose(c.update(InputState((.8, .2)), 2.).position, (80, 20), atol=1e-3)   # After a pause, return to the real hand position

    def test_cursor_glides_back_instead_of_jumping(self):
        settings = Settings()
        settings.ui.update(writing_scale=.5, writing_anchor_seconds=1., writing_return_seconds=.4)
        c = UIController(Calibration([[0, 0], [1, 0], [1, 1], [0, 1]], (101, 101)), settings)
        c.update(InputState((.2, .2), frozenset({'PINCH'})), 0)
        c.update(InputState((.6, .2), frozenset({'PINCH'})), .1)   # Displayed at 40, actual hand at 60
        c.update(InputState((.6, .2)), .2)
        xs = [c.update(InputState((.6, .2)), t).position[0] for t in (1.0, 1.1, 1.3, 1.5)]
        np.testing.assert_allclose(xs, (40, 40, 50, 60), atol=1e-3)  # Move from 40 to 60 over 0.4 s after a one-second pause

    def test_highlighter_keeps_width_and_real_position(self):
        settings = Settings()
        settings.ui['writing_scale'] = .5
        c = UIController(Calibration([[0, 0], [1, 0], [1, 1], [0, 1]], (101, 101)), settings)
        c.update(InputState((.5, .5), frozenset({'DOUBLE_CLAP'})), 0)
        c.update(InputState((.2, .2), frozenset({'PINCH'})), .1)
        hl = c.update(InputState((.6, .2), frozenset({'PINCH'})), .2)
        self.assertEqual(hl.tool, 'highlighter')
        np.testing.assert_allclose(hl.position, (60, 20), atol=1e-3)    # Do not scale the highlighter
        self.assertEqual(hl.width, settings.ui['highlighter_width'])
        self.assertLess(c.width, hl.width)

    def test_sparkle_stays_local_and_visible_on_white(self):
        image = np.full((200, 200, 3), 255, np.uint8)
        draw_sparkle(image, (100, 100), 0.)
        changed = (image != 255).any(2)
        self.assertTrue(changed[100, 105])          # Red outline adjacent to the central white point
        self.assertFalse(changed[:50].any() or changed[150:].any())
        draw_sparkle(image, (0, 199), 0.)           # Render safely even at the screen boundary


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
                self.assertEqual(r.display_position, (200., 60.))         # Highlighter follows the raw hand position even with strong smoothing
                pen = lambda x, ph: RenderPacket((x, 120), 'pen', True, ph, '#0000FF', 2, 1., ())
                for i, x in enumerate((20, 60, 100)):
                    image = r.consume(pen(x, 'down' if i == 0 else 'move'))
                self.assertFalse(np.array_equal(image[120, int(r.display_position[0]) - 1], bg[120, 1]))  # The line remains visibly connected to the cursor
            finally:
                r.close()


class CustomMappingTests(unittest.TestCase):
    """Verify custom gesture mappings from the settings UI work in practice."""

    def controller(self, **mapping):
        settings = Settings()
        settings.mapping.update(mapping)
        return UIController(Calibration([[0, 0], [1, 0], [1, 1], [0, 1]], (100, 100)), settings)

    def test_legacy_clap_page_mapping_never_turns_page(self):
        c = self.controller(DOUBLE_CLAP='page_next')
        c.update(InputState((.5, .5), navigation_allowed=False), 0)
        p = c.update(InputState((.5, .5), frozenset({'DOUBLE_CLAP'}), navigation_allowed=False), .1)
        self.assertEqual(p.commands, ())
        self.assertEqual(c.update(InputState(None, frozenset({'DOUBLE_CLAP'}), navigation_allowed=False), .2).commands, ())  # Cooldown prevents repeated input

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
        flicker = c.update(InputState((.35, .5)), .2)          # One-frame gesture detection gap
        self.assertTrue(flicker.active)
        np.testing.assert_allclose(flicker.position, (29.7, 49.5), atol=.01)   # Ink stays frozen at the previous position (.3, .5) during the gap
        resumed = c.update(InputState((.4, .5), point), .3)
        self.assertEqual(resumed.phase, 'move')                # Continue the same stroke
        c.update(InputState((.4, .5)), .4)
        self.assertFalse(c.update(InputState((.4, .5)), .7).active)   # Stroke ends after the 0.2 s grace period

    def test_far_move_during_flicker_starts_new_stroke(self):
        c = self.controller(POINTER='draw', PINCH='none')
        c.writing_scale = 1.
        c.settings.ui['draw_resume_break'] = 40
        point = frozenset({'POINTER'})
        c.update(InputState((.2, .5), point), 0)
        c.update(InputState((.2, .5)), .1)
        jumped = c.update(InputState((.8, .5), point), .2)
        self.assertEqual(jumped.phase, 'up')                   # A reacquired hand 60 px away (>40 px limit) must not reconnect the line
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
        bent = {'index': 1.4, 'middle': 1., 'ring': 1., 'pinky': 1.}   # Index slightly bent (below initial extension threshold of 1.55)
        folded = {'index': 1.2, 'middle': .7, 'ring': .7, 'pinky': .7}
        for keep in (True, False):
            r = GestureRecognizer({**cfg, 'index_writing': keep})   # Do not enable index-gesture hold when drawing with pinch
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
        self.assertEqual(stopped.phase, 'up')                 # Lift pen immediately without gesture-grace delay
        moving = c.update(InputState((.6, .5), two), .2)
        self.assertFalse(moving.active)                       # Do not draw while moving with two fingers extended
        self.assertIsNotNone(moving.position)                 # The cursor still follows
        self.assertEqual(c.update(InputState((.6, .5), point), .25).phase, 'down')   # Folding the middle finger starts a new stroke

    def test_recognizer_two_fingers_only_when_enabled_and_never_blocks_pinch(self):
        from airboard.gestures.gesture import GestureRecognizer
        curls = {'index': 1.8, 'middle': 1.7, 'ring': 1., 'pinky': 1.}
        tips = {'index': 1.5, 'middle': 1.4, 'ring': .6, 'pinky': .6}
        for flags, expect_two in (({'index_writing': True}, True), ({'two_fingers': True}, True), ({}, False)):
            r = GestureRecognizer({**gesture_cfg(), **flags})
            poses = [x.pose for x in feed(r, curls, tips, [.6] * 3)]          # Thumb stays separated
            self.assertEqual('TWO_FINGERS' in poses, expect_two, flags)
            after = feed(r, curls, tips, [.1] * 4)                           # When the thumb touches the index while holding the same hand pose
            self.assertTrue(after[-1].pen_down, flags)                       # Pinch detection remains available

    def test_index_writing_switches_to_pinch_without_reopening(self):
        from airboard.gestures.gesture import GestureRecognizer
        r = GestureRecognizer({**gesture_cfg(), 'index_writing': True, 'pinch_writing': False})
        bent = {'index': 1.4, 'middle': 1., 'ring': 1., 'pinky': 1.}
        folded = {'index': 1.2, 'middle': .7, 'ring': .7, 'pinky': .7}
        feed(r, bent, folded, [.25], pose='POINTER')
        kept = feed(r, bent, folded, [.25, .25])
        self.assertEqual([x.pose for x in kept], ['POINTER', 'POINTER'])      # Remain in index pose even when slightly bent
        pinch = feed(r, bent, folded, [.12] * 3)                             # When the thumb touches
        self.assertEqual(pinch[-1].pose, 'PINCH')                            # Switch to pinch (eraser) without requiring an open-hand reset

    def test_pinch_lift_freeze_only_when_pinch_draws(self):
        from airboard.gestures.gesture import GestureRecognizer
        open_ = {'index': 1., 'middle': 1., 'ring': 1., 'pinky': 1.}
        tips = {'index': .9, 'middle': .6, 'ring': .6, 'pinky': .6}
        for pinch_writing, frozen in ((True, True), (False, False)):
            r = GestureRecognizer({**gesture_cfg(), 'pinch_writing': pinch_writing})
            out = feed(r, open_, tips, [.1, .1, .1, .3, .3], xs=[100, 100, 100, 150, 200])
            xs = [o.pen_point[0] for o in out if o.pen_down]
            if frozen:                                                       # When pinch draws: stop inking on slight finger separation
                self.assertEqual(xs, [100, 100, 100])
            else:                                                            # When pinch erases: eraser position continues to follow the hand
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
        c = self.controller()                                  # Default mapping: PINCH=draw, FIST=eraser
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
            s.mapping['DOUBLE_CLAP'] = 'none'
            s.ui['dwell_seconds'] = 1.
            s.save()
            saved = json.loads((root / 'config/ui-user.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['ui'], {'dwell_seconds': 1.})
            self.assertEqual(saved['mapping']['DOUBLE_CLAP'], 'none')
            default = json.loads((root / 'config/default.json').read_text(encoding='utf-8'))
            default['ui']['writing_scale'] = .8                # If team defaults change later
            (root / 'config/default.json').write_text(json.dumps(default), encoding='utf-8')
            self.assertEqual(Settings(root).ui['writing_scale'], .8)   # Personal settings must not override new defaults


class MappingRuleTests(unittest.TestCase):
    """Valid action combinations in the settings menu."""

    def test_one_shot_and_held_gestures_offer_only_meaningful_actions(self):
        from airboard.ui.settings import allowed_actions
        self.assertEqual(set(allowed_actions('DOUBLE_CLAP')), {'toggle_highlighter','undo','redo','none'})
        for g in ('SWIPE_LEFT','SWIPE_RIGHT'):
            self.assertEqual(set(allowed_actions(g)), {'none'})
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
            self.assertEqual(s.mapping['PINCH'], 'draw')        # Keep the existing draw mapping
            with self.assertRaises(ValueError):
                s.set_action('DOUBLE_CLAP', 'eraser')           # Cannot map eraser to a one-shot gesture
            s.save()
            self.assertEqual(Settings(root).mapping['POINTER'], 'draw')
            self.assertEqual(Settings(root).mapping['PINCH'], 'draw')
            s.set_action('POINTER', 'none')
            s.set_action('PINCH', 'none')
            s.save()                                        # Can save settings even without a draw mapping
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
            s = Settings(root)                                  # Loads without errors
            self.assertEqual(s.mapping['FIST'], 'eraser')       # Falls back to defaults
            self.assertEqual(s.mapping['DOUBLE_CLAP'], 'toggle_highlighter')
            self.assertEqual(list(s.mapping.values()).count('draw'), 2)
            s.save()
            self.assertEqual(list(Settings(root).mapping.values()).count('draw'), 2)


if __name__ == '__main__':
    unittest.main()
