"""엄지척 좌우 제스처가 필기 Undo/Redo로만 연결되는지 검증한다."""
import unittest
import numpy as np
from airboard.gestures.gesture import GestureRecognizer, ThumbSwipeDetector, is_thumb_up, sideways_thumb_pose
from airboard.calibration import Calibration
from airboard.ui.controller import UIController, InputState, from_gesture_result
from airboard.ui.settings import Settings, allowed_actions


def thumb_hand(x=130.0):
    """검지~새끼는 접고 엄지만 손바닥 축 방향으로 세운 합성 랜드마크."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[:, 0] = x
    hand[:, 1] = 70
    hand[0, :2] = (x, 100)       # wrist
    hand[5, :2] = (x - 15, 60)  # index MCP
    hand[9, :2] = (x, 65)       # middle MCP
    hand[13, :2] = (x + 15, 60)
    hand[17, :2] = (x + 25, 65)
    hand[2, :2] = (x - 25, 60)  # thumb MCP
    hand[4, :2] = (x - 20, 15)  # extended thumb tip
    for tip in (8, 12, 16, 20):
        hand[tip, :2] = (x + 5, 75)
    return hand


class ThumbGestureTests(unittest.TestCase):
    def test_detects_thumb_but_not_folded_thumb(self):
        hand = thumb_hand()
        self.assertTrue(is_thumb_up(hand))
        hand[4, :2] = hand[2, :2]
        self.assertFalse(is_thumb_up(hand))

    def test_swipe_left_and_right_single_shot(self):
        detector = ThumbSwipeDetector()
        for t, x in ((0, 240), (.1, 210)):
            self.assertIsNone(detector.update((x, 100), True, 640, 480, t))
        self.assertEqual(detector.update((130, 100), True, 640, 480, .2), 'THUMB_LEFT')
        self.assertIsNone(detector.update((280, 100), True, 640, 480, 1.0))
        detector.update(None, False, 640, 480, 1.1)
        detector.update(None, False, 640, 480, 1.2)
        for t, x in ((1.3, 130), (1.4, 180)):
            self.assertIsNone(detector.update((x, 100), True, 640, 480, t))
        self.assertEqual(detector.update((240, 100), True, 640, 480, 1.5), 'THUMB_RIGHT')

    def test_vertical_motion_does_not_trigger(self):
        detector = ThumbSwipeDetector()
        for t, y in ((0, 80), (.1, 145), (.2, 240)):
            self.assertIsNone(detector.update((200, y), True, 640, 480, t))

    def test_recognizer_emits_thumb_events(self):
        recognizer = GestureRecognizer()
        events = []
        for t, x in ((0, 300), (.1, 260), (.2, 200)):
            result = recognizer.update([thumb_hand(x)], 640, 480, t)
            events.extend(result.events)
            self.assertFalse(result.pen_down)
        self.assertEqual(events, ['THUMB_LEFT'])

    def test_sideways_thumb_is_prioritized_over_fist(self):
        left = thumb_hand()
        left[4, :2] = (65, 65)
        left[3, :2] = (92, 62)
        right = thumb_hand()
        right[4, :2] = (220, 65)
        right[3, :2] = (180, 62)
        self.assertEqual(sideways_thumb_pose(left), 'THUMB_LEFT')
        self.assertEqual(sideways_thumb_pose(right), 'THUMB_RIGHT')
        recognizer = GestureRecognizer({'history_hold_frames': 2})
        events = []
        for t in (0, .1, .2, .3):
            result = recognizer.update([left], 640, 480, t)
            events.extend(result.events)
            self.assertNotEqual(result.pose, 'FIST')
            self.assertFalse(result.pen_down)
        self.assertEqual(events, ['THUMB_LEFT'])
        # 자세를 유지하는 동안 재입력하지 않는다.
        self.assertEqual(recognizer.update([left], 640, 480, .4).events, [])

    def test_eraser_recently_used_blocks_accidental_undo(self):
        settings = Settings()
        controller = UIController(Calibration([[0, 0], [1, 0], [1, 1], [0, 1]], (100, 100)), settings)
        controller.update(InputState((.5, .5), frozenset({'FIST'})), 0)
        controller.update(InputState((.5, .5), frozenset({'FIST'})), .1)
        controller.update(InputState((.5, .5)), .2)
        self.assertEqual(controller.update(InputState((.5, .5), frozenset({'THUMB_LEFT'})), .3).commands, ())
        self.assertEqual(controller.update(InputState((.5, .5), frozenset({'THUMB_LEFT'})), 1.0).commands, ('undo',))

    def test_maps_to_ink_history_not_pages(self):
        settings = Settings()
        self.assertEqual(settings.mapping['THUMB_LEFT'], 'undo')
        self.assertEqual(settings.mapping['THUMB_RIGHT'], 'redo')
        self.assertIn('undo', allowed_actions('THUMB_LEFT'))
        self.assertNotIn('undo', allowed_actions('FIST'))
        controller = UIController(Calibration([[0,0],[1,0],[1,1],[0,1]], (100,100)), settings)
        left = InputState((.5,.5), frozenset({'THUMB_LEFT'}))
        right = InputState((.5,.5), frozenset({'THUMB_RIGHT'}))
        self.assertEqual(controller.update(left, 0).commands, ('undo',))
        self.assertEqual(controller.update(right, 1).commands, ('redo',))
        self.assertEqual(controller.update(InputState((.5,.5), frozenset({'SWIPE_LEFT'})), 2).commands, ())


if __name__ == '__main__':
    unittest.main()
