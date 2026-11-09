"""Regression tests for the team's StrokeSmoother/renderer bridge without GUI or webcam."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import pymupdf
from airboard.render import AirBoardRenderer, RenderConfig, InkStyle, SmoothedInkBridge
from airboard.render.session import RenderSession
from airboard.smoothing.stroke import StrokeSmoother
from airboard.smoothing.filters import NoFilter


class BridgeTests(unittest.TestCase):
    """Verify preview replacement, tail trimming, and ink preservation across page changes."""

    def setUp(self) -> None:
        """Connect the team's smoother to a small colored PDF."""
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / 'source.pdf'
        with pymupdf.open() as doc:
            for _ in range(2):
                p = doc.new_page(width=320, height=180)
                p.draw_rect(p.rect, color=None, fill=(.8, .9, 1))
            doc.save(path)
        cfg = RenderConfig(320, 180, 2, '#FF0000', 4, '#FFFF00', 20, .3, 24)
        self.board = AirBoardRenderer(path, cfg)
        self.bridge = SmoothedInkBridge(self.board)
        self.smoother = StrokeSmoother(NoFilter(), 2, 3)
        self.style = InkStyle('pen')

    def tearDown(self) -> None:
        """Release resources."""
        self.board.close()
        self.temp.cleanup()

    def test_final_tail_replaces_live_preview(self) -> None:
        """The last three points appear in preview but are trimmed from the final ink layer."""
        bg = self.board.render_page()
        for i in range(10):
            result = self.smoother.update((20 + i * 10, 70), True, i / 30)
            self.bridge.consume(result, self.style)
        self.assertFalse(np.array_equal(self.board.render_page()[70, 100], bg[70, 100]))
        final = self.smoother.update((200, 70), False, 10 / 30)
        self.bridge.consume(final, self.style)
        self.assertEqual([p[0] for p in final.finished], [40, 50, 60, 70, 80])
        np.testing.assert_array_equal(self.board.render_page()[70, 100], bg[70, 100])
        self.assertEqual(self.board.operation_count(), 1)
        for i in range(3):
            self.bridge.consume(self.smoother.update((200, 70), False, (11 + i) / 30), self.style)
        self.assertEqual(self.board.operation_count(), 1)

    def test_preview_is_copied_and_can_shrink(self) -> None:
        """External mutation of a shared list must not affect preview; replacing it removes the tail."""
        points = [(20, 40), (100, 40)]
        self.board.set_preview(points)
        before = self.board.render_page()
        points.append((200, 40))
        np.testing.assert_array_equal(before, self.board.render_page())
        self.board.set_preview([(20, 40), (40, 40)])
        self.board.discard_preview()
        self.assertEqual(self.board.operation_count(), 0)

    def test_finished_uses_original_tool(self) -> None:
        """Preserve initial highlighter style even when the UI tool changes before completion."""
        points = [(30, 60), (100, 60)]
        bg = self.board.render_page()
        self.bridge.consume(SimpleNamespace(stroke=points, finished=None), InkStyle('highlighter'))
        self.bridge.consume(SimpleNamespace(stroke=None, finished=points), InkStyle('pen'))
        expected = np.rint(np.array([0, 255, 255]) * .3 + bg[60, 60] * .7)
        np.testing.assert_allclose(self.board.render_page()[60, 60], expected, atol=1)

    def test_very_short_skipped_stroke_leaves_no_ink(self) -> None:
        """A stroke ending within start_skip must not leave an empty stroke or undo entry."""
        bg = self.board.render_page()
        self.bridge.consume(self.smoother.update((30, 50), True, 0), self.style)
        self.bridge.consume(self.smoother.update((40, 50), False, .03), self.style)
        np.testing.assert_array_equal(self.board.render_page(), bg)
        self.assertEqual(self.board.operation_count(), 0)

    def test_page_switch_finalizes_and_requires_lift(self) -> None:
        """Finalize ink before changing pages and do not draw on the next page while pen remains down."""
        session = RenderSession(self.board, self.smoother, NoFilter())
        for i in range(10):
            session.feed((20 + i * 10, 70), True, i / 30)
        session.change_page(1, .5)
        self.assertEqual(self.board.operation_count(0), 1)
        for i in range(10):
            session.feed((30 + i * 10, 70), True, .6 + i / 30)
        self.assertEqual(self.board.operation_count(1), 0)
        session.feed((30, 70), False, 1)
        for i in range(10):
            session.feed((30 + i * 10, 70), True, 1.1 + i / 30)
        session.flush(1.6)
        self.assertEqual(self.board.operation_count(1), 1)

    def test_eraser_survives_flush_and_undo(self) -> None:
        """Eraser actions must not be lost when previews are cleared before save/page changes."""
        bg = self.board.render_page()
        self.board.commit_stroke([(20, 60), (150, 60)])
        ink = self.board.render_page()
        session = RenderSession(self.board, self.smoother, NoFilter())
        session.feed((80, 60), False, 0, temporary_tool='eraser', eraser_active=True)
        session.flush(.1)
        self.assertEqual(self.board.operation_count(), 2)
        np.testing.assert_array_equal(self.board.render_page()[60, 80], bg[60, 80])
        self.board.undo()
        np.testing.assert_array_equal(self.board.render_page(), ink)

    def test_pointer_and_hand_loss_do_not_draw(self) -> None:
        """Pointer use and tracking loss must not create permanent ink."""
        session = RenderSession(self.board, self.smoother, NoFilter())
        session.feed((80, 60), False, 0, temporary_tool='pointer')
        session.feed(None, False, .1)
        self.assertEqual(self.board.operation_count(), 0)
