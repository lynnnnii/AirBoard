"""팀 StrokeSmoother와 렌더러 연결 회귀 테스트. GUI/웹캠이 필요 없다."""
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
    """미리보기 교체와 꼬리 제거, 페이지 전환 시 필기 보존을 검사한다."""

    def setUp(self) -> None:
        """작은 유색 PDF와 팀 보정기를 연결한다."""
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
        """자원을 해제한다."""
        self.board.close()
        self.temp.cleanup()

    def test_final_tail_replaces_live_preview(self) -> None:
        """마지막 세 점이 미리보기에는 보이지만 최종 레이어에서는 제거된다."""
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
        """공유 리스트를 외부에서 바꿔도 현재 프리뷰에 영향이 없고 교체 시 꼬리가 사라진다."""
        points = [(20, 40), (100, 40)]
        self.board.set_preview(points)
        before = self.board.render_page()
        points.append((200, 40))
        np.testing.assert_array_equal(before, self.board.render_page())
        self.board.set_preview([(20, 40), (40, 40)])
        self.board.discard_preview()
        self.assertEqual(self.board.operation_count(), 0)

    def test_finished_uses_original_tool(self) -> None:
        """완료 시 UI 도구가 바뀌어도 시작 시 형광펜 스타일을 보존한다."""
        points = [(30, 60), (100, 60)]
        bg = self.board.render_page()
        self.bridge.consume(SimpleNamespace(stroke=points, finished=None), InkStyle('highlighter'))
        self.bridge.consume(SimpleNamespace(stroke=None, finished=points), InkStyle('pen'))
        expected = np.rint(np.array([0, 255, 255]) * .3 + bg[60, 60] * .7)
        np.testing.assert_allclose(self.board.render_page()[60, 60], expected, atol=1)

    def test_very_short_skipped_stroke_leaves_no_ink(self) -> None:
        """시작 skip 안에 끝난 동작은 빈 획/undo 항목을 남기지 않는다."""
        bg = self.board.render_page()
        self.bridge.consume(self.smoother.update((30, 50), True, 0), self.style)
        self.bridge.consume(self.smoother.update((40, 50), False, .03), self.style)
        np.testing.assert_array_equal(self.board.render_page(), bg)
        self.assertEqual(self.board.operation_count(), 0)

    def test_page_switch_finalizes_and_requires_lift(self) -> None:
        """페이지 이동 전 최종 획 저장, 이후 누른 채 입력은 새 페이지에 기록하지 않는다."""
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
        """저장/페이지 이동 직전 지우개 동작이 프리뷰 제거로 유실되지 않는다."""
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
        """포인터와 손 소실은 영구 필기를 만들지 않는다."""
        session = RenderSession(self.board, self.smoother, NoFilter())
        session.feed((80, 60), False, 0, temporary_tool='pointer')
        session.feed(None, False, .1)
        self.assertEqual(self.board.operation_count(), 0)
