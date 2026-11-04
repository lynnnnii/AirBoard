"""웹캠/GUI 없이 실제 PDF와 픽셀을 검사하는 회귀 테스트."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pymupdf
from airboard.render import AirBoardRenderer, RenderConfig


class RenderTests(unittest.TestCase):
    """소스 보존, 마스크 합성, 이력, 페이지와 저장을 검증한다."""
    def setUp(self) -> None:
        """색이 있는 PDF와 크기/회전이 다른 두 번째 페이지를 준비한다."""
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.pdf'
        with pymupdf.open() as doc:
            page = doc.new_page(width=320, height=180)
            page.draw_rect(page.rect, color=None, fill=(.2, .4, .6))
            page.insert_text((20, 150), 'Original slide text')
            page = doc.new_page(width=180, height=240)
            page.set_rotation(90)
            doc.save(self.source)
        self.config = RenderConfig(320, 180, 1, '#FF0000', 8, '#FFFF00', 20, .3, 24)
        self.board = AirBoardRenderer(self.source, self.config)

    def tearDown(self) -> None:
        """자원을 해제한다."""
        self.board.close()
        self.temp.cleanup()

    def stroke(self, tool: str, points: list[tuple[int, int]]) -> None:
        """한 작업을 확정한다."""
        self.board.begin_stroke(tool)
        for point in points:
            self.board.add_point(*point)
        self.board.end_stroke()

    def test_pen_dot_and_undo(self) -> None:
        """한 점도 그려지며 undo가 원본을 정확히 복원한다."""
        bg = self.board.render_page()
        self.stroke('pen', [(40, 40)])
        np.testing.assert_array_equal(self.board.render_page()[40, 40], [0, 0, 255])
        self.assertTrue(self.board.undo())
        np.testing.assert_array_equal(self.board.render_page(), bg)
        self.assertFalse(self.board.undo())

    def test_undo_redo_and_new_stroke_invalidates_redo(self) -> None:
        background = self.board.render_page()
        self.stroke('pen', [(40, 40)])
        first = self.board.render_page()
        self.stroke('pen', [(80, 80)])
        second = self.board.render_page()
        self.assertTrue(self.board.undo())
        np.testing.assert_array_equal(self.board.render_page(), first)
        self.assertTrue(self.board.undo())
        np.testing.assert_array_equal(self.board.render_page(), background)
        self.assertFalse(self.board.undo())
        self.assertTrue(self.board.redo())
        np.testing.assert_array_equal(self.board.render_page(), first)
        self.assertTrue(self.board.redo())
        np.testing.assert_array_equal(self.board.render_page(), second)
        self.assertFalse(self.board.redo())
        self.assertTrue(self.board.undo())
        self.stroke('pen', [(120, 100)])
        self.assertFalse(self.board.redo())

    def test_highlighter_preview_does_not_accumulate(self) -> None:
        """동일 획의 중복 점/반복 렌더링은 형광펜을 더 진하게 하지 않는다."""
        bg = self.board.render_page()
        self.board.begin_stroke('highlighter')
        for point in [(30, 60), (150, 60), (150, 60), (150, 60)]:
            self.board.add_point(*point)
        preview = self.board.render_page()
        for _ in range(8):
            np.testing.assert_array_equal(preview, self.board.render_page())
        expected = np.rint(np.array([0, 255, 255]) * .3 + bg[60, 100] * .7).astype(np.uint8)
        np.testing.assert_array_equal(preview[60, 100], expected)
        self.board.end_stroke()
        np.testing.assert_allclose(self.board.render_page(), preview, atol=1)

    def test_partial_eraser_preserves_pdf_and_undo(self) -> None:
        """빠른 지우개 이동도 연속해서 지우고 undo로 복원한다."""
        bg = self.board.render_page()
        self.stroke('pen', [(20, 70), (250, 70)])
        before = self.board.render_page()
        self.stroke('eraser', [(100, 50), (100, 90)])
        result = self.board.render_page()
        np.testing.assert_array_equal(result[70, 100], bg[70, 100])
        np.testing.assert_array_equal(result[70, 200], before[70, 200])
        self.board.undo()
        np.testing.assert_array_equal(self.board.render_page(), before)
        self.board.undo()
        np.testing.assert_array_equal(self.board.render_page(), bg)

    def test_pages_survive_cache_eviction(self) -> None:
        """다른 페이지를 방문해 캐시에서 빠져도 획/지우개 이력이 유지된다."""
        self.stroke('pen', [(20, 70), (250, 70)])
        self.stroke('eraser', [(100, 70)])
        first = self.board.render_page()
        self.assertTrue(self.board.change_page(1))
        self.assertEqual(self.board.operation_count(), 0)
        self.assertFalse(self.board.change_page(1))
        self.board.change_page(-1)
        np.testing.assert_array_equal(self.board.render_page(), first)
        self.assertEqual(self.board.operation_count(), 2)

    def test_tracking_loss_and_tool_change_split_strokes(self) -> None:
        """입력 소실/페이지 이동/도구 전환 시 획이 안전하게 끊긴다."""
        self.board.update((20, 20), 'pen', True)
        self.board.update(None, 'pen', True)
        self.board.update((100, 20), 'pen', True)
        self.board.update((150, 50), 'highlighter', True)
        self.board.update(None, 'highlighter', False)
        self.assertEqual(self.board.operation_count(), 3)
        self.assertFalse(self.board.add_point(5, 5))

    def test_eraser_is_independent_from_pinch(self) -> None:
        """pen_down=False인 주먹 지우개도 명시적 활성 상태에서 작동한다."""
        bg = self.board.render_page()
        self.stroke('pen', [(60, 60)])
        self.board.update((60, 60), 'eraser', False, eraser_active=True)
        self.board.update(None, 'eraser', False, eraser_active=False)
        np.testing.assert_array_equal(self.board.render_page()[60, 60], bg[60, 60])

    def test_clear_is_undoable(self) -> None:
        """전체 필기 삭제도 원본은 유지하고 undo 가능하다."""
        bg = self.board.render_page()
        self.stroke('pen', [(20, 30)])
        before = self.board.render_page()
        self.board.clear_page()
        np.testing.assert_array_equal(self.board.render_page(), bg)
        self.board.undo()
        np.testing.assert_array_equal(self.board.render_page(), before)

    def test_invalid_coordinates_end_stroke(self) -> None:
        """잘못된 좌표를 화면 가장자리로 연결하지 않는다."""
        self.board.begin_stroke('pen')
        self.board.add_point(10, 10)
        self.assertFalse(self.board.add_point(float('nan'), 20))
        self.assertFalse(self.board.add_point(300, 100))
        self.assertEqual(self.board.operation_count(), 1)

    def test_pdf_cache_reuses_page(self) -> None:
        """프레임 렌더링은 PDF 재렌더링을 수행하지 않는다."""
        with patch.object(pymupdf.Page, 'get_pixmap', side_effect=AssertionError('cache miss')):
            for _ in range(10):
                self.board.render_page()

    def test_export_all_pages_and_unicode_paths(self) -> None:
        """전체 페이지 수, 크기/회전, 픽셀과 현재 위치/원본 보존을 검사한다."""
        original_bytes = self.source.read_bytes()
        self.stroke('pen', [(40, 40)])
        first = self.board.render_page()
        self.board.change_page(1)
        second = self.board.render_page()
        png = self.board.save_png(self.root / '한글 경로' / '필기.png')
        decoded = cv2.imdecode(np.frombuffer(png.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, second)
        path = self.board.save_pdf(self.root / '한글 경로' / '결과.pdf')
        self.assertEqual(self.board.current_page, 1)
        self.assertEqual(self.source.read_bytes(), original_bytes)
        with pymupdf.open(path) as doc:
            self.assertEqual(len(doc), 2)
            self.assertEqual(tuple(doc[0].rect), tuple(self.board.doc[0].rect))
            self.assertEqual(tuple(doc[1].rect), tuple(self.board.doc[1].rect))
            pix = doc[0].get_pixmap(colorspace=pymupdf.csRGB, alpha=False)
            bgr = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)
            np.testing.assert_allclose(bgr[40, 40], first[40, 40], atol=2)
        with self.assertRaises(ValueError):
            self.board.save_pdf(self.source)


if __name__ == '__main__':
    unittest.main()
