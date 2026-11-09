"""Regression tests against actual PDF pixels without a webcam or GUI."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pymupdf
from airboard.render import AirBoardRenderer, RenderConfig
from airboard.render.engine import color_bgr


class RenderTests(unittest.TestCase):
    """Verify source preservation, mask compositing, history, pages, and exports."""
    def setUp(self) -> None:
        """Create a colored PDF with a second page of different size and rotation."""
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
        """Release resources."""
        self.board.close()
        self.temp.cleanup()

    def stroke(self, tool: str, points: list[tuple[int, int]]) -> None:
        """Commit one operation."""
        self.board.begin_stroke(tool)
        for point in points:
            self.board.add_point(*point)
        self.board.end_stroke()

    def test_pen_dot_and_undo(self) -> None:
        """Single points render correctly and undo restores the exact original."""
        bg = self.board.render_page()
        self.stroke('pen', [(40, 40)])
        np.testing.assert_array_equal(self.board.render_page()[40, 40], [0, 0, 255])
        self.assertTrue(self.board.undo())
        np.testing.assert_array_equal(self.board.render_page(), bg)
        self.assertFalse(self.board.undo())

    def full_preview(self) -> np.ndarray:
        """Reference result: blend the whole in-progress stroke every frame (the previous render_page)."""
        board = self.board
        canvas = board._canvas(board.current_page)
        result = canvas.composed.copy()
        active = board._active
        if active is not None and active.bounds is not None:
            x0, y0, x1, y1 = active.bounds
            region = np.s_[y0:y1, x0:x1]
            a = active.mask[region].astype(np.float32)[..., None] / 255.0 * active.opacity
            if active.stroke.tool == 'eraser':
                value = canvas.color[region] * (1 - a) + canvas.background[region] * (1 - canvas.alpha[region] * (1 - a))
            else:
                value = np.array(color_bgr(active.stroke.color), np.float32) * a + canvas.composed[region] * (1 - a)
            result[region] = np.rint(value).clip(0, 255).astype(np.uint8)
        return result

    def test_incremental_preview_matches_full_blend(self) -> None:
        """Blending only newly drawn segments gives exactly the same frame as blending the whole stroke."""
        self.stroke('pen', [(30, 30), (200, 120)])
        self.stroke('highlighter', [(20, 100), (290, 60)])
        path = [(20 + i * 9, 90 + round(50 * np.sin(i / 3))) for i in range(30)]
        path += path[::-1]                                  # go back over the same area
        for tool in ('pen', 'highlighter', 'eraser'):
            self.board.begin_stroke(tool)
            for i, point in enumerate(path):
                self.board.add_point(*point)
                if i % 3:                                   # some frames add several points before rendering
                    continue
                np.testing.assert_array_equal(self.board.render_page(), self.full_preview())
            np.testing.assert_array_equal(self.board.render_page(), self.full_preview())
            self.board.end_stroke()

    def test_preview_copy_is_not_shared(self) -> None:
        """Drawing UI on a returned frame must not leak into the next frame."""
        self.board.begin_stroke('pen')
        self.board.add_point(40, 40)
        frame = self.board.render_page()
        frame[:] = 0
        self.assertFalse(np.array_equal(self.board.render_page(), frame))

    def test_highlighter_preview_does_not_accumulate(self) -> None:
        """Duplicate points and repeated rendering must not darken the same highlighter stroke."""
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
        """Fast eraser movement removes connected strokes and can be undone."""
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
        """Stroke and eraser history survives cache eviction after navigating pages."""
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
        """Strokes end safely on tracking loss, page changes, and tool changes."""
        self.board.update((20, 20), 'pen', True)
        self.board.update(None, 'pen', True)
        self.board.update((100, 20), 'pen', True)
        self.board.update((150, 50), 'highlighter', True)
        self.board.update(None, 'highlighter', False)
        self.assertEqual(self.board.operation_count(), 3)
        self.assertFalse(self.board.add_point(5, 5))

    def test_eraser_is_independent_from_pinch(self) -> None:
        """Fist erasing works when explicitly active, even if pen_down=False."""
        bg = self.board.render_page()
        self.stroke('pen', [(60, 60)])
        self.board.update((60, 60), 'eraser', False, eraser_active=True)
        self.board.update(None, 'eraser', False, eraser_active=False)
        np.testing.assert_array_equal(self.board.render_page()[60, 60], bg[60, 60])

    def test_clear_is_undoable(self) -> None:
        """Clearing all ink preserves the source and remains undoable."""
        bg = self.board.render_page()
        self.stroke('pen', [(20, 30)])
        before = self.board.render_page()
        self.board.clear_page()
        np.testing.assert_array_equal(self.board.render_page(), bg)
        self.board.undo()
        np.testing.assert_array_equal(self.board.render_page(), before)

    def test_invalid_coordinates_end_stroke(self) -> None:
        """Invalid positions must not connect lines to the screen edge."""
        self.board.begin_stroke('pen')
        self.board.add_point(10, 10)
        self.assertFalse(self.board.add_point(float('nan'), 20))
        self.assertFalse(self.board.add_point(300, 100))
        self.assertEqual(self.board.operation_count(), 1)

    def test_pdf_cache_reuses_page(self) -> None:
        """Rendering an output frame must not re-render the underlying PDF."""
        with patch.object(pymupdf.Page, 'get_pixmap', side_effect=AssertionError('cache miss')):
            for _ in range(10):
                self.board.render_page()

    def test_export_all_pages_and_unicode_paths(self) -> None:
        """Check page count, size/rotation, pixels, and preservation of the current page/source."""
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
