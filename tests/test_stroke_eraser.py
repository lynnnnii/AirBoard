"""획 단위 지우개 회귀 테스트 (임준서 features/coordinate-ui e39f28c의 test_requested_changes.py에서 가져옴)."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pymupdf
from unittest.mock import patch
from airboard.render import AirBoardRenderer,RenderConfig


class StrokeEraserTests(unittest.TestCase):
    def test_stationary_eraser_removes_only_topmost_stroke_and_reuses_pdf_cache(self):
        """주먹 유지/작은 떨림은 겹친 다음 획을 지우지 않고 PDF 재렌더링도 하지 않는다."""
        from airboard.ui.demo import PreviewRenderer
        from airboard.ui.controller import RenderPacket
        renderer = PreviewRenderer((200,200))
        try:
            b = renderer.board
            bg = b.render_page()
            b.commit_stroke([(20,100),(180,100)], 'pen')
            first = b.render_page()
            b.commit_stroke([(100,20),(100,180)], 'highlighter')
            both = b.render_page()
            def erase(point, active=True, phase='move'):
                return renderer.consume(RenderPacket(point,'eraser',active,phase,'#000000',56,1.,()))
            with patch.object(pymupdf.Page, 'get_pixmap', side_effect=AssertionError('PDF cache miss')):
                erase((100,100), phase='down')
                for point in ((100,100),(101,99),(100,101)):
                    np.testing.assert_array_equal(erase(point),first)
                erase(None,False,'up')
                b.undo()
                np.testing.assert_array_equal(b.render_page(),both)
                erase((100,100), phase='down')
                erase(None,False,'up')
                erase((100,100), phase='down')
                np.testing.assert_array_equal(b.render_page(),bg)
        finally:
            renderer.close()

    def test_fast_path_erasure_is_saved_and_undoable_after_reopening(self):
        """커서 두 점 사이를 빠르게 지난 획도 지우고 세션에서 복구할 수 있다."""
        from airboard.ui.demo import PDFRenderer
        from airboard.ui.controller import RenderPacket
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.pdf'
            with pymupdf.open() as doc:
                doc.new_page(width=200,height=200); doc.save(path)
            cfg=RenderConfig(200,200,1,'#000000',2,'#FFFF00',16,.3,30)
            r=PDFRenderer(str(path),cfg,Path(folder)/'sessions')
            bg=r.board.render_page()
            r.board.commit_stroke([(20,100),(180,100)])
            ink=r.board.render_page()
            for point,phase in (((100,20),'down'),((100,180),'move'),(None,'up')):
                r.consume(RenderPacket(point,'eraser',point is not None,phase,'#000000',30,1.,()))
            r.close()
            r=PDFRenderer(str(path),cfg,Path(folder)/'sessions')
            try:
                np.testing.assert_array_equal(r.board.render_page(),bg)
                self.assertTrue(r.board.undo())
                np.testing.assert_array_equal(r.board.render_page(),ink)
            finally:
                r.close()

    def test_whole_stroke_erase_undo_session(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.pdf'
            with pymupdf.open() as doc:
                doc.new_page(width=200,height=200);doc.save(path)
            cfg=RenderConfig(200,200,1,'#000000',2,'#FFFF00',24,.3,30)
            with AirBoardRenderer(path,cfg) as b:
                b.commit_stroke([(20,100),(180,100)])
                first=b.render_page()
                b.commit_stroke([(100,20),(100,180)],'highlighter')
                both=b.render_page()
                self.assertTrue(b.erase_stroke_at((100,100)))
                np.testing.assert_array_equal(b.render_page(),first)
                session=Path(folder)/'state.json';b.save_session(session)
                b.load_session(session)
                np.testing.assert_array_equal(b.render_page(),first)
                self.assertTrue(b.undo())
                np.testing.assert_array_equal(b.render_page(),both)

    def test_eraser_continues_without_releasing_fist(self):
        from airboard.ui.demo import PDFRenderer
        from airboard.ui.controller import RenderPacket
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.pdf'
            with pymupdf.open() as doc:
                doc.new_page(width=200,height=200);doc.save(path)
            cfg=RenderConfig(200,200,1,'#000000',2,'#FFFF00',24,.3,30)
            renderer=PDFRenderer(str(path),cfg)
            try:
                bg=renderer.board.render_page()
                renderer.board.commit_stroke([(30,40),(70,40)])
                renderer.board.commit_stroke([(130,140),(170,140)])
                renderer.consume(RenderPacket((50,40),'eraser',True,'down','#000000',56,1.,()))
                self.assertFalse(np.array_equal(renderer.board.render_page(),bg))
                renderer.consume(RenderPacket((150,140),'eraser',True,'move','#000000',56,1.,()))
                np.testing.assert_array_equal(renderer.board.render_page(),bg)
            finally:
                renderer.close()


if __name__ == '__main__':
    unittest.main()
