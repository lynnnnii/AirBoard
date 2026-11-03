"""획 단위 지우개 회귀 테스트 (임준서 features/coordinate-ui e39f28c의 test_requested_changes.py에서 가져옴)."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pymupdf
from airboard.render import AirBoardRenderer,RenderConfig


class StrokeEraserTests(unittest.TestCase):
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
