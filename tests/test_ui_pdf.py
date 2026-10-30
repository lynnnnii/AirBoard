"""UI 패킷과 PDF 렌더러 연결을 실제 픽셀로 검증한다."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pymupdf
from airboard.render import RenderConfig
from airboard.ui.controller import RenderPacket
from airboard.ui.demo import PDFRenderer


class UIPDFTests(unittest.TestCase):
    def test_page_ink_eraser_and_original(self):
        """UI 필기, 페이지 왕복, 지우개와 원본 PDF 보존을 확인한다."""
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample.pdf'
            with pymupdf.open() as doc:
                page=doc.new_page(width=320,height=180)
                page.draw_rect(page.rect,color=None,fill=(.3,.5,.7))
                doc.new_page(width=180,height=240)
                doc.save(path)
            original=path.read_bytes()
            renderer=PDFRenderer(str(path),RenderConfig(320,240,1,'#FF0000',8,'#FFFF00',20,.3,30))
            board=renderer.board
            def packet(position,tool='pen',active=True,phase='move'):
                return RenderPacket(position,tool,active,phase,'#FF0000',8,1.,())
            try:
                bg=board.render_page()
                renderer.consume(packet((40,60),phase='down'))
                renderer.consume(packet((140,60)))
                renderer.consume(packet(None,active=False,phase='up'))
                ink=board.render_page()
                self.assertFalse(np.array_equal(bg,ink))
                board.change_page(1)
                self.assertEqual(board.slide_size,(180,240))
                board.change_page(-1)
                np.testing.assert_array_equal(board.render_page(),ink)
                renderer.consume(packet((90,60),'eraser',phase='down'))
                renderer.consume(packet(None,active=False,phase='up'))
                np.testing.assert_array_equal(board.render_page()[60,90],bg[60,90])
            finally:
                board.close()
            self.assertEqual(path.read_bytes(),original)
