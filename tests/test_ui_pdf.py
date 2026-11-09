"""Verify UI packet integration with the PDF renderer using actual pixel output."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from airboard.ui import demo
import numpy as np
import pymupdf
from airboard.render import RenderConfig
from airboard.ui.controller import RenderPacket
from airboard.ui.demo import PDFRenderer


class UIPDFTests(unittest.TestCase):
    def test_page_ink_eraser_and_original(self):
        """Check UI drawing, page navigation, erasing, and preservation of the original PDF."""
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


    def test_input_end_keeps_pdf_navigation_and_quit(self):
        """After video ends or camera reading fails, keep the window, page navigation, and quit controls functional."""
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample.pdf'
            with pymupdf.open() as doc:
                doc.new_page(width=320,height=180)
                doc.new_page(width=320,height=180)
                doc.save(path)
            for source in (['--video','missing.mp4'], []):
                with self.subTest(source=source):
                    cap=MagicMock()
                    cap.isOpened.return_value=True
                    cap.read.return_value=(False,None)
                    captured=[]
                    real_renderer=demo.PDFRenderer
                    def make_renderer(*args,**kwargs):
                        renderer=real_renderer(*args,state_dir=Path(folder)/('state-video' if source else 'state-camera'),smoothing_config=kwargs.get('smoothing_config'))
                        renderer.board.close=MagicMock(wraps=renderer.board.close)
                        captured.append(renderer)
                        return renderer
                    with patch('sys.argv',['demo','--pdf',str(path)]+source), \
                         patch('cv2.VideoCapture',return_value=cap), \
                         patch('airboard.input.camera.open_camera',return_value=cap), \
                         patch('airboard.tracking.hand_tracker.HandTracker') as tracker, \
                         patch('cv2.namedWindow'),patch('cv2.resizeWindow'), \
                         patch('cv2.setMouseCallback'),patch('cv2.imshow') as show, \
                         patch('cv2.getWindowProperty',return_value=1), \
                         patch('cv2.destroyAllWindows'), \
                         patch('cv2.waitKeyEx',side_effect=[2621440,ord('q')]) as wait, \
                         patch.object(demo,'PDFRenderer',side_effect=make_renderer):
                        demo.main()
                        self.assertEqual(wait.call_count,2)
                        self.assertEqual(show.call_count,2)
                        self.assertEqual(captured[0].board.current_page,1)
                        cap.read.assert_called_once()
                        tracker.return_value.detect.assert_not_called()
                        captured[0].board.close.assert_called_once()
                        cap.release.assert_called_once()
