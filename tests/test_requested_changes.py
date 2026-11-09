import unittest
import tempfile
from pathlib import Path
import numpy as np
import pymupdf
from airboard.render import AirBoardRenderer,RenderConfig
from airboard.ui.settings import Settings
from airboard.ui.controller import UIController,InputState
from airboard.calibration import Calibration
from airboard.gestures.gesture import SwipeDetector

class RequestedChanges(unittest.TestCase):
    def test_fixed_style_and_no_horizontal_swipe(self):
        c=UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())
        c.color='#dc2626'
        p=c.update(InputState((.5,.5)),0)
        h=c.update(InputState((.5,.5),frozenset({'DOUBLE_CLAP'})),1)
        self.assertEqual(h.color,'#FFFF00')
        self.assertGreater(h.width,p.width)
        d=SwipeDetector()
        for i in range(5): self.assertIsNone(d.update((i*20,50),True,100,100,i*.05))

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

    def test_small_writing_keeps_hover_full_range(self):
        c=UIController(Calibration([[0,0],[1,0],[1,1],[0,1]],(101,101)),Settings())
        c.writing_scale=.5
        c.update(InputState((.2,.2),frozenset({'POINTER'})),0)
        p=c.update(InputState((.4,.4),frozenset({'POINTER'})),.1)
        np.testing.assert_allclose(p.position,(30,30))
        p=c.update(InputState((.8,.8)),.2)
        np.testing.assert_allclose(p.position,(80,80))
        self.assertEqual(p.phase,'up')

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
