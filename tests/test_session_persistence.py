"""Verify page/edit-history restoration and separation of per-document session data."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pymupdf
from airboard.render import RenderConfig
from airboard.ui.demo import PDFRenderer
from airboard.ui.controller import RenderPacket


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        """Prepare PDFs with identical names but different contents."""
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.path=self.root/'source.pdf'
        with pymupdf.open() as doc:
            for i in range(3):
                p=doc.new_page(width=320,height=180)
                p.draw_rect(p.rect,color=None,fill=(.4,.6,.8))
            doc.save(self.path)
        self.cfg=RenderConfig(320,180,1,'#FF0000',8,'#FFFF00',20,.3,30)
        self.state=self.root/'sessions'

    def tearDown(self):
        self.temp.cleanup()

    def open(self,path=None):
        return PDFRenderer(str(path or self.path),self.cfg,state_dir=self.state)

    def test_restore_all_pages_eraser_clear_and_undo(self):
        """Verify per-page color, opacity, erasing, clearing, and last visited page."""
        original=self.path.read_bytes()
        r=self.open(); b=r.board
        b.commit_stroke([(20,50),(200,50)],'pen',color='#00FF00',width=8)
        b.commit_stroke([(20,80),(200,80)],'highlighter')
        b.commit_stroke([(90,30),(90,100)],'eraser',width=25)
        first=b.render_page()
        r.change_page(1)
        b.commit_stroke([(60,60)],'pen')
        b.clear_page()
        second=b.render_page()
        r.close()
        r=self.open(); b=r.board
        try:
            self.assertEqual(b.current_page,1)
            np.testing.assert_array_equal(b.render_page(),second)
            self.assertTrue(b.undo())
            self.assertFalse(np.array_equal(b.render_page(),second))
            b.goto_page(0)
            np.testing.assert_array_equal(b.render_page(),first)
        finally:
            r.close()
        self.assertEqual(self.path.read_bytes(),original)

    def test_close_saves_active_stroke(self):
        """Save an active stroke on shutdown even if pen-up has not occurred."""
        r=self.open()
        for p,phase in [((20,60),'down'),((150,60),'move')]:
            r.consume(RenderPacket(p,'pen',True,phase,'#FF0000',8,1.,()))
        expected=r.board.render_page()
        r.close()
        r=self.open()
        try:
            np.testing.assert_array_equal(r.board.render_page(),expected)
        finally:
            r.close()

    def test_same_filename_different_pdf_isolated(self):
        """Store distinct session states for PDFs with the same name but different contents."""
        r=self.open(); key=r.state_path
        r.change_page(1); r.close()
        other=self.root/'other/source.pdf'; other.parent.mkdir()
        with pymupdf.open() as doc:
            doc.new_page(width=320,height=180)
            doc.save(other)
        r=self.open(other)
        try:
            self.assertNotEqual(r.state_path,key)
            self.assertEqual(r.board.current_page,0)
            self.assertEqual(r.board.operation_count(),0)
        finally:
            r.close()

    def test_invalid_state_does_not_replace_live_or_saved_ink(self):
        """Reject corrupt sessions and preserve valid saves on write failures."""
        r=self.open(); b=r.board
        b.commit_stroke([(40,40)])
        b.save_session(r.state_path)
        saved=r.state_path.read_bytes()
        expected=b.render_page()
        bad=json.loads(saved); bad['current_page']=999
        bad_path=self.root/'bad.json'; bad_path.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            b.load_session(bad_path)
        np.testing.assert_array_equal(b.render_page(),expected)
        with patch('airboard.render.engine.os.replace',side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                b.save_session(r.state_path)
        self.assertEqual(r.state_path.read_bytes(),saved)
        r.close()
