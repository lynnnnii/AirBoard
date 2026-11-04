"""Index writing, immediate L release, and gesture-independent coordinates."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from airboard.gestures.gesture import GestureRecognizer, classify_pose, finger_curls
from airboard.ui.controller import UIController, InputState, from_gesture_result
from airboard.ui.settings import Settings
from airboard.calibration import Calibration

def hand(l_shape=False):
    lm=np.zeros((21,3),dtype=float)
    lm[:,:2]=[100,160]
    lm[0,:2]=[100,200]
    for i,x in [(5,85),(9,100),(13,115),(17,125)]:lm[i,:2]=[x,140]
    lm[8,:2]=[85,70]
    lm[2,:2]=[75,165]
    lm[3,:2]=[55,165] if l_shape else [85,150]
    lm[4,:2]=[25,165] if l_shape else [100,160]
    return lm

class IndexWritingTests(unittest.TestCase):
    def controller(self):
        return UIController(Calibration([[0,0],[1,0],[1,1],[0,1]]),Settings())

    def test_pose_rotation_scale_and_mirror(self):
        for l_shape,expected in [(False,'POINTER'),(True,'L_SHAPE')]:
            for angle in [0,.7,1.8]:
                for mirror in [-1,1]:
                    a=hand(l_shape)
                    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                    a[:,0]*=mirror
                    a[:,:2]=a[:,:2]@rot*1.5+200
                    self.assertEqual(classify_pose(a,finger_curls(a)),expected)

    def test_write_stop_resume_and_lost_hand(self):
        r=GestureRecognizer();c=self.controller()
        phases=[]
        for i,h in enumerate([[hand()],[hand()],[hand(True)],[hand(True)],[hand()],[]]):
            result=r.update(h,640,480,i*.1)
            phases.append(c.update(from_gesture_result(result,(640,480)),i*.1).phase)
        self.assertEqual(phases,['down','move','up','hover','down','up'])

    def test_stop_has_priority(self):
        c=self.controller()
        c.update(InputState((.5,.5),frozenset({'POINTER'})),0)
        p=c.update(InputState((.6,.6),frozenset({'POINTER','L_SHAPE','FIST'})),.1)
        self.assertEqual(p.phase,'up');self.assertFalse(p.active)

    def test_identical_coordinate_pipeline_for_remapped_gestures(self):
        outputs=[]
        for pose in ['PINCH','POINTER','OPEN']:
            c=self.controller();c.settings.mapping[pose]='draw'
            points=[]
            for i,p in enumerate([(200,150),(210,155),(220,160)]):
                result=SimpleNamespace(events=[pose],pose=pose,pen_down=pose=='PINCH',hands=[1],
                                       pen_point=(12,34),pointer_point=p)
                packet=c.update(from_gesture_result(result,(640,480)),i*.1)
                self.assertTrue(packet.active);points.append(packet.position)
            outputs.append(points)
        np.testing.assert_allclose(outputs[0],outputs[1]);np.testing.assert_allclose(outputs[1],outputs[2])

    def test_clap_guard_blocks_index_then_allows_highlighter(self):
        r=GestureRecognizer();c=self.controller()
        with patch.object(r.clap,'update',return_value=True):
            packet=c.update(from_gesture_result(r.update([hand(),hand()],640,480,0),(640,480)),0)
        self.assertEqual(packet.tool,'highlighter');self.assertFalse(packet.active)
        packet=c.update(from_gesture_result(r.update([hand()],640,480,.2),(640,480)),.2)
        self.assertTrue(packet.active);self.assertEqual(packet.tool,'highlighter')

    def test_default_disables_pinch_and_laser(self):
        c=self.controller()
        self.assertFalse(c.update(InputState((.5,.5),frozenset({'PINCH'})),0).active)
        self.assertNotIn('pointer',c.settings.mapping.values())
        self.assertEqual(c.settings.mapping['POINTER'],'draw')
