"""Reference-inspired writing: continuous short strokes and explicit release."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from airboard.gestures.gesture import IndexStrokeGate, GestureRecognizer
from airboard.smoothing.stroke import StrokeSmoother
from airboard.smoothing.filters import OneEuroFilter
from airboard.ui.settings import Settings, ROOT
from tests.test_index_writing import hand

class ReferenceWritingTests(unittest.TestCase):
    def test_ambiguous_pose_freezes_then_resumes(self):
        gate=IndexStrokeGate()
        gate.update('POINTER',(100.,100.),0,100)
        self.assertEqual(gate.update('NONE',(180.,180.),.03,100),('POINTER',(100.,100.)))
        self.assertEqual(gate.update('POINTER',(104.,102.),.06,100),('POINTER',(104.,102.)))

    def test_grace_expires_without_extending_itself(self):
        gate=IndexStrokeGate()
        gate.update('POINTER',(100.,100.),0,100)
        for t in (.02,.04,.06): self.assertEqual(gate.update('NONE',(150.,150.),t,100)[0],'POINTER')
        self.assertEqual(gate.update('NONE',(150.,150.),.08,100)[0],'NONE')

    def test_stop_and_tools_never_use_grace(self):
        for pose in ('L_SHAPE','FIST','OPEN','PINCH'):
            gate=IndexStrokeGate();gate.update('POINTER',(100.,100.),0,100)
            self.assertEqual(gate.update(pose,(120.,120.),.01,100)[0],pose)
            self.assertEqual(gate.update('NONE',(150.,150.),.02,100)[0],'NONE')

    def test_busy_or_loss_clears_grace(self):
        for point,blocked in [(None,False),((120.,120.),True)]:
            gate=IndexStrokeGate();gate.update('POINTER',(100.,100.),0,100)
            gate.update('NONE',point,.01,100,blocked)
            self.assertEqual(gate.update('NONE',(150.,150.),.02,100)[0],'NONE')

    def test_distant_reacquisition_starts_separate_stroke(self):
        gate=IndexStrokeGate();gate.update('POINTER',(100.,100.),0,100)
        gate.update('NONE',(100.,100.),.01,100)
        self.assertEqual(gate.update('POINTER',(200.,200.),.02,100)[0],'NONE')
        self.assertEqual(gate.update('POINTER',(201.,201.),.03,100)[0],'POINTER')

    def test_short_stroke_end_and_next_stroke_preserved(self):
        s=StrokeSmoother.from_config(Settings().ui['handwriting_smoothing'])
        points=[]
        for i in range(6): points.append(s.update((100+i*2,100),True,i/30).point)
        finished=s.update((400,400),False,.3).finished
        self.assertEqual(finished,points)
        next_stroke=s.update((300,300),True,.4)
        self.assertEqual(next_stroke.stroke,[(300.,300.)])

    def test_fractional_landmarks_reach_adapter(self):
        r=GestureRecognizer();lm=hand();lm[8,0]+=.375
        result=r.update([lm],640,480,0)
        self.assertEqual(result.pointer_point[0],85.375)
        self.assertEqual(result.pen_point,result.pointer_point)

    def test_profile_migration_preserves_calibration_and_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir()
            for name in ('default.json','gestures.json'):
                (root/'config'/name).write_bytes((ROOT/'config'/name).read_bytes())
            ui=json.loads((ROOT/'config/default.json').read_text(encoding='utf-8'))['ui']
            ui['profile_version']=4;ui['handwriting_smoothing']['end_trim']=3
            mapping={'POINTER':'draw','L_SHAPE':'stop','WAVE':'toggle_highlighter'}
            (root/'config/ui-user.json').write_text(json.dumps({'ui':ui,'mapping':mapping}))
            s=Settings(root)
            self.assertEqual(s.ui['profile_version'],6)
            self.assertEqual(s.ui['handwriting_smoothing']['end_trim'],0)
            self.assertEqual(s.ui['default_corners'],ui['default_corners'])
            self.assertEqual(s.mapping,mapping)

    def test_less_lag_than_previous_profile_with_noise_suppression(self):
        raw=np.array([(50+i*3,80+(-4 if i%2==0 else 4)) for i in range(100)],float)
        old=OneEuroFilter(1.2,.008);new=OneEuroFilter(1.8,.025)
        before=np.array([old(p,i/30) for i,p in enumerate(raw)])
        after=np.array([new(p,i/30) for i,p in enumerate(raw)])
        self.assertLess(np.mean(raw[20:,0]-after[20:,0]),.7*np.mean(raw[20:,0]-before[20:,0]))
        self.assertLess(np.sqrt(np.mean((after[20:,1]-80)**2)),2.)
