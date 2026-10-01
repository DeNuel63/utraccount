import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('experiment_runner',ROOT/'run_five_frames.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class FakeAdapter:
    def __init__(self,fail=False):self.calls=[];self.fail=fail
    def generate(self,frame,seed,steps,guidance):
        self.calls.append((frame.frame_index,seed,steps,guidance))
        if self.fail:raise RuntimeError('Injected generation failure')
        return Image.new('RGB',(256,256),'blue'),'fixture prompt',False

class FiveFrameTests(unittest.TestCase):
    def test_frozen_snapshot_unchanged(self):
        # The frozen copy, not the evolving/cleaned live repository, is the
        # experiment's source authority. verify_snapshot checks every file hash.
        digest=runner.verify_snapshot()
        data=json.loads((ROOT/'baseline_snapshot.json').read_text())
        self.assertTrue(data['files'])
        self.assertEqual(len(digest),64)

    def test_five_frames_same_seed_unverified_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run';adapter=FakeAdapter();manifest=runner.layout()
            result=runner.generate(manifest,output,adapter)
            self.assertEqual(len(adapter.calls),5)
            self.assertEqual({r[1] for r in adapter.calls},{0})
            self.assertEqual(len(list((output/'frames').glob('*.png'))),5)
            self.assertEqual(result['status'],'INFERENCE_COMPLETE_REVIEW_REQUIRED')
            ids=[]
            for path in sorted((output/'annotations').glob('*.json')):
                objs=json.loads(path.read_text())['objects']
                ids.append([o['instance_id'] for o in objs])
                self.assertTrue(all(o['bbox_verified_xyxy'] is None for o in objs))
            self.assertTrue(all(v==ids[0] for v in ids))

    def test_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            with self.assertRaises(RuntimeError):runner.generate(runner.layout(),output,FakeAdapter(True))
            self.assertEqual(json.loads((output/'run.json').read_text())['status'],'FAILED')

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):runner.generate(runner.layout(),Path(temp),FakeAdapter())

    def test_output_restricted_to_experiment(self):
        with self.assertRaises(ValueError):runner.safe_output(ROOT.parents[1]/'outputs/normal')
        with self.assertRaises(ValueError):runner.safe_output(ROOT/'runs')
        with self.assertRaises(ValueError):runner.safe_output(ROOT/'runs/checkpoint/new',[ROOT/'runs/checkpoint'])

    def test_missing_checkpoint_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):runner.checkpoint_record(Path(temp))

if __name__=='__main__':unittest.main()
