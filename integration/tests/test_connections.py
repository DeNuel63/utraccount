import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'COUNTGD++/src'))
from integration.common import read_clip, validate_class_map, write_json
from integration.count_stage import detect_clip
from integration.track_stage import track_clip, tracking_input
from integration.run_pipeline import execute
from utraccount_count.backend import RawDetection, RawInference


class Connections(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        clip_root = self.root / 'clip'
        clip_root.mkdir()
        for i in range(3):
            Image.new('RGB', (100, 60)).save(clip_root / ('%06d.png' % i))
        self.manifest = clip_root / 'manifest.json'
        self.clip = dict(schema_version='1.0', clip_id='clip', width=100, height=60, fps=10,
            frames=[dict(frame_index=i, image_path='%06d.png' % i, objects=[dict(instance_id=77,
                class_id=0, class_name='fish', bbox_xyxy=[.1, .2, .5, .8])]) for i in range(3)])
        write_json(self.manifest, self.clip)
        self.output = self.root / 'output'
        self.config = dict(classes={'fish': 0}, prompt='fish', confidence_threshold=.2,
            clip_manifest=str(self.manifest), countgd={}, covtrack={})

    def test_clip_detector_tracker_handoff_with_empty_middle_frame(self):
        backend = Mock()
        detection = RawDetection((30, 30, 40, 36), 'fish', .9, 0)
        backend.infer.side_effect = [RawInference([detection], {'frame': 0}), RawInference([], {'frame': 1}), RawInference([detection], {'frame': 2})]
        detect_clip(self.config, self.output, backend)
        model = types.SimpleNamespace(roi_head=types.SimpleNamespace(CLASSES=['fish']))
        seen = []
        def infer(model, image, index, external_detections):
            seen.append(external_detections)
            rows = [[42] + row for row in external_detections['boxes']]
            return {'track_results': [np.asarray(rows).reshape(-1, 6)]}
        track_clip(self.config, self.output, model, infer)
        self.assertEqual(seen[0]['boxes'][0][:4], [10, 12, 50, 48])
        self.assertEqual(seen[1]['boxes'], [])
        self.assertNotIn('instance_id', seen[0])
        result = json.loads((self.output / 'tracking_sequence.json').read_text())
        self.assertEqual([f['timestamp_ms'] for f in result['frames']], [0, 100, 200])
        self.assertEqual(result['frames'][0]['tracks'][0]['track_id'], 42)
        self.assertEqual(result['frames'][1]['tracks'], [])
        refs = json.loads((self.output / 'reference_annotations.json').read_text())
        self.assertEqual(refs['frames'][0]['references'][0]['instance_id'], 77)
        batch = json.loads((self.output / 'detections/000000.json').read_text())
        self.assertIsNone(batch['detections'][0]['ground_truth_instance_id'])

    def test_rejects_bad_frame_order_and_class_mapping(self):
        self.clip['frames'][1]['frame_index'] = 2
        write_json(self.manifest, self.clip)
        with self.assertRaises(ValueError):
            read_clip(self.manifest, {'fish': 0})
        with self.assertRaises(ValueError):
            validate_class_map({'fish': 0, 'turtle': 0})

    def test_rejects_wrong_dimensions_and_reference_coordinates(self):
        self.clip['width'] = 99
        write_json(self.manifest, self.clip)
        with self.assertRaises(ValueError):
            read_clip(self.manifest, {'fish': 0})
        self.clip['width'] = 100
        self.clip['frames'][0]['objects'][0]['bbox_xyxy'][2] = 1.1
        write_json(self.manifest, self.clip)
        with self.assertRaises(ValueError):
            read_clip(self.manifest, {'fish': 0})

    def test_tracker_rejects_vocabulary_mismatch(self):
        detect_clip(self.config, self.output, Mock(infer=Mock(return_value=RawInference([], {}))))
        with self.assertRaises(ValueError):
            track_clip(self.config, self.output, types.SimpleNamespace(roi_head=types.SimpleNamespace(CLASSES=['turtle'])), Mock())

    def runtime_config(self):
        checkpoint = self.root / 'checkpoint.pth'
        checkpoint.touch()
        (self.root / 'app.py').touch()
        covconfig = self.root / 'cov.py'
        covconfig.touch()
        self.config.update(output=str(self.output), countgd=dict(python=sys.executable,
            checkpoint=str(checkpoint), repository=str(self.root)), covtrack=dict(python=sys.executable,
            checkpoint=str(checkpoint), config=str(covconfig)))
        return self.config

    def test_runner_invokes_separate_interpreters_and_retains_failure(self):
        config = self.runtime_config()
        launch = Mock(side_effect=[None, RuntimeError('tracker failure')])
        with self.assertRaises(RuntimeError):
            execute(config, launch=launch)
        self.assertEqual(launch.call_count, 2)
        self.assertIn('count_stage.py', launch.call_args_list[0].args[0][1])
        self.assertIn('track_stage.py', launch.call_args_list[1].args[0][1])
        status = json.loads((self.output / 'status.json').read_text())
        self.assertEqual(status['status'], 'FAILED')
        self.assertFalse(status['model_execution_verified'])

    def test_dry_run_writes_nothing_and_rejects_existing_output(self):
        config = self.runtime_config()
        self.assertEqual(execute(config, dry_run=True)['status'], 'PREFLIGHT_PASS')
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            execute(config, dry_run=True)


class Tensor(np.ndarray):
    """Minimal CPU tensor double; these tests do not qualify torch/MMCV."""
    def size(self, axis):
        return self.shape[axis]
    @property
    def device(self):
        return 'cpu'
    def new_tensor(self, value):
        return np.asarray(value).view(Tensor)
    def unsqueeze(self, axis):
        return np.expand_dims(self, axis).view(Tensor)
    def squeeze(self, axis=None):
        return np.ndarray.squeeze(self, axis=axis).view(Tensor)


class ExternalEntry(unittest.TestCase):
    def setUp(self):
        torch = types.ModuleType('torch')
        torch.float32, torch.long = np.float32, np.int64
        torch.as_tensor = lambda value, dtype, device: np.asarray(value, dtype=dtype).view(Tensor)
        torch.isfinite = np.isfinite
        functional = types.ModuleType('torch.nn.functional')
        functional.normalize = lambda value, p, dim: value / np.linalg.norm(value, axis=dim, keepdims=True)
        nn = types.ModuleType('torch.nn')
        nn.functional = functional
        torch.nn = nn
        core = types.ModuleType('mmdet.core')
        core.bbox2result = lambda boxes, labels, count: boxes
        core.bbox2roi = lambda boxes: boxes
        package = types.ModuleType('external_test')
        package.__path__ = []
        apis = types.ModuleType('external_test.apis')
        apis.__path__ = []
        tracking = types.ModuleType('external_test.core')
        tracking.track2result = Mock(return_value=['tracked'])
        modules = {'torch': torch, 'torch.nn': nn, 'torch.nn.functional': functional,
            'mmdet.core': core, 'external_test': package, 'external_test.apis': apis, 'external_test.core': tracking}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location('external_test.apis.external_detections', ROOT / 'CovTrack_Model/COVTrack/ovtrack/apis/external_detections.py')
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        head = types.SimpleNamespace(num_classes=1, cem_head=None, _track_forward=Mock(return_value='appearance'))
        self.model = types.SimpleNamespace(roi_head=head, method='ovtrack-teta', extract_feat=Mock(return_value='feature-map'), rpn_head=Mock())
        self.model.init_tracker = Mock(side_effect=lambda: setattr(self.model, 'tracker', types.SimpleNamespace(match=Mock(return_value=('boxes', 'labels', 'ids')))))
        self.image = np.zeros((1, 3, 60, 100)).view(Tensor)
        self.meta = [dict(ori_shape=(60, 100, 3), scale_factor=[2, 2, 2, 2], flip=False)]

    def payload(self, index=0, sequence='one', empty=False):
        return dict(sequence_id=sequence, frame_index=index, width_px=100, height_px=60,
            boxes=[] if empty else [[10, 12, 50, 48, .9]], labels=[] if empty else [0])

    def test_roi_scaling_external_boxes_and_native_association(self):
        self.module.track_external_detections(self.model, self.image, self.meta, self.payload())
        roi_boxes = self.model.roi_head._track_forward.call_args.args[1][0]
        np.testing.assert_array_equal(roi_boxes, [[20, 24, 100, 96]])
        match = self.model.tracker.match.call_args.kwargs
        np.testing.assert_array_equal(match['bboxes'][:, :4], [[10, 12, 50, 48]])
        self.assertEqual(match['embeds'], 'appearance')
        self.model.rpn_head.assert_not_called()

    def test_empty_frames_order_and_sequence_reset(self):
        self.module.track_external_detections(self.model, self.image, self.meta, self.payload(empty=True))
        self.model.extract_feat.assert_not_called()
        self.module.track_external_detections(self.model, self.image, self.meta, self.payload(index=1, empty=True))
        with self.assertRaises(ValueError):
            self.module.track_external_detections(self.model, self.image, self.meta, self.payload(index=3))
        self.module.track_external_detections(self.model, self.image, self.meta, self.payload(sequence='two', empty=True))
        self.assertEqual(self.model.init_tracker.call_count, 2)

    def test_flipped_frame_and_invalid_box_rejected(self):
        self.meta[0]['flip'] = True
        with self.assertRaises(ValueError):
            self.module.track_external_detections(self.model, self.image, self.meta, self.payload())
        self.meta[0]['flip'] = False
        value = self.payload()
        value['boxes'][0][2] = 101
        with self.assertRaises(ValueError):
            self.module.track_external_detections(self.model, self.image, self.meta, value)

    def test_uncertainty_semantic_and_location_heads_are_reused(self):
        head = self.model.roi_head
        head.use_cls_static_ratio = head.use_motion_static_ratio = True
        head.ensemble = True
        features = np.asarray([[1., 2.]]).view(Tensor)
        head._track_forward.return_value = features
        head._bbox_forward = Mock(return_value=(None, features))
        head._bbox_forward_for_image = Mock(return_value=(None, features))
        head.projection = head.projection_for_image = lambda value: value
        head.cls_head = Mock(return_value=features)
        head.loc_head = Mock(return_value=features)
        head.stog = Mock(side_effect=lambda key, ref: (key, ref))
        head_module = types.ModuleType('external_test.models.roi_heads.ovtrack_roi_head')
        head_module.normalize_detections = Mock(return_value='normalized locations')
        with patch.dict(sys.modules, {'external_test.models.roi_heads.ovtrack_roi_head': head_module}):
            appearance, semantic = self.module.extract_external_features(head, 'feature-map',
                np.asarray([[10., 12., 50., 48., .9]]).view(Tensor), self.meta[0])
        head._bbox_forward.assert_called_once()
        head._bbox_forward_for_image.assert_called_once()
        head.cls_head.assert_called_once()
        head.loc_head.assert_called_once_with('normalized locations')
        head.stog.assert_called_once()
        np.testing.assert_allclose(appearance, [[3., 6.]])
        np.testing.assert_allclose(semantic, features / np.linalg.norm(features))

    def test_uncertainty_fusion_head_is_attached_after_each_reset(self):
        head = self.model.roi_head
        head.use_cls_static_ratio = head.use_motion_static_ratio = True
        head.fusion_head = 'existing fusion'
        head.track_head = types.SimpleNamespace(loss_cyc='existing consistency')
        def reset():
            self.model.tracker = types.SimpleNamespace(set_fusion_head=Mock())
        self.model.init_tracker.side_effect = reset
        self.module.track_external_detections(self.model, self.image, self.meta, self.payload(empty=True))
        self.model.tracker.set_fusion_head.assert_called_once_with('existing fusion', 'existing consistency')


if __name__ == '__main__':
    unittest.main()
