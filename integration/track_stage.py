"""Run in COVTrack's existing MMCV/MMDetection environment."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'COUNTGD++' / 'src'))
sys.path.insert(0, str(ROOT / 'CovTrack_Model' / 'COVTrack'))
from integration.common import validate_class_map, write_json


def load_batches(sequence):
    # Count package needs Python >=3.10, so COVTrack consumes its JSON without
    # importing that package into its legacy environment.
    batches = [json.loads(Path(p).read_text(encoding='utf-8')) for p in sequence['batches']]
    for index, batch in enumerate(batches):
        if batch['schema_version'] != 'utraccount.detection-batch/v1' or batch['frame_index'] != index or batch['sequence_id'] != sequence['sequence_id']:
            raise ValueError('Invalid detection sequence')
        if batch['source']['box_format'] != 'xywh_center' or batch['source']['coordinate_space'] != 'pixel':
            raise ValueError('Unsupported detection coordinates')
        ids = set()
        for d in batch['detections']:
            if d['label'] not in sequence['classes'] or d['detection_id'] in ids:
                raise ValueError('Unknown detection label or duplicate ID')
            ids.add(d['detection_id'])
    if not batches:
        raise ValueError('An empty sequence cannot be tracked')
    return batches


def tracking_input(batch, classes):
    boxes, labels = [], []
    for detection in batch['detections']:
        x, y, w, h = detection['bbox_xywh_center']
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2, detection['confidence']])
        labels.append(classes[detection['label']])
    return dict(sequence_id=batch['sequence_id'], frame_index=batch['frame_index'],
        width_px=batch['frame']['width_px'], height_px=batch['frame']['height_px'], boxes=boxes, labels=labels)


def track_clip(config, output, model=None, infer=None):
    sequence = json.loads((output / 'detection_sequence.json').read_text(encoding='utf-8'))
    classes = validate_class_map(config['classes'])
    if sequence['classes'] != classes:
        raise ValueError('Detector and tracker class mappings differ')
    batches = load_batches(sequence)
    if model is None:
        # The checkout is imported under its existing namespace package name.
        from utraccount.CovTrack_Model.COVTrack.ovtrack.apis.inference import init_model, inference_model
        settings = config['covtrack']
        options = dict(settings.get('cfg_options', {}))
        if settings.get('custom_vocabulary', False):
            if sorted(classes.values()) != list(range(len(classes))):
                raise ValueError('Custom vocabulary requires consecutive IDs from zero')
            from utraccount.CovTrack_Model.COVTrack.ovtrack.models.roi_heads import ovtrack_roi_head
            ovtrack_roi_head.text_input = [name for name, index in sorted(classes.items(), key=lambda item: item[1])]
            options.update({'model.roi_head.custom_classes': True,
                'model.roi_head.only_validation_categories': False,
                'model.roi_head.only_test_categories': False})
        model = init_model(settings['config'], settings['checkpoint'], device=settings.get('device', 'cuda:0'), cfg_options=options)
        infer = inference_model
    vocabulary = tuple(model.roi_head.CLASSES)
    if any(index >= len(vocabulary) or vocabulary[index] != name for name, index in classes.items()):
        raise ValueError('Shared classes do not match COVTrack roi_head.CLASSES; supply its actual vocabulary IDs')
    results = []
    reverse = {v: k for k, v in classes.items()}
    for batch in batches:
        result = infer(model, batch['frame']['uri'], batch['frame_index'], external_detections=tracking_input(batch, classes))
        objects = []
        for label_id, rows in enumerate(result['track_results']):
            for row in rows:
                identity, x1, y1, x2, y2, score = row.tolist()
                objects.append(dict(track_id=int(identity), class_id=label_id, label=reverse[label_id],
                    bbox_xyxy_pixel=[x1, y1, x2, y2], confidence=score))
        frame = dict(sequence_id=batch['sequence_id'], frame_index=batch['frame_index'],
            image=batch['frame']['uri'], timestamp_ms=batch['frame']['timestamp_ms'], tracks=objects)
        write_json(output / 'tracks' / ('%06d.json' % batch['frame_index']), frame)
        results.append(frame)
    write_json(output / 'tracking_sequence.json', dict(schema_version='utraccount.tracking-sequence/v1',
        sequence_id=sequence['sequence_id'], classes=classes, frames=results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    track_clip(json.loads(args.config.read_text(encoding='utf-8')), args.output)


if __name__ == '__main__':
    main()
