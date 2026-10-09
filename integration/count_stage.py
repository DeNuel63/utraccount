"""Run only in the CountGD++ Python >=3.10 environment."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'COUNTGD++' / 'src'))
from integration.common import read_clip, validate_class_map, write_json


def detect_clip(config, output, backend=None):
    from utraccount_count.adapter import CountGDAdapter
    from utraccount_count.contracts import FrameInfo
    from utraccount_count.countgdplusplus import CountGDPlusPlusBackend, write_detection_batch
    classes = validate_class_map(config['classes'])
    prompt = config['prompt'].strip()
    if prompt not in classes:
        raise ValueError('Positive prompt must be an exact shared class name')
    clip, frames = read_clip(config['clip_manifest'], classes)
    settings = config['countgd']
    if backend is None:
        checkpoint = Path(settings['checkpoint'])
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                digest.update(block)
        version = digest.hexdigest()
        backend = CountGDPlusPlusBackend(Path(settings['repository']), checkpoint, device=settings.get('device', 'cuda'))
    else:
        version = 'injected-test-backend'
    adapter = CountGDAdapter(backend, model_version=version)
    paths = []
    for frame in frames:
        index = frame['frame_index']
        batch = adapter.infer_frame(Path(frame['image']), sequence_id=clip['clip_id'], frame_index=index,
            frame=FrameInfo(frame['image'], clip['width'], clip['height'], frame['timestamp_ms']),
            prompt=prompt, confidence_threshold=config['confidence_threshold'],
            raw_output_path=output / 'raw' / ('%06d.json' % index))
        target = output / 'detections' / ('%06d.json' % index)
        write_detection_batch(batch, target)
        paths.append(str(target.resolve()))
    write_json(output / 'reference_annotations.json', dict(annotation_status='ULGF_REFERENCE_NOT_INDEPENDENTLY_VERIFIED', frames=frames))
    write_json(output / 'detection_sequence.json', dict(sequence_id=clip['clip_id'], classes=classes, batches=paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    detect_clip(json.loads(args.config.read_text(encoding='utf-8')), args.output)


if __name__ == '__main__':
    main()
