"""Restore only reviewed-split images; never change annotations or originals."""
import argparse
import io
import json
from pathlib import Path
import shutil
from PIL import Image
from audit_ruod_dataset import file_hash, pixel_hash, strip_jpeg_exif, export_filename, safe_path
from apply_ruod_visual_families import contract


def restore_one(source, destination, record, item):
    blob = source.read_bytes()
    import hashlib
    if hashlib.sha256(blob).hexdigest() != record['file_sha256']:
        raise ValueError('Original hash mismatch: ' + str(source))
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        expected_name = export_filename(item['id'], image.mode, image.format)
        if destination.name != expected_name or image.size != (item['width'], item['height']):
            raise ValueError('Derived filename/dimensions mismatch: ' + str(source))
        expected = record['stored_pixels_sha256']
        if pixel_hash(image) != expected:
            raise ValueError('Source decoded pixels differ from audit: ' + str(source))
        def validate(path):
            with Image.open(path) as check:
                if check.mode != 'RGB' or check.getexif() or pixel_hash(check) != expected:
                    raise ValueError('Derived image differs: ' + str(path))
        if destination.exists():
            validate(destination)
            return 'verified_existing'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination.parent).free < image.width * image.height * 6 + 512 * 1024**2:
            raise OSError('Insufficient disk space; verified exports can be resumed.')
        temporary = destination.with_suffix('.recovery.tmp')
        if expected_name.endswith('.jpg'):
            temporary.write_bytes(strip_jpeg_exif(blob))
        else:
            rgb = image.convert('RGB')
            Image.frombytes('RGB', rgb.size, rgb.tobytes()).save(temporary, format='PNG', compress_level=1)
        validate(temporary)
        temporary.replace(destination)
        return 'restored'


def restore(source, base, bundle, output):
    spec = json.loads((bundle / 'recovery_manifest.json').read_text())
    output = output.resolve()
    if output == source.resolve() or source.resolve() in output.parents or output in source.resolve().parents:
        raise ValueError('Output must be separate from originals')
    jobs = []
    test = base / 'annotations/instances_test.json'
    if file_hash(test) != spec['official_test_sha256']:
        raise ValueError('Official test hash mismatch')
    for origin, expected in spec['source_annotation_hashes'].items():
        if file_hash(source / 'RUOD_ANN' / ('instances_' + origin + '.json')) != expected:
            raise ValueError('Original annotation hash mismatch: ' + origin)
    for split in ('train', 'validation', 'test'):
        data = json.loads((base / ('annotations/instances_' + split + '.json')).read_text())
        if contract(data) != spec['base_contracts'][split]:
            raise ValueError('Recovery base contract mismatch: ' + split)
        origin = 'test' if split == 'test' else 'train'
        for item in data['images']:
            record = spec['records'][origin][str(item['id'])]
            if Path(item['file_name']).parts[0] != origin:
                raise ValueError('Unexpected image origin')
            jobs.append((safe_path(source / 'RUOD_pic' / origin, record['file_name']),
                         safe_path(output, item['file_name']), record, item))
    if len({str(job[1]) for job in jobs}) != len(jobs):
        raise ValueError('Duplicate export destination')
    paths = json.loads((base / 'paths.json').read_text())
    if Path(paths['image_prefix']).resolve() != output:
        raise ValueError('Image root differs from split paths')
    counts = {'restored': 0, 'verified_existing': 0}
    for index, job in enumerate(jobs, 1):
        counts[restore_one(*job)] += 1
        if index % 250 == 0:
            print('Images: {}/{}'.format(index, len(jobs)), flush=True)
    print(json.dumps(dict(status='DERIVED_IMAGE_RECOVERY_PASS', required=len(jobs),
                         counts=counts, membership_changed=False, training_started=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'base', 'bundle', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    restore(args.source, args.base, args.bundle, args.output)
