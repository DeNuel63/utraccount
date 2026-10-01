import json
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from tools.audit_ruod_dataset import audit, file_hash, pixel_hash, safe_path, valid_box, strip_jpeg_exif


class AuditTests(unittest.TestCase):
    def test_xmp_orientation_removed_without_recompression(self):
        stream = io.BytesIO()
        Image.new('RGB', (12, 8), 'green').save(stream, format='JPEG', progressive=True)
        original = stream.getvalue()
        payloads = [
            b'http://ns.adobe.com/xap/1.0/\x00'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
            b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
            b'tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>',
            b'http://ns.adobe.com/xmp/extension/\x00test-data',
        ]
        segments = b''.join(b'\xff\xe1' + (len(p) + 2).to_bytes(2, 'big') + p for p in payloads)
        tagged = original[:2] + segments + original[2:]
        cleaned = strip_jpeg_exif(tagged)
        self.assertEqual(cleaned, original)  # All non-XMP bytes, including scan data, unchanged.
        self.assertEqual(strip_jpeg_exif(cleaned), cleaned)
        with Image.open(io.BytesIO(tagged)) as before, Image.open(io.BytesIO(cleaned)) as after:
            self.assertEqual(pixel_hash(before), pixel_hash(after))
            self.assertFalse(after.getexif())

    def test_decode_failures_and_duplicate_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            for split in ('train', 'test'):
                images = source / 'RUOD_pic' / split
                images.mkdir(parents=True)
                Image.new('RGB', (8, 8), 'red').save(images / 'a.jpg')
                (images / 'broken.jpg').write_bytes(b'not an image')
                data = dict(images=[dict(id=1, file_name='a.jpg', width=8, height=8),
                                    dict(id=2, file_name='broken.jpg', width=8, height=8)],
                            annotations=[], categories=[])
                annotations = source / 'RUOD_ANN'
                annotations.mkdir(exist_ok=True)
                (annotations / ('instances_' + split + '.json')).write_text(json.dumps(data))
            report = audit(source, Path(tmp) / 'audit')
            self.assertEqual(report['quarantined_images'], 2)
            self.assertEqual(report['train_test_exact_pixel_leakage_groups'], 1)
            self.assertFalse(report['training_ready'])

    def test_bounds_and_traversal(self):
        self.assertTrue(valid_box([0, 712, 587, 311], (683, 1024)))
        self.assertFalse(valid_box([0, 712, 587, 311], (1024, 683)))
        self.assertFalse(valid_box([0, 0, float('nan'), 2], (5, 5)))
        with self.assertRaises(ValueError):
            safe_path(Path('images'), '../outside.jpg')

    def test_audit_export_quarantine_and_resume(self):
        for orientation in (0, 6):
            with self.subTest(orientation=orientation):
                self.check_audit_export(orientation)

    def check_audit_export(self, orientation):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            images = source / 'RUOD_pic/train'
            images.mkdir(parents=True)
            annotation_dir = source / 'RUOD_ANN'
            annotation_dir.mkdir()
            Image.new('RGB', (8, 12), 'red').save(images / 'a.jpg')
            exif = Image.Exif()
            exif[274] = orientation
            Image.new('RGB', (8, 12), 'blue').save(images / 'b.jpg', exif=exif)
            data = dict(images=[dict(id=1, file_name='a.jpg', width=8, height=12),
                                dict(id=2, file_name='b.jpg', width=12, height=8)],
                        categories=[dict(id=1, name='fish')],
                        annotations=[dict(id=i, image_id=i, category_id=1,
                                          bbox=[0, 2, 5, 8], area=40) for i in (1, 2)])
            (annotation_dir / 'instances_train.json').write_text(json.dumps(data))
            original_hash = file_hash(images / 'b.jpg')
            output = Path(tmp) / 'out'
            report = audit(source, output, True)
            self.assertEqual(report['quarantined_images'], 1)
            self.assertTrue((output / 'images/train/1.jpg').is_file())
            self.assertFalse((output / 'images/train/2.jpg').exists())
            audit(source, output, True)  # resumes and verifies exports
            approved_output = Path(tmp) / 'approved'
            report = audit(source, approved_output, True, {'train:2': original_hash})
            self.assertEqual(report['quarantined_images'], 0)
            with Image.open(approved_output / 'images/train/2.jpg') as img:
                self.assertEqual(img.size, (8, 12))
                self.assertFalse(img.getexif())
                with Image.open(images / 'b.jpg') as original:
                    self.assertEqual(pixel_hash(img), pixel_hash(original))
            self.assertEqual(original_hash, file_hash(images / 'b.jpg'))


if __name__ == '__main__':
    unittest.main()
