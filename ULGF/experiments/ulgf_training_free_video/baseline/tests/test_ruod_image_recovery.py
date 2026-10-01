import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from restore_ruod_images import restore_one
from audit_ruod_dataset import file_hash, pixel_hash


class RecoveryTests(unittest.TestCase):
    def example(self, root, mode='RGB'):
        source = root / 'original.jpg'
        image = Image.new(mode, (24, 16))
        exif = Image.Exif()
        exif[274] = 6
        image.save(source, exif=exif)
        with Image.open(source) as check:
            record = dict(file_sha256=file_hash(source), stored_pixels_sha256=pixel_hash(check))
        return source, record, dict(id=1, width=24, height=16)

    def test_jpeg_orientation_removed_and_resumable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, record, item = self.example(root)
            before = source.read_bytes()
            target = root / 'derived/1.jpg'
            self.assertEqual(restore_one(source, target, record, item), 'restored')
            self.assertEqual(restore_one(source, target, record, item), 'verified_existing')
            self.assertEqual(source.read_bytes(), before)
            with Image.open(target) as check:
                self.assertEqual(check.size, (24, 16))
                self.assertFalse(check.getexif())

    def test_cmyk_uses_clean_png(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, record, item = self.example(root, 'CMYK')
            self.assertEqual(restore_one(source, root / 'derived/1.png', record, item), 'restored')

    def test_changed_original_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, record, item = self.example(root)
            record['file_sha256'] = 'wrong'
            target = root / 'derived/1.jpg'
            with self.assertRaisesRegex(ValueError, 'Original hash'):
                restore_one(source, target, record, item)
            self.assertFalse(target.exists())

    def test_wrong_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, record, item = self.example(root)
            item['width'] = 16
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                restore_one(source, root / 'derived/1.jpg', record, item)
