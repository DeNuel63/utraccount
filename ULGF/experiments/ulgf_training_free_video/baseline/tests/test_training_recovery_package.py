import ast
import hashlib
import json
from pathlib import Path
import unittest
import zipfile
from tools.recover_training_prerequisites import DETECTION, TRAINING, TORCH
from tools.consolidated_preflight import CONTRACTS, TEST_SHA

ROOT=Path(__file__).resolve().parents[1]


class RecoveryPackageTests(unittest.TestCase):
    def test_exact_v11_recovery_evidence(self):
        with zipfile.ZipFile(ROOT/'colab/ULGF_recovery_and_checks.zip') as z:
            manifest=json.loads(z.read('recovery_manifest.json'))
        self.assertEqual(manifest['base_contracts'],CONTRACTS)
        self.assertEqual(manifest['official_test_sha256'],TEST_SHA)
        self.assertEqual(sum(map(len,manifest['records'].values())),12557)
        for records in manifest['records'].values():
            for record in records.values():
                self.assertEqual(len(record['file_sha256']),64)
                self.assertEqual(len(record['stored_pixels_sha256']),64)

    def test_cell_binds_archive_and_rejects_wrong_package(self):
        blob=(ROOT/'colab/ULGF_recovery_and_checks.zip').read_bytes()
        source=(ROOT/'colab/ULGF_recovery_and_checks_cell.py').read_text(encoding='utf-8')
        ast.parse(source)
        self.assertIn(hashlib.sha256(blob).hexdigest(),source)
        self.assertLess(source.index('hexdigest()'),source.index('extractall'))

    def test_no_cv2_or_coco_distribution_overlap(self):
        names=[p.split('==')[0] for p in DETECTION+TRAINING+TORCH]
        self.assertEqual(len(names),len(set(names)))
        self.assertNotIn('opencv-python-headless',names)
        self.assertNotIn('mmpycocotools',names)
        self.assertNotIn('mmcv',names)

    def test_bundle_python_is_37_syntax_compatible(self):
        with zipfile.ZipFile(ROOT/'colab/ULGF_recovery_and_checks.zip') as z:
            for name in z.namelist():
                if name.endswith('.py'): ast.parse(z.read(name),feature_version=(3,7))


if __name__=='__main__': unittest.main()
