import json
import shutil
import tempfile
import unittest
from pathlib import Path
from tools.check_ruod_v11_coverage import check
from tools.apply_ruod_visual_families import digest_file

class CoverageTests(unittest.TestCase):
    def test_real_v11_accounting(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'report'
            check(repo/'outputs/ruod-v11-refresh-base',repo/'outputs/ruod-v11-local-refresh-inputs',repo.parent/'RUOD',out)
            r=json.loads((out/'coverage_checks.json').read_text())
            self.assertEqual(r['exact_cross_split_pixel_leakage_groups'],0)
            self.assertEqual(r['excluded_images'],1443)
            self.assertTrue(r['all_classes_present'])
            self.assertFalse(r['training_ready'])

    def test_simulated_exact_leakage_rejected(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            bundle=Path(tmp)/'bundle'; shutil.copytree(repo/'outputs/ruod-v11-local-refresh-inputs',bundle)
            base=repo/'outputs/ruod-v11-refresh-base'
            train=json.loads((base/'annotations/instances_train.json').read_text())['images'][0]['id']
            test=json.loads((base/'annotations/instances_test.json').read_text())['images'][0]['id']
            p=bundle/'exact_pixel_hashes.json'; hashes=json.loads(p.read_text())
            hashes['train:'+str(train)]=hashes['test:'+str(test)]; p.write_text(json.dumps(hashes))
            spec=json.loads((bundle/'spec.json').read_text()); spec['payload_hashes'][p.name]=digest_file(p)
            (bundle/'spec.json').write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'Exact leakage'):
                check(base,bundle,repo.parent/'RUOD',Path(tmp)/'report')

if __name__=='__main__': unittest.main()
